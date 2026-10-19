import ee

PROJECT_ID = '#####'

try:
    ee.Initialize(project=PROJECT_ID)
except Exception:
    ee.Authenticate()
    ee.Initialize(project=PROJECT_ID)

gaul = ee.FeatureCollection('FAO/GAUL/2015/level2')
roi = gaul.filter(ee.Filter.And(
    ee.Filter.eq('ADM1_NAME', 'Sumatera Utara'),
    ee.Filter.eq('ADM2_NAME', 'Tapanuli Tengah')
)).geometry()

SCALE = 30
TARGET_CRS = 'EPSG:32647'
EPS = 1e-6


EROSION_PERCENTILE = 67

target_proj = ee.Projection(TARGET_CRS).atScale(SCALE)


def to_grid(image, method='bilinear'):
    return image.resample(method).reproject(target_proj)


dem_raw = ee.Image('NASA/NASADEM_HGT/001').select('elevation').clip(roi)
dem = to_grid(dem_raw, 'bilinear').rename('elevation').toFloat().unmask(0)

slope = ee.Terrain.slope(dem).rename('slope').toFloat().unmask(0)
aspect = ee.Terrain.aspect(dem).rename('aspect').toFloat().unmask(0)

slope_rad = slope.multiply(3.14159265 / 180.0)
tan_slope = slope_rad.tan().max(EPS)
flow_acc_raw = ee.Image('MERIT/Hydro/v1_0_1').select('upa').clip(roi)
flow_acc = to_grid(flow_acc_raw, 'bilinear')

twi = flow_acc.divide(tan_slope).add(EPS).log().rename('twi').toFloat().unmask(0)
spi = flow_acc.multiply(tan_slope).add(EPS).log().rename('spi').toFloat().unmask(0)

chirps_raw = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY') \
    .filterDate('2023-01-01', '2023-12-31') \
    .select('precipitation') \
    .sum() \
    .clip(roi)
rainfall = to_grid(chirps_raw, 'bilinear').rename('rainfall').toFloat().unmask(0)

L = float(SCALE)

zx_kernel = ee.Kernel.fixed(3, 3, [[0, 0, 0], [-1 / (2 * L), 0, 1 / (2 * L)], [0, 0, 0]])
zy_kernel = ee.Kernel.fixed(3, 3, [[0, 1 / (2 * L), 0], [0, 0, 0], [0, -1 / (2 * L), 0]])
zxx_kernel = ee.Kernel.fixed(3, 3, [[0, 0, 0], [1 / L**2, -2 / L**2, 1 / L**2], [0, 0, 0]])
zyy_kernel = ee.Kernel.fixed(3, 3, [[0, 1 / L**2, 0], [0, -2 / L**2, 0], [0, 1 / L**2, 0]])
zxy_kernel = ee.Kernel.fixed(3, 3, [[-1 / (4 * L**2), 0, 1 / (4 * L**2)],
                                     [0, 0, 0],
                                     [1 / (4 * L**2), 0, -1 / (4 * L**2)]])

zx = dem.convolve(zx_kernel).rename('zx')
zy = dem.convolve(zy_kernel).rename('zy')
zxx = dem.convolve(zxx_kernel).rename('zxx')
zyy = dem.convolve(zyy_kernel).rename('zyy')
zxy = dem.convolve(zxy_kernel).rename('zxy')

deriv_stack = ee.Image.cat([zx, zy, zxx, zyy, zxy])

profile_curv = deriv_stack.expression(
    '-(zxx*zx*zx + 2*zxy*zx*zy + zyy*zy*zy) / ((zx*zx + zy*zy + eps) * pow(1 + zx*zx + zy*zy, 1.5))',
    {'zx': zx, 'zy': zy, 'zxx': zxx, 'zyy': zyy, 'zxy': zxy, 'eps': EPS}
).rename('profile_curvature').toFloat().unmask(0)

plan_curv = deriv_stack.expression(
    '-(zxx*zy*zy - 2*zxy*zx*zy + zyy*zx*zx) / pow(zx*zx + zy*zy + eps, 1.5)',
    {'zx': zx, 'zy': zy, 'zxx': zxx, 'zyy': zyy, 'zxy': zxy, 'eps': EPS}
).rename('plan_curvature').toFloat().unmask(0)

erosion_index = rainfall.multiply(tan_slope).rename('erosion_index')

erosion_threshold = ee.Number(
    erosion_index.reduceRegion(
        reducer=ee.Reducer.percentile([EROSION_PERCENTILE]),
        geometry=roi,
        scale=SCALE,
        maxPixels=1e9,
        bestEffort=True,
    ).get('erosion_index')
)

erosion_class = erosion_index.gte(erosion_threshold) \
    .rename('erosion_class').toFloat().unmask(0)

raster_stack = ee.Image.cat([
    dem, slope, aspect, twi, spi, rainfall, plan_curv, profile_curv,
    erosion_class,
])

print("Erosion threshold (67th percentile of index):", erosion_threshold.getInfo())
print("Bands to export:", raster_stack.bandNames().getInfo())

task = ee.batch.Export.image.toDrive(
    image=raster_stack,
    description='Tapanuli_Tengah_Erosion_Predictors',
    folder='GEE_Exports',
    fileNamePrefix='tapanuli_tengah_erosion_predictors_30m',
    scale=SCALE,
    crs=TARGET_CRS,
    region=roi,
    maxPixels=1000000000,
)

task.start()
