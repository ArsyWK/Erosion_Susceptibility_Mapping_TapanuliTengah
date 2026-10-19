import json
import pathlib
import numpy as np
import pandas as pd
import rioxarray
import skgstat as skg
from rasterio.transform import xy

GIS_RASTER = pathlib.Path(r"C:\Users\freya\Downloads\GIS\tapa\Erosion\tapanuli_tengah_erosion_predictors_30m.tif")
BAND_NAMES = ["dem", "slope", "aspect", "twi", "spi", "rainfall",
              "plan_curvature", "profile_curvature", "erosion_class"]
TARGET_BAND = "erosion_class"
PREDICTOR_BANDS = [b for b in BAND_NAMES if b != TARGET_BAND]

SAMPLE_SIZE = 3000
MODEL = "spherical"
N_LAGS = 15
MAXLAG = "median"
RANDOM_STATE = 42
SUMMARY_METHOD = "max"


def load_bands(raster_path, band_names):

    raster = rioxarray.open_rasterio(raster_path)
    n_bands = raster.shape[0]
    if n_bands != len(band_names):
        raise ValueError(f"Raster has {n_bands} bands but BAND_NAMES has {len(band_names)} entries")

    transform = raster.rio.transform()
    rows, cols = raster.shape[1], raster.shape[2]
    flat = raster.values.reshape(n_bands, -1).T
    df = pd.DataFrame(flat, columns=band_names)

    valid_mask = ~(df[band_names] == 0).all(axis=1)
    valid_mask &= np.isfinite(df[band_names]).all(axis=1)

    row_idx, col_idx = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    eastings, northings = xy(transform, row_idx.flatten(), col_idx.flatten())
    coords_all = np.column_stack([eastings, northings])

    df = df[valid_mask].reset_index(drop=True)
    coords = coords_all[valid_mask.values]
    return df, coords


def fit_variogram_range(values, coords, sample_size, model, n_lags, maxlag, seed):
    rng = np.random.default_rng(seed)
    if len(values) > sample_size:
        idx = rng.choice(len(values), size=sample_size, replace=False)
        values = values[idx]
        coords = coords[idx]

    V = skg.Variogram(
        coordinates=coords, values=values,
        model=model, n_lags=n_lags, maxlag=maxlag,
    )
    fitted_range = V.parameters[0]
    return fitted_range, V


def main():
    df, coords = load_bands(GIS_RASTER, BAND_NAMES)
    print(f"{len(df)} valid ROI pixels loaded\n")

    results = {}
    for band in PREDICTOR_BANDS:
        values = df[band].values.astype(float)
        try:
            fitted_range, V = fit_variogram_range(
                values, coords, SAMPLE_SIZE, MODEL, N_LAGS, MAXLAG, RANDOM_STATE
            )
            results[band] = float(fitted_range)
            print(f"  range: {fitted_range:8.1f} m   sill: {V.parameters[1]:.4f}   "
                  f"nugget: {V.parameters[2]:.4f}   rmse: {V.rmse:.4f}")
        except Exception as e:
            print(f"variogram fit failed'{band}',{e}")

    if not results:
        raise RuntimeError("No variograms ")

    ranges = np.array(list(results.values()))
    median_range = float(np.median(ranges))
    max_range = float(np.max(ranges))
    max_band = max(results, key=results.get)

    for band, r in sorted(results.items(), key=lambda kv: kv[1]):
        print(f"  {band:20s},{r:8.1f} m")

    print(f"\nMedian range across {len(results)} bands: {median_range:.1f} m")
    print(f"Max range across {len(results)} bands:    {max_range:.1f} m  (from '{max_band}')")

    suggested_buffer_radius = max_range if SUMMARY_METHOD == "max" else median_range

    out_path = pathlib.Path("variogram_ranges.json")
    out_path.write_text(json.dumps(
        {
            "per_band_range_m": results,
            "median_range_m": median_range,
            "max_range_m": max_range,
            "max_range_band": max_band,
            "model": MODEL,
            "sample_size": SAMPLE_SIZE,
        },
        indent=2,
    ))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()