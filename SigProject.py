import pathlib
import joblib
import numpy as np
import pandas as pd
import rasterio
import rioxarray
import verde as vd
from rasterio.transform import xy
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV

GIS_RASTER = pathlib.Path(r"C:\Users\freya\Downloads\GIS\tapa\Erosion\tapanuli_tengah_erosion_predictors_30m.tif")
BAND_NAMES = ["dem", "slope",'aspect', "twi","spi", "rainfall",'plan_curvature','profile_curvature','erosion_class']
TARGET_BAND = "erosion_class"
PREDICTOR_BANDS = [b for b in BAND_NAMES if b != TARGET_BAND]

OUTER_SPLITS = 5
INNER_SPLITS = 4
BLOCK_SPACING = 5000
BUFFER_RADIUS = 7424


def load_raster_dataset(raster_path, band_names, target_band):
    raster = rioxarray.open_rasterio(raster_path)   # (bands, row count, cols count)
    n_bands = raster.shape[0]
    if n_bands != len(band_names):
        raise ValueError(f"Raster has {n_bands} bands but BAND_NAMES has {len(band_names)} entries")

    transform = raster.rio.transform()
    crs = raster.rio.crs
    if crs is not None and crs.is_geographic:
        print("WARNING")

    array = raster.values                     # (bands, rows, cols)
    rows, cols = array.shape[1], array.shape[2]
    flat = array.reshape(n_bands, -1).T        # (n_pixels, n_bands)
    df = pd.DataFrame(flat, columns=band_names)

    valid_mask = ~(df[band_names] == 0).all(axis=1)
    valid_mask &= np.isfinite(df[band_names]).all(axis=1)

    row_idx, col_idx = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    longitude, latitude = xy(transform, row_idx.flatten(), col_idx.flatten())
    coords_all = np.column_stack([longitude, latitude])

    predictor_cols = [b for b in band_names if b != target_band]
    X = df.loc[valid_mask, predictor_cols].values
    y = df.loc[valid_mask, target_band].values
    coords = coords_all[valid_mask.values]

    print(f"{len(y)} valid ROI pixels "
          f"({int((y == 1).sum())} positive / {int((y == 0).sum())} negative)")

    return X, y, coords, predictor_cols, raster, transform, crs, valid_mask.values



def apply_buffer(train_idx, test_idx, coords, buffer_radius):
    if buffer_radius <= 0:
        return train_idx, test_idx

    train_coords = coords[train_idx]
    test_coords = coords[test_idx]

    test_tree = cKDTree(test_coords)
    dist_to_test, _ = test_tree.query(train_coords, k=1)
    keep_train = dist_to_test >= buffer_radius
    train_idx_clean = train_idx[keep_train]

    remaining_train_coords = coords[train_idx_clean]
    if len(remaining_train_coords) == 0:
        return train_idx_clean, test_idx
    train_tree = cKDTree(remaining_train_coords)
    dist_to_train, _ = train_tree.query(test_coords, k=1)
    keep_test = dist_to_train >= buffer_radius
    test_idx_clean = test_idx[keep_test]

    return train_idx_clean, test_idx_clean


def nested_spatial_split(X, y, coords, outer_splits, inner_splits, block_spacing,
                          buffer_radius, random_state=42):
    outer_cv = vd.BlockKFold(spacing=block_spacing, n_splits=outer_splits,
                              shuffle=True, random_state=random_state)
    trainval_idx, test_idx = list(outer_cv.split(coords))[0]
    trainval_idx, test_idx = apply_buffer(trainval_idx, test_idx, coords, buffer_radius)

    X_test, y_test = X[test_idx], y[test_idx]
    X_trainval, y_trainval = X[trainval_idx], y[trainval_idx]
    coords_trainval = coords[trainval_idx]

    inner_cv = vd.BlockKFold(spacing=block_spacing, n_splits=inner_splits,
                              shuffle=True, random_state=random_state)
    inner_folds = [
        apply_buffer(tr, va, coords_trainval, buffer_radius)
        for tr, va in inner_cv.split(coords_trainval)
    ]

    return X_trainval, y_trainval, X_test, y_test, inner_folds


def tune_and_evaluate(X_trainval, y_trainval, inner_folds):
    param_grid = {
        "n_estimators": [100, 150, 180],
        "max_depth": [None, 5, 10],
        "min_samples_leaf": [1, 5, 8],
    }

    grid = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=inner_folds,
        scoring="f1",
        n_jobs=-1,
    )
    grid.fit(X_trainval, y_trainval)

    print("Best params:", grid.best_params_)
    print("Best spatial train/val F1:", grid.best_score_)
    return grid.best_estimator_, grid.best_params_


def evaluate_final_model(model, X_test, y_test, feature_names):
    test_pred = model.predict(X_test)
    test_proba = model.predict_proba(X_test)[:, 1]

    print("\n===== FINAL TEST SET PERFORMANCE =====")
    print(f"Accuracy:  {accuracy_score(y_test, test_pred):.4f}")
    print(f"Precision: {precision_score(y_test, test_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, test_pred):.4f}")
    print(f"F1 score:  {f1_score(y_test, test_pred):.4f}")
    print(f"ROC AUC:   {roc_auc_score(y_test, test_proba):.4f}")

    print("\nConfusion matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_test, test_pred))
    print("\nClassification report:")
    print(classification_report(y_test, test_pred, target_names=["No Flood", "Flood"]))

    print("\nFeature importances:")
    for name, imp in sorted(zip(feature_names, model.feature_importances_), key=lambda x: -x[1]):
        print(f"  {name}: {imp:.4f}")


if __name__ == "__main__":
    X, y, coords, predictor_cols, raster, transform, crs, valid_mask = load_raster_dataset(
        GIS_RASTER, BAND_NAMES, TARGET_BAND
    )

    X_trainval, y_trainval, X_test, y_test, inner_folds = nested_spatial_split(
        X, y, coords,
        outer_splits=OUTER_SPLITS, inner_splits=INNER_SPLITS,
        block_spacing=BLOCK_SPACING, buffer_radius=BUFFER_RADIUS,
    )
    print(f"train+val pixels: {len(y_trainval)}  |  test pixels: {len(y_test)}")

    best_model, best_params = tune_and_evaluate(X_trainval, y_trainval, inner_folds)
    evaluate_final_model(best_model, X_test, y_test, predictor_cols)

    # ---- Save the trained model ----
    joblib.dump(best_model, "flood_rf_model.joblib")
    print("\nSaved model -> flood_rf_model.joblib")

    # ---- Predict across the FULL raster (not just ROI/test pixels) ----
    n_bands = len(BAND_NAMES)
    X_all = raster.values.reshape(n_bands, -1).T
    df_all = pd.DataFrame(X_all, columns=BAND_NAMES)
    X_all_predictors = df_all[predictor_cols].values

    pred_class_flat = np.full(X_all.shape[0], 255, dtype=np.uint8)      # 255 = nodata
    pred_proba_flat = np.full(X_all.shape[0], -1.0, dtype=np.float32)   # -1 = nodata

    pred_class_flat[valid_mask] = best_model.predict(X_all_predictors[valid_mask]).astype(np.uint8)
    pred_proba_flat[valid_mask] = best_model.predict_proba(X_all_predictors[valid_mask])[:, 1].astype(np.float32)

    height, width = raster.shape[1], raster.shape[2]
    pred_class_map = pred_class_flat.reshape(height, width)
    pred_proba_map = pred_proba_flat.reshape(height, width)

    out_meta = dict(driver="GTiff", height=height, width=width, count=1,
                     crs=crs, transform=transform, compress="LZW")

    with rasterio.open("erosion_susceptibility_class.tif", "w", dtype="uint8", nodata=255, **out_meta) as dst:
        dst.write(pred_class_map, 1)
    with rasterio.open("erosion_susceptibility.tif", "w", dtype="float32", nodata=-1.0, **out_meta) as dst:
        dst.write(pred_proba_map, 1)

    print("Saved -> erosion_susceptibility_class.tif (0=no-erosion, 1=erosion, 255=nodata)")
    print("Saved -> erosion_susceptibility.tif (erosion probability 0-1, -1=nodata)")
    print(f"erosion_susceptibility pixels: {(pred_class_map == 1).sum()}, "
          f"non-erosion: {(pred_class_map == 0).sum()}")