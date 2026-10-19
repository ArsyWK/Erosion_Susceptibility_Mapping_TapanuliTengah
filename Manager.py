import json
import pathlib
import pickle
import sqlite3
from typing import TypedDict
import joblib
import numpy as np
import pandas as pd
import rasterio
import rioxarray
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from SigProject import (
    BAND_NAMES,
    BLOCK_SPACING,
    BUFFER_RADIUS,
    GIS_RASTER,
    INNER_SPLITS,
    OUTER_SPLITS,
    PREDICTOR_BANDS,
    TARGET_BAND,
    evaluate_final_model,
    load_raster_dataset,
    nested_spatial_split,
    tune_and_evaluate,
)

WORKDIR = pathlib.Path("../erosion_run_cache")
WORKDIR.mkdir(exist_ok=True)

DATA_NPZ = WORKDIR / "data.npz"
SPLIT_PKL = WORKDIR / "split.pkl"
MODEL_PATH = WORKDIR / "erosion_rf_model.joblib"
BEST_PARAMS_JSON = WORKDIR / "best_params.json"
METRICS_JSON = WORKDIR / "metrics.json"
PRED_CLASS_TIF = WORKDIR / "erosion_prediction_class.tif"
PRED_PROBA_TIF = WORKDIR / "erosion_prediction_probability.tif"


class PipelineState(TypedDict, total=False):
    status: str

def node_load_data(state: PipelineState) -> PipelineState:
    if DATA_NPZ.exists():
        print("[load_data] cache hit")
        return {"status": "data_loaded"}

    X, y, coords, predictor_cols, raster, transform, crs, valid_mask = load_raster_dataset(
        GIS_RASTER, BAND_NAMES, TARGET_BAND
    )
    np.savez(
        DATA_NPZ,
        X=X, y=y, coords=coords, valid_mask=valid_mask,
        predictor_cols=np.array(predictor_cols),
    )
    return {"status": "data_loaded"}


def node_spatial_split(state: PipelineState) -> PipelineState:
    if SPLIT_PKL.exists():
        print("[spatial_split] cache hit")
        return {"status": "split_done"}

    print("[spatial_split]")
    d = np.load(DATA_NPZ, allow_pickle=True)
    X, y, coords = d["X"], d["y"], d["coords"]

    X_trainval, y_trainval, X_test, y_test, inner_folds = nested_spatial_split(
        X, y, coords,
        outer_splits=OUTER_SPLITS, inner_splits=INNER_SPLITS,
        block_spacing=BLOCK_SPACING, buffer_radius=BUFFER_RADIUS,
    )
    with open(SPLIT_PKL, "wb") as f:
        pickle.dump(
            dict(X_trainval=X_trainval, y_trainval=y_trainval,
                 X_test=X_test, y_test=y_test, inner_folds=inner_folds),
            f,
        )
    print(f"[spatial_split] {len(y_trainval)},test pixels: {len(y_test)}")
    return {"status": "split_done"}


def node_tune_model(state: PipelineState) -> PipelineState:
    if MODEL_PATH.exists():
        print("[tune_model] cache hit")
        return {"status": "model_trained"}

    print("[tune_model]")
    with open(SPLIT_PKL, "rb") as f:
        split = pickle.load(f)

    best_model, best_params = tune_and_evaluate(
        split["X_trainval"], split["y_trainval"], split["inner_folds"]
    )
    joblib.dump(best_model, MODEL_PATH)
    BEST_PARAMS_JSON.write_text(json.dumps(best_params, indent=2))
    return {"status": "model_trained"}


def node_evaluate(state: PipelineState) -> PipelineState:
    if METRICS_JSON.exists():
        print("[evaluate] cache hit")
        return {"status": "evaluated"}

    with open(SPLIT_PKL, "rb") as f:
        split = pickle.load(f)
    d = np.load(DATA_NPZ, allow_pickle=True)
    predictor_cols = list(d["predictor_cols"])
    best_model = joblib.load(MODEL_PATH)

    evaluate_final_model(best_model, split["X_test"], split["y_test"], predictor_cols)
    METRICS_JSON.write_text(json.dumps({"evaluated": True}))
    return {"status": "evaluated"}


def node_predict_raster(state: PipelineState) -> PipelineState:
    if PRED_CLASS_TIF.exists() and PRED_PROBA_TIF.exists():
        print("[predict_raster] cache hit")
        return {"status": "done"}

    d = np.load(DATA_NPZ, allow_pickle=True)
    valid_mask = d["valid_mask"]
    predictor_cols = list(d["predictor_cols"])
    best_model = joblib.load(MODEL_PATH)

    raster = rioxarray.open_rasterio(GIS_RASTER)
    n_bands = len(BAND_NAMES)
    X_all = raster.values.reshape(n_bands, -1).T
    df_all = pd.DataFrame(X_all, columns=BAND_NAMES)
    X_all_predictors = df_all[predictor_cols].values

    pred_class_flat = np.full(X_all.shape[0], 255, dtype=np.uint8)
    pred_proba_flat = np.full(X_all.shape[0], -1.0, dtype=np.float32)
    pred_class_flat[valid_mask] = best_model.predict(X_all_predictors[valid_mask]).astype(np.uint8)
    pred_proba_flat[valid_mask] = best_model.predict_proba(X_all_predictors[valid_mask])[:, 1].astype(np.float32)

    height, width = raster.shape[1], raster.shape[2]
    out_meta = dict(driver="GTiff", height=height, width=width, count=1,
                     crs=raster.rio.crs, transform=raster.rio.transform(), compress="LZW")

    with rasterio.open(PRED_CLASS_TIF, "w", dtype="uint8", nodata=255, **out_meta) as dst:
        dst.write(pred_class_flat.reshape(height, width), 1)
    with rasterio.open(PRED_PROBA_TIF, "w", dtype="float32", nodata=-1.0, **out_meta) as dst:
        dst.write(pred_proba_flat.reshape(height, width), 1)

    print(f"[predict_raster] saved -> {PRED_CLASS_TIF}, {PRED_PROBA_TIF}")
    return {"status": "done"}


def build_graph(checkpointer):
    graph = StateGraph(PipelineState)
    graph.add_node("load_data", node_load_data)
    graph.add_node("spatial_split", node_spatial_split)
    graph.add_node("tune_model", node_tune_model)
    graph.add_node("evaluate", node_evaluate)
    graph.add_node("predict_raster", node_predict_raster)

    graph.add_edge(START, "load_data")
    graph.add_edge("load_data", "spatial_split")
    graph.add_edge("spatial_split", "tune_model")
    graph.add_edge("tune_model", "evaluate")
    graph.add_edge("evaluate", "predict_raster")
    graph.add_edge("predict_raster", END)

    return graph.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    conn = sqlite3.connect(WORKDIR / "checkpoints.db", check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    app = build_graph(checkpointer)

    THREAD_ID = "erosion_rf_run_1"
    config = {"configurable": {"thread_id": THREAD_ID}}

    final_state = app.invoke({"status": "start"}, config=config)
    print("\nPipeline finished:", final_state)