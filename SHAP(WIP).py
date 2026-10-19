import joblib
import numpy as np
import pathlib
import pickle
import shap
import pandas as pd
WORKDIR = pathlib.Path("../erosion_run_cache")
WORKDIR.mkdir(exist_ok=True)
MODEL_PATH = WORKDIR / "erosion_rf_model.joblib"
SPLIT_PKL = WORKDIR / "split.pkl"

BAND_NAMES = ["dem", "slope",'aspect', "twi","spi", "rainfall",'plan_curvature','profile_curvature']

N_BOOTSTRAPS = 30
SAMPLE_SIZE = 200
mean_abs_shap_list = []

model = joblib.load(MODEL_PATH)
with open(SPLIT_PKL, "rb") as f:
    split = pickle.load(f)

x_test = split["X_test"]

explainer = shap.TreeExplainer(model)

arr_x = np.asarray(x_test)
total_rows = arr_x.shape[0]

for i in range(N_BOOTSTRAPS):
    np.random.seed(i)
    idx = np.random.choice(total_rows,size=SAMPLE_SIZE,replace=False)
    x_test = pd.DataFrame(arr_x[idx],columns=BAND_NAMES)

    shap_values = explainer(x_test,check_additivity=False)

    vals = np.abs(shap_values.values[:,:,1].mean(axis=0))
    mean_abs_shap_list.append(vals)

sum_df = pd.DataFrame(mean_abs_shap_list,columns=BAND_NAMES)



result = pd.DataFrame({
    "Mean_SHAP":sum_df.mean(),
    "CI_lower2.5%":sum_df.quantile(0.025),
    "CI_upper97.5%":sum_df.quantile(0.975)
})
print(result)

shap.plots.beeswarm(shap_values[:, :, 1])

#Tips
"""
High val SHAP = feature that influence the prediction the most


"""