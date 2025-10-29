# Compute basic per-point quality metrics and fit a 2D polynomial mapping
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv("first_ui_test_calibration_stats.csv")

# Basic sanity checks
assert df.shape[0] >= 9, "Expected at least 9 calibration points"

# Inter-point spacing (in angle space)
means = df[["row","col","mean_pitch_deg","mean_yaw_deg"]].copy()
means["id"] = list(range(len(means)))

# Compute average within-point noise (use std; if n_frames given, standard error also helpful)
means["std_pitch"] = df["std_pitch_deg"]
means["std_yaw"] = df["std_yaw_deg"]
means["se_pitch"] = df["std_pitch_deg"] / np.sqrt(df["n_frames"].clip(lower=1))
means["se_yaw"] = df["std_yaw_deg"] / np.sqrt(df["n_frames"].clip(lower=1))

# Distance between adjacent grid points (by row/col adjacency)
adj_pairs = []
for _, a in means.iterrows():
    for _, b in means.iterrows():
        if (abs(a["row"]-b["row"]) + abs(a["col"]-b["col"])) == 1:  # 4-neighborhood
            d = np.hypot(a["mean_yaw_deg"]-b["mean_yaw_deg"], a["mean_pitch_deg"]-b["mean_pitch_deg"])
            adj_pairs.append(d)
adj_pairs = np.array(adj_pairs)/2  # each pair counted twice
adj_spacing = np.median(adj_pairs) if adj_pairs.size>0 else np.nan

avg_within_std = np.median(np.hypot(means["std_yaw"], means["std_pitch"]))

# Signal-to-noise ratio (higher is better): grid spacing vs within-std
snr = adj_spacing / (avg_within_std + 1e-9)

# Prepare targets: normalized screen coords (0..1). Left->x=0, right->x=1; top->y=0, bottom->y=1.
X = df[["mean_yaw_deg","mean_pitch_deg"]].to_numpy()
yaw = X[:,0]
pitch = X[:,1]
# 2nd-order polynomial features: [1, yaw, pitch, yaw^2, pitch^2, yaw*pitch]
Phi = np.column_stack([
    np.ones(len(df)),
    yaw, pitch, yaw**2, pitch**2, yaw*pitch
])

# Targets from grid indices
tx = df["col"].to_numpy() / df["col"].max()  # 0,0.5,1
ty = df["row"].to_numpy() / df["row"].max()  # 0,0.5,1

# Fit ridge regression (tiny L2 for stability)
lam = 1e-4
A = Phi.T @ Phi + lam*np.eye(Phi.shape[1])
w_x = np.linalg.solve(A, Phi.T @ tx)
w_y = np.linalg.solve(A, Phi.T @ ty)

pred_x = Phi @ w_x
pred_y = Phi @ w_y

# Compute R^2 and residuals
def r2(y, yhat):
    ss_res = np.sum((y-yhat)**2)
    ss_tot = np.sum((y - y.mean())**2)
    return 1 - ss_res/ss_tot if ss_tot>0 else np.nan

r2_x = r2(tx, pred_x)
r2_y = r2(ty, pred_y)

res = pd.DataFrame({
    "row": df["row"],
    "col": df["col"],
    # "n_frames": df["n_frames"],
    "std_pitch_deg": df["std_pitch_deg"].round(3),
    "std_yaw_deg": df["std_yaw_deg"].round(3),
    "pred_x": pred_x.round(3),
    "pred_y": pred_y.round(3),
    "target_x": tx,
    "target_y": ty,
    "abs_err_x": np.abs(pred_x-tx).round(3),
    "abs_err_y": np.abs(pred_y-ty).round(3),
})

# Leave-one-out (LOO) check to spot problematic points
loo_errors = []
for i in range(len(df)):
    keep = np.ones(len(df), dtype=bool); keep[i]=False
    Phi_k = Phi[keep]; tx_k = tx[keep]; ty_k = ty[keep]
    A = Phi_k.T @ Phi_k + lam*np.eye(Phi_k.shape[1])
    w_x_k = np.linalg.solve(A, Phi_k.T @ tx_k)
    w_y_k = np.linalg.solve(A, Phi_k.T @ ty_k)
    pred_x_i = Phi[~keep] @ w_x_k
    pred_y_i = Phi[~keep] @ w_y_k
    loo_errors.append([int(df.loc[i,"row"]), int(df.loc[i,"col"]), float(abs(pred_x_i-tx[i])), float(abs(pred_y_i-ty[i]))])
loo_df = pd.DataFrame(loo_errors, columns=["row","col","loo_abs_err_x","loo_abs_err_y"])

# Merge for a single table
summary = res.merge(loo_df, on=["row","col"])
# get loo error sum top 3
summary["loo_error_sum"] = summary["loo_abs_err_x"] + summary["loo_abs_err_y"]
summary = summary.sort_values(by="loo_error_sum", ascending=False)

# Display the summary to the user
# import caas_jupyter_tools as cj
# cj.display_dataframe_to_user("Calibration fit & quality (per point)", summary)
print(summary)

# Also show high-level metrics
print("Grid-adjacent spacing in angle space (deg):", round(adj_spacing,3))
print("Median within-point std (combined yaw/pitch) (deg):", round(avg_within_std,3))
print("SNR (spacing / within-std):", round(snr,2))
print("R^2 (x):", round(r2_x,4), " R^2 (y):", round(r2_y,4))

# Median |error| (x)(y) , max |error| (x)(y)
print("Median |error| (x):", round(summary["abs_err_x"].median(),3))
print("Median |error| (y):", round(summary["abs_err_y"].median(),3))
print("Max |error| (x):", round(summary["abs_err_x"].max(),3))
print("Max |error| (y):", round(summary["abs_err_y"].max(),3))

# LLO oo_abs_err_x / oo_abs_err_y mean / max error
print("LLO mean error (x,y):", round(summary["loo_abs_err_x"].mean(),3), round(summary["loo_abs_err_y"].mean(),3))
print("LLO max error (x,y):", round(summary["loo_abs_err_x"].max(),3), round(summary["loo_abs_err_y"].max(),3))

# Quick scatter in angle space, colored by grid index
# plt.figure()
# plt.scatter(df["mean_yaw_deg"], df["mean_pitch_deg"])
# for idx, r in df.iterrows():
#     plt.text(r["mean_yaw_deg"], r["mean_pitch_deg"], f'({int(r["row"])},{int(r["col"])})')
# plt.xlabel("mean_yaw_deg")
# plt.ylabel("mean_pitch_deg")
# plt.title("Calibration centroids in angle space")
# plt.tight_layout()
# # plt.show()


#    row  col  std_pitch_deg  std_yaw_deg  pred_x  pred_y  target_x  target_y  abs_err_x  abs_err_y  loo_abs_err_x  loo_abs_err_y  loo_error_sum
# 7    0    1          6.643        3.076   0.513  -0.043       0.5       0.0      0.013      0.043       0.101402       0.336255       0.437657
# 6    0    2          4.006        1.643   0.964   0.088       1.0       0.0      0.036      0.088       0.099168       0.238621       0.337789
# 5    1    2          2.288        2.208   1.045   0.415       1.0       0.5      0.045      0.085       0.091523       0.172283       0.263806
# 4    2    2          6.847        1.602   0.980   1.018       1.0       1.0      0.020      0.018       0.120694       0.105992       0.226686
# 2    2    0          1.954        2.177   0.010   0.984       0.0       1.0      0.010      0.016       0.066107       0.107602       0.173709
# 0    0    0          0.847        1.432  -0.002   0.042       0.0       0.0      0.002      0.042       0.005863       0.161375       0.167238
# 3    2    1          1.452        1.264   0.501   1.041       0.5       1.0      0.001      0.041       0.002006       0.086882       0.088888
# 8    1    1          4.386        2.214   0.500   0.477       0.5       0.5      0.000      0.023       0.001265       0.059399       0.060665
# 1    1    0          2.263        1.894  -0.011   0.478       0.0       0.5      0.011      0.022       0.020348       0.038778       0.059126
# Grid-adjacent spacing in angle space (deg): 6.062
# Median within-point std (combined yaw/pitch) (deg): 3.18
# SNR (spacing / within-std): 1.91
# R^2 (x): 0.9972  R^2 (y): 0.9854
# Median |error| (x): 0.011
# Median |error| (y): 0.041
# Max |error| (x): 0.045
# Max |error| (y): 0.088