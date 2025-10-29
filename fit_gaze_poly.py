#!/usr/bin/env python3
"""
fit_gaze_poly.py

Weighted 2nd-order polynomial ridge mapping:
(yaw, pitch) --> normalized (x, y)

Usage:
  python fit_gaze_poly.py --csv sven_manual_frame_window_15_cali_range_rob.csv \
      --out model_gaze_poly.json \
      --lam 1e-4 \
      --weighted 1 \
      --plot 1

Input CSV needs columns:
  row, col, mean_yaw_deg, mean_pitch_deg, std_yaw_deg, std_pitch_deg, n_frames
(Extra columns are ignored.)
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

def make_features(yaw_deg: np.ndarray, pitch_deg: np.ndarray) -> np.ndarray:
    """
    Build 2nd-order polynomial features:
      phi = [1, yaw, pitch, yaw^2, pitch^2, yaw*pitch]
    Inputs are in degrees (same as your CSV).
    """
    yaw = yaw_deg.astype(float)
    pitch = pitch_deg.astype(float)
    Phi = np.column_stack([
        np.ones_like(yaw),
        yaw,
        pitch,
        yaw**2,
        pitch**2,
        yaw * pitch
    ])
    return Phi

def ridge_fit(Phi: np.ndarray, t: np.ndarray, lam: float, W: np.ndarray | None):
    """
    Solve weighted ridge:
      w = (Phi^T W Phi + lam I)^(-1) Phi^T W t
    If W is None, uses identity (unweighted).
    """
    d = Phi.shape[1]
    I = np.eye(d)
    if W is None:
        A = Phi.T @ Phi + lam * I
        b = Phi.T @ t
    else:
        A = Phi.T @ W @ Phi + lam * I
        b = Phi.T @ W @ t
    w = np.linalg.solve(A, b)
    return w

def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

def loo_errors(Phi, tx, ty, lam, W):
    """
    Leave-one-out absolute errors for each point (in normalized units).
    Returns a DataFrame with abs err_x, err_y per point.
    """
    n = Phi.shape[0]
    rows = []
    for i in range(n):
        keep = np.ones(n, dtype=bool)
        keep[i] = False
        Phi_k = Phi[keep]
        tx_k = tx[keep]
        ty_k = ty[keep]

        if W is None:
            W_k = None
        else:
            # Subselect diagonal weights for the kept points
            w_vec = np.diag(W)[keep]
            W_k = np.diag(w_vec)

        w_x_k = ridge_fit(Phi_k, tx_k, lam, W_k)
        w_y_k = ridge_fit(Phi_k, ty_k, lam, W_k)

        # Predict the left-out sample
        pred_x_i = float(Phi[~keep] @ w_x_k)
        pred_y_i = float(Phi[~keep] @ w_y_k)

        rows.append((pred_x_i, pred_y_i))
    pred_x_loo = np.array([r[0] for r in rows])
    pred_y_loo = np.array([r[1] for r in rows])

    err_x = np.abs(pred_x_loo - tx)
    err_y = np.abs(pred_y_loo - ty)

    return err_x, err_y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to calibration CSV")
    ap.add_argument("--out", default="model_gaze_poly.json", help="Where to save coefficients JSON")
    ap.add_argument("--lam", type=float, default=1e-4, help="Ridge lambda (L2 strength)")
    ap.add_argument("--weighted", type=int, default=1, help="1=use noise-based weights, 0=unweighted")
    ap.add_argument("--plot", type=int, default=0, help="1=show diagnostic plot (requires matplotlib)")
    ap.add_argument("--plot-out", type=str, default=None, help="Path to save plot (e.g., plots/best.png)")
    ap.add_argument("--show-plot", type=int, default=0, help="1=show plot, 0=don't show")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    required = ["row","col","mean_yaw_deg","mean_pitch_deg","std_yaw_deg","std_pitch_deg","n_frames"]
    for col in required:
        if col not in df.columns:
            sys.exit(f"ERROR: Missing column '{col}' in {args.csv}")

    # Features
    Phi = make_features(df["mean_yaw_deg"].to_numpy(),
                        df["mean_pitch_deg"].to_numpy())

    # Targets (normalized to [0,1])
    max_col = float(df["col"].max())
    max_row = float(df["row"].max())
    if max_col <= 0 or max_row <= 0:
        sys.exit("ERROR: row/col must span at least 0..2 for a 3x3 grid.")

    tx = df["col"].to_numpy() / max_col  # 0, 0.5, 1
    ty = df["row"].to_numpy() / max_row  # 0, 0.5, 1

    # Weights from noise (std of yaw/pitch); small epsilon to avoid div-by-zero.
    W = None
    if args.weighted:
        noise = (df["std_yaw_deg"].to_numpy()**2 +
                 df["std_pitch_deg"].to_numpy()**2)
        w_vec = 1.0 / (noise + 1e-6)
        # Normalize weights so median = 1 (keeps conditioning nice, optional)
        w_vec = w_vec / np.median(w_vec)
        W = np.diag(w_vec)

    # Fit weighted ridge for x and y
    w_x = ridge_fit(Phi, tx, args.lam, W)
    w_y = ridge_fit(Phi, ty, args.lam, W)

    # In-sample predictions & metrics
    pred_x = Phi @ w_x
    pred_y = Phi @ w_y

    abs_err_x = np.abs(pred_x - tx)
    abs_err_y = np.abs(pred_y - ty)
    r2_x = r2_score(tx, pred_x)
    r2_y = r2_score(ty, pred_y)

    # LOO errors
    loo_abs_x, loo_abs_y = loo_errors(Phi, tx, ty, args.lam, W)

    # Print summary
    def summarize(err, name):
        return {
            "median": float(np.median(err)),
            "mean": float(np.mean(err)),
            "max": float(np.max(err)),
            "p90": float(np.percentile(err, 90))
        }

    print("\n=== Weighted Polynomial Ridge Fit ===")
    print(f"File: {args.csv}")
    print(f"Weighted: {bool(args.weighted)}   lambda: {args.lam}")
    print(f"R^2_x: {r2_x:.4f}   R^2_y: {r2_y:.4f}")
    # print("Train |x-error|:", summarize(abs_err_x, "x"))
    # print("Train |y-error|:", summarize(abs_err_y, "y"))
    print("LOO   |x-error|:", summarize(loo_abs_x, "x"))
    print("LOO   |y-error|:", summarize(loo_abs_y, "y"))

    # Save model
    out_path = Path(args.out)
    model = {
        "feature_order": ["1","yaw","pitch","yaw2","pitch2","yaw_pitch"],
        "lambda": args.lam,
        "weighted": bool(args.weighted),
        "coeff_x": w_x.tolist(),
        "coeff_y": w_y.tolist(),
        "col_max": max_col,   # for normalization reference
        "row_max": max_row
    }
    out_path.write_text(json.dumps(model, indent=2))
    # print(f"\nSaved coefficients to: {out_path.resolve()}")

    # Optional diagnostic plot
# Optional diagnostic plot
    if args.plot or args.plot_out:
        try:
            import os
            import matplotlib
            # In case you're running headless (e.g., from a script/cron/Docker):
            if not matplotlib.get_backend().lower().startswith("qt") and "DISPLAY" not in os.environ:
                matplotlib.use("Agg")  # non-interactive backend
    
            import matplotlib.pyplot as plt
    
            # Create figure and keep a handle so we can close it safely
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111)
    
            ax.scatter(tx, ty, c="tab:red", label="Targets (calibration grid)")
            ax.scatter(pred_x, pred_y, c="tab:blue", marker="x", label="Fitted (in-sample)")
    
            # draw connecting segments
            for i in range(len(tx)):
                ax.plot([tx[i], pred_x[i]], [ty[i], pred_y[i]], "k--", alpha=0.5)
    
            ax.set_xlabel("x (normalized)")
            ax.set_ylabel("y (normalized)")
            ax.set_title("Calibration fit: targets vs. predictions")
            ax.legend()
            ax.axis("equal")
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
    
            # 1) Always save a default copy next to the run for quick inspection
            default_png = "Michael_calibration_fit.png"
            fig.savefig(default_png, dpi=160)
    
            # 2) If a custom output path is provided, ensure directory exists and save again
            if getattr(args, "plot_out", None):
                out_dir = os.path.dirname(args.plot_out)
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)
                print(f"Saving plot to {args.plot_out}")
                fig.savefig(args.plot_out, dpi=160)
    
            # 3) Show or close
            if getattr(args, "show_plot", 0):
                plt.show()
            else:
                plt.close(fig)
    
        except Exception as e:
            print(f"Error creating plot: {e}")


if __name__ == "__main__":
    main()
