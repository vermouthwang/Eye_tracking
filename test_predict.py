#!/usr/bin/env python3
# predict_frames.py
import argparse, json
import numpy as np
import pandas as pd
from pathlib import Path

def features(yaw, pitch):
    return np.array([1.0, yaw, pitch, yaw*yaw, pitch*pitch, yaw*pitch], dtype=float)

def predict_xy_norm(yaw_deg, pitch_deg, coeff_x, coeff_y):
    phi = features(yaw_deg, pitch_deg)
    x = float(phi @ coeff_x)
    y = float(phi @ coeff_y)
    # clip to [0,1]
    x = 0.0 if x < 0 else (1.0 if x > 1 else x)
    y = 0.0 if y < 0 else (1.0 if y > 1 else y)
    return x, y

def ema(series, alpha=0.3):
    s = np.asarray(series, dtype=float)
    if len(s) == 0: return s
    out = np.empty_like(s)
    out[0] = s[0]
    for i in range(1, len(s)):
        out[i] = alpha * s[i] + (1 - alpha) * out[i-1]
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True, help="Input CSV with columns: frame,pitch_deg,yaw_deg")
    ap.add_argument("--model_json", required=True, help="model_gaze_poly.json")
    # ap.add_argument("--screen_w", type=int, default=800)
    # ap.add_argument("--screen_h", type=int, default=600)
    ap.add_argument("--frame_start", type=int, default=1)
    ap.add_argument("--frame_end", type=int, default=1000)
    ap.add_argument("--smoothing", choices=["none","ema"], default="ema")
    ap.add_argument("--alpha", type=float, default=0.3, help="EMA alpha if smoothing=ema")
    ap.add_argument("--out_csv", default="gaze_predictions.csv")
    args = ap.parse_args()

    # Load model
    with open(args.model_json, "r") as f:
        m = json.load(f)
    coeff_x = np.array(m["coeff_x"], dtype=float)  # [1,yaw,pitch,yaw^2,pitch^2,yaw*pitch]
    coeff_y = np.array(m["coeff_y"], dtype=float)

    # Load frames
    df = pd.read_csv(args.in_csv)
    df = df[df["frame_idx"] >= args.frame_start]
    df = df[df["frame_idx"] <= args.frame_end]
    for col in ["frame_idx","pitch_deg","yaw_deg"]:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {args.in_csv}")

    # Predict per-frame
    xs, ys = [], []
    for yaw_deg, pitch_deg in zip(df["yaw_deg"].to_numpy(), df["pitch_deg"].to_numpy()):
        x, y = predict_xy_norm(yaw_deg, pitch_deg, coeff_x, coeff_y)
        xs.append(x); ys.append(y)
    xs = np.array(xs); ys = np.array(ys)

    # Optional smoothing
    if args.smoothing == "ema":
        xs_s = ema(xs, alpha=args.alpha)
        ys_s = ema(ys, alpha=args.alpha)
    else:
        xs_s, ys_s = xs, ys

    # To pixels
    # x_px = (xs_s * args.screen_w).astype(int)
    # y_px = (ys_s * args.screen_h).astype(int)

    # Save
    out = df.copy()
    out["x_norm"] = xs
    out["y_norm"] = ys
    out["x_norm_smooth"] = xs_s
    out["y_norm_smooth"] = ys_s
    # out["x_px"] = x_px
    # out["y_px"] = y_px
    Path(args.out_csv).write_text(out.to_csv(index=False))
    print(f"Saved: {Path(args.out_csv).resolve()}")

if __name__ == "__main__":
    main()
