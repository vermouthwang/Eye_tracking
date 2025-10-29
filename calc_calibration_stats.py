# save as calc_calibration_stats.py
# This script computes the calibration stats for the pitch/yaw data in trimmed mean way
# and saves the results to a CSV file.
#
# Usage:
#   python calc_calibration_stats.py --csv <path-to-gaze-csv> --out <path-to-output-csv>
#
# Example:
#   python calc_calibration_stats.py --csv gaze.csv --out calibration_stats.csv
#
# Input CSV should have the following columns:
#   frame_idx, pitch_deg, yaw_deg
#
# Output CSV will have the following columns:
#   row, col, center_frame, start_frame, end_frame, n_frames, mean_pitch_deg, mean_yaw_deg, std_pitch_deg, std_yaw_deg, range_pitch_deg, range_yaw_deg, max_abs_dev_pitch_deg, max_abs_dev_yaw_deg


import pandas as pd
import numpy as np
import argparse
from collections import OrderedDict

def trimmed_mean(values: np.ndarray, trim: float = 0.3) -> float:
    """Compute a trimmed mean by removing the top/bottom fraction of samples
       with the largest absolute deviation from the median."""
    if len(values) == 0:
        return float('nan')
    if trim <= 0:
        return float(np.mean(values))
    if trim >= 0.5:
        trim = 0.49

    med = np.median(values)
    deviations = np.abs(values - med)
    k = int(np.floor(trim * len(values)))

    if k == 0:
        return float(np.mean(values))

    idx = np.argsort(deviations)
    kept = values[idx[k: len(values) - k]]
    return float(np.mean(kept)) if len(kept) else float(np.mean(values))

def trimmed_range(values: np.ndarray, trim: float = 0.3) -> float:
    """Compute a robust range (max - min) after trimming outliers by deviation."""
    if len(values) == 0:
        return float('nan')
    if trim <= 0:
        return float(np.max(values) - np.min(values))
    if trim >= 0.5:
        trim = 0.49

    med = np.median(values)
    deviations = np.abs(values - med)
    k = int(np.floor(trim * len(values)))

    if k == 0:
        return float(np.max(values) - np.min(values))

    idx = np.argsort(deviations)
    kept = values[idx[k: len(values) - k]]
    return float(np.max(kept) - np.min(kept)) if len(kept) else float(np.max(values) - np.min(values))

def compute_stats(series):
    mean = trimmed_mean(series, 0.5) if len(series) else float('nan')
    std = float(np.std(series, ddof=0)) if len(series) else float('nan')
    rng = trimmed_range(series, 0.5) if len(series) else float('nan')
    mad = float(np.max(np.abs(series - mean))) if len(series) else float('nan')
    return mean, std, rng, mad

def main():
    parser = argparse.ArgumentParser(description="Compute calibration window stats for pitch/yaw.")
    parser.add_argument("--csv", required=True, help="Path to gaze CSV (with pitch_deg,yaw_deg columns).")
    parser.add_argument("--window", type=int, default=15, help="+/- frames around center (default: 15).")
    parser.add_argument("--out", default="calibration_stats.csv", help="Output CSV path.")
    parser.add_argument(
        "--middleFrames",
        nargs=9,
        type=int,
        required=True,
        help="Nine integers specifying the middle frame indices for each grid position."
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    required_cols = {"frame_idx", "pitch_deg", "yaw_deg"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    grid_positions = [
        (0,0), (1,0), (2,0),
        (2,1), (2,2), (1,2),
        (0,2), (0,1), (1,1),
    ]
    centers = OrderedDict(zip(grid_positions, args.middleFrames))

    rows = []
    for (r, c), center in centers.items():
        start = center - args.window
        end = center + args.window
        window_df = df[(df["frame_idx"] >= start) & (df["frame_idx"] <= end)]

        pitch = window_df["pitch_deg"].to_numpy()
        yaw = window_df["yaw_deg"].to_numpy()

        mean_pitch, std_pitch, range_pitch, max_abs_dev_pitch = compute_stats(pitch)
        mean_yaw, std_yaw, range_yaw, max_abs_dev_yaw = compute_stats(yaw)

        rows.append({
            "row": r,
            "col": c,
            "center_frame": center,
            "start_frame": int(start),
            "end_frame": int(end),
            "n_frames": int(len(window_df)),
            "mean_pitch_deg": round(mean_pitch, 4),
            "mean_yaw_deg": round(mean_yaw, 4),
            "std_pitch_deg": round(std_pitch, 4),
            "std_yaw_deg": round(std_yaw, 4),
            "range_pitch_deg": round(range_pitch, 4),      # trimmed range
            "range_yaw_deg": round(range_yaw, 4),          # trimmed range
            "max_abs_dev_pitch_deg": round(max_abs_dev_pitch, 4),
            "max_abs_dev_yaw_deg": round(max_abs_dev_yaw, 4),
        })

    out_df = pd.DataFrame(rows, columns=[
        "row","col","center_frame","start_frame","end_frame","n_frames",
        "mean_pitch_deg","mean_yaw_deg",
        "std_pitch_deg","std_yaw_deg",
        "range_pitch_deg","range_yaw_deg",
        "max_abs_dev_pitch_deg","max_abs_dev_yaw_deg",
    ])
    out_df.to_csv(args.out, index=False)

    print(out_df[[
        "row","col","center_frame","n_frames",
        "mean_pitch_deg","mean_yaw_deg"
    ]])

if __name__ == "__main__":
    main()
