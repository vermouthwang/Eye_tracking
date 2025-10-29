#!/usr/bin/env bash
set -euo pipefail
python fit_gaze_poly.py --csv "Michael_calibration_frame.csv" --out "Michael_regression_model_lam0.019306977288832496.json" --lam 0.019306977288832496 --weighted 1 --plot 1 --plot-out lambda_search/plots/Michael_regression_model_best.png --show-plot 0
