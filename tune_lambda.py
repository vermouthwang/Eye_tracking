from __future__ import annotations
"""
Lambda tuner for fit_gaze_poly.py

Runs the provided fit_gaze_poly.py multiple times with different ridge
regularization strengths (lambda) and selects the lambda that minimizes
(x_mean + y_mean) from the reported LOO errors.

It performs a two‑stage search:
  1) Coarse log‑spaced sweep over [min_lam, max_lam]
  2) Local refinement around the current best by sampling multiplicative
     neighbors. Repeatable for a few iterations.

Outputs a CSV summary of all trials and re‑runs the best configuration to
save the final model + (optionally) plot.

Example:
  python tune_lambda.py \
    --csv Michael_calibration_frame.csv \
    --out-prefix Michael_reg_model \
    --weighted 1 \
    --plot-best 1 \
    --min-lam 1e-6 --max-lam 1e-1 --coarse-points 8 --refine-iters 2

Assumptions about fit_gaze_poly.py stdout (typical lines):
  R^2_x: 0.9940   R^2_y: 0.9680
  LOO   |x-error|: {'median': 0.1117, 'mean': 0.1077, ...}
  LOO   |y-error|: {'median': 0.1989, 'mean': 0.2387, ...}

If your script prints slightly different keys, adjust the regex below.
"""
#!/usr/bin/env python3
"""
Lambda tuner for fit_gaze_poly.py

Runs the provided fit_gaze_poly.py multiple times with different ridge
regularization strengths (lambda) and selects the lambda that minimizes
(x_mean + y_mean) from the reported LOO errors.

It performs a two‑stage search:
  1) Coarse log‑spaced sweep over [min_lam, max_lam]
  2) Local refinement around the current best by sampling multiplicative
     neighbors. Repeatable for a few iterations.

Outputs a CSV summary of all trials and re‑runs the best configuration to
save the final model + (optionally) plot.

Example:
  python tune_lambda.py \
    --csv Michael_calibration_frame.csv \
    --out-prefix Michael_reg_model \
    --weighted 1 \
    --plot-best 1 \
    --min-lam 1e-6 --max-lam 1e-1 --coarse-points 8 --refine-iters 2

Assumptions about fit_gaze_poly.py stdout (typical lines):
  R^2_x: 0.9940   R^2_y: 0.9680
  LOO   |x-error|: {'median': 0.1117, 'mean': 0.1077, ...}
  LOO   |y-error|: {'median': 0.1989, 'mean': 0.2387, ...}

If your script prints slightly different keys, adjust the regex below.
"""

import argparse
import csv
import math
import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import ast

# --- Regex patterns to parse fit_gaze_poly.py output ---
R2_LINE = re.compile(r"R\^2_x:\s*([0-9.]+)\s*R\^2_y:\s*([0-9.]+)")
XERR_LINE = re.compile(r"LOO\s*\|x-error\|:\s*(\{.*?\})")
YERR_LINE = re.compile(r"LOO\s*\|y-error\|:\s*(\{.*?\})")

@dataclass
class TrialResult:
    lam: float
    r2x: Optional[float]
    r2y: Optional[float]
    x_err: Dict[str, float]
    y_err: Dict[str, float]
    score: float  # objective = x_mean + y_mean
    model_path: str
    cmd: str
    ok: bool
    raw_stdout: str
    raw_stderr: str


def fmt_lam(l: float) -> str:
    """Format lambda into a filename‑friendly token with high precision."""
    s = f"{l:.17g}"
    s = s.replace("+", "")
    return s


def run_fit(script_path: str, csv_path: str, out_model: str, lam: float,
            weighted: int, plot: int, extra_args: Optional[str]) -> Tuple[bool, str, str]:
    cmd = [
        "python", script_path,
        "--csv", csv_path,
        "--out", out_model,
        "--lam", str(lam),
        "--weighted", str(weighted),
        "--plot", str(plot),
    ]
    if extra_args:
        cmd.extend(shlex.split(extra_args))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return (proc.returncode == 0), proc.stdout, proc.stderr


def parse_stdout(stdout: str) -> Tuple[Optional[float], Optional[float], Dict[str, float], Dict[str, float]]:
    # R^2
    r2x = r2y = None
    m = R2_LINE.search(stdout)
    if m:
        r2x = float(m.group(1))
        r2y = float(m.group(2))

    # error dicts
    x_err: Dict[str, float] = {}
    y_err: Dict[str, float] = {}
    mx = XERR_LINE.search(stdout)
    my = YERR_LINE.search(stdout)
    if mx:
        try:
            x_err = ast.literal_eval(mx.group(1))
        except Exception:
            x_err = {}
    if my:
        try:
            y_err = ast.literal_eval(my.group(1))
        except Exception:
            y_err = {}

    return r2x, r2y, x_err, y_err


def try_one(script_path: str, csv_path: str, out_dir: str, lam: float, weighted: int,
            plot: int, extra_args: Optional[str], x_weight: float, y_weight: float) -> TrialResult:
    os.makedirs(out_dir, exist_ok=True)
    out_model = os.path.join(out_dir, f"model_lam_{fmt_lam(lam)}.json")
    ok, stdout, stderr = run_fit(script_path, csv_path, out_model, lam, weighted, plot, extra_args)
    r2x, r2y, x_err, y_err = parse_stdout(stdout)

    x_mean = x_err.get("mean", float("inf"))
    y_mean = y_err.get("mean", float("inf"))
    score = x_weight * x_mean + y_weight * y_mean

    return TrialResult(
        lam=lam,
        r2x=r2x, r2y=r2y,
        x_err=x_err, y_err=y_err,
        score=score,
        model_path=out_model,
        cmd=f"python {script_path} --csv {csv_path} --out {out_model} --lam {lam} --weighted {weighted} --plot {plot} {extra_args or ''}",
        ok=ok,
        raw_stdout=stdout,
        raw_stderr=stderr,
    )


def logspace(min_lam: float, max_lam: float, num: int) -> List[float]:
    if num <= 1:
        return [min_lam]
    lo = math.log10(min_lam)
    hi = math.log10(max_lam)
    step = (hi - lo) / (num - 1)
    return [10 ** (lo + i * step) for i in range(num)]


def unique_sorted(vals: List[float]) -> List[float]:
    seen = set()
    out = []
    for v in sorted(vals):
        k = round(math.log10(v), 12)
        if k in seen:
            continue
        seen.add(k)
        out.append(v)
    return out


def refine_candidates(best: float, factor: float, neighbors: int,
                      min_lam: float, max_lam: float) -> List[float]:
    # multiplicative band around best: best * factor^k for k in [-n..n]
    cands = []
    for k in range(-neighbors, neighbors + 1):
        v = best * (factor ** k)
        if v < min_lam or v > max_lam:
            continue
        cands.append(v)
    return unique_sorted(cands)


def write_csv(results: List[TrialResult], path: str) -> None:
    fieldnames = [
        "lambda","score","x_mean","y_mean","r2x","r2y","model_path","ok","cmd"
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in sorted(results, key=lambda z: z.lam):
            w.writerow({
                "lambda": r.lam,
                "score": r.score,
                "x_mean": r.x_err.get("mean"),
                "y_mean": r.y_err.get("mean"),
                "r2x": r.r2x,
                "r2y": r.r2y,
                "model_path": r.model_path,
                "ok": r.ok,
                "cmd": r.cmd,
            })


def choose_best(results: List[TrialResult]) -> TrialResult:
    # Prefer valid parses; break ties by lower lambda to avoid overfitting bias if equal
    valid = [r for r in results if math.isfinite(r.score)]
    if not valid:
        # fallback: pick the one with minimal score even if inf
        return sorted(results, key=lambda r: (r.score, r.lam))[0]
    return sorted(valid, key=lambda r: (r.score, r.lam))[0]


def main():
    ap = argparse.ArgumentParser(description="Lambda tuner for fit_gaze_poly.py")
    ap.add_argument("--csv", required=True, help="Calibration CSV path")
    ap.add_argument("--out-prefix", required=True, help="Prefix for outputs (dir + filenames)")
    ap.add_argument("--script", default="fit_gaze_poly.py", help="Path to fit_gaze_poly.py")
    ap.add_argument("--weighted", type=int, default=1, help="Pass-through to fit_gaze_poly --weighted")
    ap.add_argument("--plot-best", type=int, default=1, help="Re-run best lambda with plotting enabled")
    ap.add_argument("--min-lam", type=float, default=1e-6)
    ap.add_argument("--max-lam", type=float, default=1e-1)
    ap.add_argument("--coarse-points", type=int, default=8, help="# of points in coarse log sweep")
    ap.add_argument("--refine-iters", type=int, default=2, help="# of refinement rounds around best")
    ap.add_argument("--refine-factor", type=float, default=3.0, help="Multiplicative spacing for neighbors in refinement")
    ap.add_argument("--neighbors", type=int, default=2, help="# of neighbors on each side during refinement")
    ap.add_argument("--workdir", default="lambda_search", help="Directory to store intermediate models & CSV")
    ap.add_argument("--extra-args", default=None, help="Additional arguments to pass to fit_gaze_poly (quoted)")
    ap.add_argument("--x-weight", type=float, default=1.0, help="Weight for x_mean in the score")
    ap.add_argument("--y-weight", type=float, default=1.0, help="Weight for y_mean in the score")
    args = ap.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    # We'll keep intermediate models under a subdirectory named after out‑prefix
    tag = os.path.basename(args.out_prefix)
    trial_dir = os.path.join(args.workdir, f"trials_{tag}")
    os.makedirs(trial_dir, exist_ok=True)

    results: List[TrialResult] = []

    # 1) Coarse sweep
    coarse = logspace(args.min_lam, args.max_lam, args.coarse_points)
    for lam in coarse:
        r = try_one(args.script, args.csv, trial_dir, lam, args.weighted, 0, args.extra_args, args.x_weight, args.y_weight)
        results.append(r)

    best = choose_best(results)

    # 2) Refinement rounds
    for _ in range(args.refine_iters):
        neighborhood = refine_candidates(best.lam, args.refine_factor, args.neighbors, args.min_lam, args.max_lam)
        # Skip ones we've already tried (by ~log10 equality)
        tried_keys = {round(math.log10(r.lam), 12) for r in results}
        cands = [lam for lam in neighborhood if round(math.log10(lam), 12) not in tried_keys]
        if not cands:
            break
        for lam in cands:
            r = try_one(args.script, args.csv, trial_dir, lam, args.weighted, 0, args.extra_args, args.x_weight, args.y_weight)
            results.append(r)
        best = choose_best(results)

    # Write summary CSV
    csv_path = os.path.join(args.workdir, f"{tag}_lambda_search_summary.csv")
    write_csv(results, csv_path)

    # Re‑run best with plotting & final output name
    best_model = f"{args.out_prefix}_lam{fmt_lam(best.lam)}.json"
    ok, stdout, stderr = run_fit(args.script, args.csv, best_model, best.lam, args.weighted, args.plot_best, args.extra_args)

    # Also write a tiny text report
    report_path = os.path.join(args.workdir, f"{tag}_best.txt")
    with open(report_path, "w") as f:
        f.write(f"Best lambda: {best.lam}\n")
        f.write(f"Scoring weights: x_weight={args.x_weight}, y_weight={args.y_weight}\n")
        f.write(f"Score (x_weight*x_mean + y_weight*y_mean): {best.score}\n")
        f.write(f"x_err: {best.x_err}\n")
        f.write(f"y_err: {best.y_err}\n")
        f.write(f"r2x: {best.r2x}, r2y: {best.r2y}\n")
        f.write(f"Intermediate models dir: {trial_dir}\n")
        f.write(f"Summary CSV: {csv_path}\n")
        f.write(f"Final model saved to: {best_model}\n")

    print("==== Lambda Tuning Complete ====")
    print(f"Best lambda: {best.lam}")
    print(f"Scoring weights: x_weight={args.x_weight}, y_weight={args.y_weight}")
    print(f"Score (x_weight*x_mean + y_weight*y_mean): {best.score}")
    print(f"Final model: {best_model}")
    print(f"Summary CSV: {csv_path}")
    print(f"Report: {report_path}")
    # Write a small replay script to reproduce the best run exactly
    replay_path = os.path.join(args.workdir, f"{tag}_replay_best.sh")
    with open(replay_path, "w") as rf:
        rf.write("#!/usr/bin/env bash\nset -euo pipefail\n")
        rf.write(
            f'python {args.script} --csv "{args.csv}" --out "{args.out_prefix}_lam{fmt_lam(best.lam)}.json" '
            f"--lam {best.lam:.17g} --weighted {args.weighted} --plot 1 {args.extra_args or ''}\n"
        )
    os.chmod(replay_path, 0o755)
    print(f"Replay script: {replay_path}")


if __name__ == "__main__":
    main()
