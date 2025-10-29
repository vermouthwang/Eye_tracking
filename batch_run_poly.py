import subprocess
jobs=[
  {
    "csv":"Ous_fb_output1.csv",
    "out":"Ous_f_output1.json",
    "lam":0.2,
  },
  {
    "csv":"Ous_f_output2.csv",
    "out":"Ous_f_output2.json",
    "lam":0.02,
  },
  {
    "csv":"Ous_f_output3.csv",
    "out":"Ous_f_output3.json",
    "lam":0.005,
  },
  # {
  #   "csv":"Elizabeth_output4.csv",
  #   "out":"Elizabeth_output4.json",
  #   "lam":0.005,
  # },
  # {
  #   "csv":"Elizabeth_output5_6.csv",
  #   "out":"Elizabeth_output5_6.json",
  #   "lam":0.001,
  # },
  # {
  #   "csv":"Elizabeth_output7.csv",
  #   "out":"Elizabeth_output7.json",
  #   "lam":0.001,
  # },
  # {
  #   "csv":"Elizabeth_output8_9.csv",
  #   "out":"Elizabeth_output8_9.json",
  #   "lam":0.001,
  # },
  # {
  #   "csv":"Elizabeth_output10.csv",
  #   "out":"Elizabeth_output10.json",
  #   "lam":0.001,
  # },
]

for job in jobs:
  cmd = [
    "python", "fit_gaze_poly.py",
    "--csv", job["csv"],
    "--out", job["out"],
    "--lam", str(job["lam"]),
    "--weighted", "1",
    "--plot","1",
  ]
  print("Running:", " ".join(cmd))
  subprocess.run(cmd, check=True)
  # python fit_gaze_poly.py --csv Henry_B_output2.csv \
  #     --out Henry_B_output1.json \
  #     --lam 1e-4 \
  #     --weighted 1 \
  #     --plot 1