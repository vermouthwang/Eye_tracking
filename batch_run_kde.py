import subprocess
jobs = [
  {
    "in_csv":"Ous_f_predict1.csv",
    "out":"Ous_f_1.png"
  },
  {
    "in_csv":"Ous_f_predict2.csv",
    "out":"Ous_f_2.png"
  },
  {
    "in_csv":"Ous_f_predict3.csv",
    "out":"Ous_f_3.png"
  },
  # {
  #   "in_csv":"Elizabeth_S_predict4.csv",
  #   "out":"Elizabeth_S_4.png"
  # },
  # {
  #   "in_csv":"Elizabeth_S_predict5.csv",
  #   "out":"Elizabeth_S_5.png"
  # },
  # {
  #   "in_csv":"Elizabeth_S_predict6.csv",
  #   "out":"Elizabeth_S_6.png"
  # },
  # {
  #   "in_csv":"Elizabeth_S_predict6.csv",
  #   "out":"Elizabeth_S_6.png"
  # },
  # {
  #   "in_csv":"Elizabeth_S_predict7.csv",
  #   "out":"Elizabeth_S_7.png"
  # },
  # {
  #   "in_csv":"Elizabeth_S_predict8.csv",
  #   "out":"Elizabeth_S_8.png"
  # },
  # {
  #   "in_csv":"Elizabeth_S_predict9.csv",
  #   "out":"Elizabeth_S_9.png"
  # },
  # {
  #   "in_csv":"Elizabeth_S_predict10.csv",
  #   "out":"Elizabeth_S_10.png"
  # },
  # {
  #   "in_csv":"Ous_predict11x.csv",
  #   "out":"Ous_11.png"
  # },
  # {
  #   "in_csv":"Laila_B_5_predict.csv",
  #   "out":"Laila_B_5.png"
  # }, 
  # {
  #   "in_csv":"Laila_B_6_predict.csv",
  #   "out":"Laila_B_6.png"
  # },
  # {
  #   "in_csv":"Laila_B_7_predict.csv",
  #   "out":"Laila_B_7.png"
  # },
  # {
  #   "in_csv":"Laila_B_8_predict.csv",
  #   "out":"Laila_B_8.png"
  # },
  # {
  #   "in_csv":"Laila_B_9_predict.csv",
  #   "out":"Laila_B_9.png"
  # },
  # {
  #   "in_csv":"Laila_S_1_predict.csv",
  #   "out":"Laila_S_1.png"
  # },
  # {
  #   "in_csv":"Laila_S_1_predict.csv",
  #   "out":"Laila_S_1.png"
  # },
  # {
  #   "in_csv":"Laila_S_1_predict.csv",
  #   "out":"Laila_S_1.png"
  # },
  # {
  #   "in_csv":"Laila_S_2_predict.csv",
  #   "out":"Laila_S_2.png"
  # },
  # {
  #   "in_csv":"Laila_S_3_predict.csv",
  #   "out":"Laila_S_3.png"
  # },
  # {
  #   "in_csv":"Laila_S_4_predict.csv",
  #   "out":"Laila_S_4.png"
  # },
  # {
  #   "in_csv":"Laila_S_5_predict.csv",
  #   "out":"Laila_S_5.png"
  # },
  # {
  #   "in_csv":"Laila_S_6_predict.csv",
  #   "out":"Laila_S_6.png"
  # },
  # {
  #   "in_csv":"Laila_S_7_predict.csv",
  #   "out":"Laila_S_7.png"
  # },
  # {
  #   "in_csv":"Laila_S_8_predict.csv",
  #   "out":"Laila_S_8.png"
  # },
  # {
  #   "in_csv":"Laila_S_9_predict.csv",
  #   "out":"Laila_S_9.png"
  # },
]

for job in jobs:
  cmd = [
    "python", "gaze_kde.py",
    "--csv", job["in_csv"],
    "--out",job["out"],
  ]
  print("Running:", " ".join(cmd))
  subprocess.run(cmd, check=True)