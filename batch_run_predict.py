import subprocess
jobs=[
  {"in_csv":"Ous_f_output.csv",
  "model":"Ous_f_output1.json",
  "start":2103,
  "end":2287,
  "out_csv":"Ous_f_predict1.csv"
  },
  {"in_csv":"Ous_f_output.csv",
  "model":"Ous_f_output2.json",
  "start":3666,
  "end":3909,
  "out_csv":"Ous_f_predict2.csv"
  },
  {"in_csv":"Ous_f_output.csv",
  "model":"Ous_f_output3.json",
  "start":6325,
  "end":6509,
  "out_csv":"Ous_f_predict3.csv"
  },
  # {"in_csv":"Elizabeth_S_output.csv",
  # "model":"Elizabeth_S_output1.json",
  # "start":3129,
  # "end":3356,
  # "out_csv":"Elizabeth_S_predict4.csv"
  # },
  # {"in_csv":"Elizabeth_S_output.csv",
  # "model":"Elizabeth_S_output2.json",
  # "start":5650,
  # "end":6077,
  # "out_csv":"Elizabeth_S_predict5.csv"
  # },
  # {"in_csv":"Elizabeth_S_output.csv",
  # "model":"Elizabeth_S_output2.json",
  # "start":6123,
  # "end":6424,
  # "out_csv":"Elizabeth_S_predict6.csv"
  # },
  # {"in_csv":"Elizabeth_S_output.csv",
  # "model":"Elizabeth_S_output2.json",
  # "start":6592,
  # "end":6830,
  # "out_csv":"Elizabeth_S_predict7.csv"
  # },
  # {"in_csv":"Elizabeth_S_output.csv",
  # "model":"Elizabeth_S_output3.json",
  # "start":9159,
  # "end":9562,
  # "out_csv":"Elizabeth_S_predict8.csv"
  # },
  # {"in_csv":"Elizabeth_S_output.csv",
  # "model":"Elizabeth_S_output3.json",
  # "start":9602,
  # "end":10000,
  # "out_csv":"Elizabeth_S_predict9.csv"
  # },
  # {"in_csv":"Elizabeth_S_output.csv",
  # "model":"Elizabeth_S_output3.json",
  # "start":10065,
  # "end":10315,
  # "out_csv":"Elizabeth_S_predict10.csv"
  # },
]

for job in jobs:
  cmd = [
    "python", "test_predict.py",
    "--in_csv", job["in_csv"],
    "--model_json",job["model"],
    "--frame_start",str(job["start"]),
    "--frame_end",str(job["end"]),
    "--smoothing","ema",
    "--out_csv",job["out_csv"]
  ]
  print("Running:", " ".join(cmd))
  subprocess.run(cmd, check=True)