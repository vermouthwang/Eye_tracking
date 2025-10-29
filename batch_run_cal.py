# batch_run_stats.py
import subprocess

jobs = [
    {
        "csv": "Ous_fb_output.csv",
        "window": 25,
        "out": "Ous_fb_output1.csv",
        "middleFrames": [1610, 349, 529, 710, 889, 1069, 1250, 1430, 1791],
    },
    # {
    #     "csv": "Ous_f_output.csv",
    #     "window": 15,
    #     "out": "Ous_f_output2.csv",
    #     "middleFrames": [2418, 2520, 2622, 2724, 2826, 2928, 3030, 3132, 3336],
    # },
    # {
    #     "csv": "Ous_f_output.csv",
    #     "window": 18,
    #     "out": "Ous_f_output3.csv",
    #     "middleFrames": [5102, 5204, 5306, 5408, 5510, 5612, 5714, 5816, 6120],
    # },
    # {
    #     "csv": "Elizabeth_output.csv",
    #     "window": 20,
    #     "out": "Elizabeth_output4.csv",
    #     "middleFrames": [8922, 7661, 7841, 8022, 8201, 8381, 8562, 8742, 9103],
    # },
    # {
    #     "csv": "Elizabeth_output.csv",
    #     "window": 25,
    #     "out": "Elizabeth_output5_6.csv",
    #     "middleFrames": [11307, 10046, 10226, 10407, 10586, 10766, 10947, 11127, 11488],
    # },
    # {
    #     "csv": "Elizabeth_output.csv",
    #     "window": 25,
    #     "out": "Elizabeth_output7.csv",
    #     "middleFrames": [14613, 13352, 13532, 13713, 13892, 14072, 14253, 14433, 14794],
    # },
    # {
    #     "csv": "Elizabeth_output.csv",
    #     "window": 18,
    #     "out": "Elizabeth_output8_9.csv",
    #     "middleFrames": [16908, 15647, 15827, 16008, 16187, 16367, 16548, 16728, 17089],
    # },
    # {
    #     "csv": "Elizabeth_output.csv",
    #     "window": 25,
    #     "out": "Elizabeth_output10.csv",
    #     "middleFrames": [20089, 18828, 19008, 19189, 19268, 19548, 19729, 19909, 20270],
    # },

]

for job in jobs:
    cmd = [
        "python", "calc_calibration_stats.py",
        "--csv", job["csv"],
        "--window", str(job["window"]),
        "--out", job["out"],
        "--middleFrames", *map(str, job["middleFrames"])
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
