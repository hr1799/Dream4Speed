import numpy as np

EXPT_NAME = "austria_vel0_01"
# EXPT_NAME = "austria_vel0_01_smoothsteer_speed"

MAP_NAMES = ["austria", "barcelona", "columbia", "gbr"]

for map in MAP_NAMES:
    # Load CSV
    csv_path = "eval_logs/"+EXPT_NAME+"_"+map+".csv"
    # data has a header row
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    # #,Progress,Time,Reward
    average_progress = np.mean(data[:,1]).round(2)
    average_time = np.mean(data[:,2]).round(2)
    best_progress = np.max(data[:,1]).round(2)
    best_time = np.max(data[:,2]).round(2)
    # Map, Average Progress, Average Time, Best Progress, Best Time
    print(map, ",", average_progress, ",", average_time, ",", best_progress, ",", best_time)
    