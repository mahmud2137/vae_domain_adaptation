import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.io import loadmat
import scipy
import re
import os

# d_lider = pd.read_csv("lider_radar_data/sitting/lider.csv")


def read_lider_data(path):
    lider_data = []
    ld_sample = []
    started = False

    with open(path, "r") as fp:
        for i, line in enumerate(fp.readlines()):
            nums = re.findall(f'\d+', line)
            nums = [int(x) for x in nums]
            # print(len(nums))
            if(len(nums)==13):
                started = True
                ld_sample = []
                continue
            if(len(nums)==0):
                if started:
                    lider_data.append(ld_sample)
                started = False
                continue
            if started:
                ld_sample.extend(nums)
    lider_data = np.array(lider_data)
    return lider_data

# path = 'lider_radar_data/sitting/radar/'

def read_radar_data(path):
    radar_files = os.listdir(path)
    radar_data = []
    for rf in radar_files:
        d = loadmat(path+rf, squeeze_me = True)
        img = d['sImage2Save']['images'].item()
        radar_data.append(img)

    radar_data = np.array(radar_data)
    return radar_data


path = 'sandpaper_data/'
sp_grades = os.listdir(path)
X_lidar = np.array([])
X_radar = np.array([])
y_lidar = []
y_radar = []
for spg in sp_grades[:2]:
    lidar_path = f"{path}{spg}/lidar.csv"
    radar_path = f"{path}{spg}/radar/"
    lidar_data = read_lider_data(lidar_path)
    radar_data = read_radar_data(radar_path)
    X_lidar = np.vstack((X_lidar, lidar_data)) if len(X_lidar) else lidar_data
    X_radar = np.vstack((X_radar, radar_data)) if len(X_radar) else radar_data
    y_lidar.extend([int(spg)] * lidar_data.shape[0])
    y_radar.extend([int(spg)] * radar_data.shape[0])

X_lidar.shape
X_radar.shape

np.array(X_lidar)