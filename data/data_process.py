import numpy as np
import os

path = "/home/gene/Downloads/SinD"
files = os.listdir(path)
track_id = np.array([])
agent_type = np.array([])
frame_id = np.array([])
timestamp_ms = np.array([])
x = np.array([])
y = np.array([])
vx = np.array([])
vy = np.array([])
ax = np.array([])
ay = np.array([])
for file in files:
    f = path+'/'+file+'/Ped_smoothed_tracks.csv'
    tr,ag = np.loadtxt(f,dtype=str,delimiter=',',unpack=True,usecols=(0,3),comments='track_id')
    track_id = np.concatenate((track_id, tr), axis=0)
    agent_type = np.concatenate((agent_type, ag), axis=0)
    fr, ti, xa, ya, vxa, vya, axa, aya = np.loadtxt(f,delimiter=',',unpack=True,usecols=(1,2,4,5,6,7,8,9),comments='track_id')
    frame_id = np.concatenate((frame_id, fr), axis=0)
    timestamp_ms = np.concatenate((timestamp_ms, ti), axis=0)
    x = np.concatenate((x, xa), axis=0)
    y = np.concatenate((y, ya), axis=0)
    vx = np.concatenate((vx, vxa), axis=0)
    vy = np.concatenate((vy, vya), axis=0)
    axa =  np.concatenate((ax, axa), axis=0)
    aya =  np.concatenate((ax, aya), axis=0)
print(x.shape)


#SinD = np.loadtxt("/home/gene/Downloads/SinD", delimiter)