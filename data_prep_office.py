import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import image
from os import listdir
from PIL import Image

office_dir = 'office_data/'
domains = ['amazon', 'dslr', 'webcam']
classes = listdir(f'{office_dir}{domains[0]}/images')
fs = listdir(f'{office_dir}{domains[0]}/images/{classes[0]}/')

img = Image.open(f'{office_dir}{domains[0]}/images/{classes[0]}/{fs[14]}')
img_r = img.resize((100,100))
img_np = np.array(img_r)


X = {}
y = {}

for d in domains:

    X_d = []
    y_d = []
    # Loading Amazon data
    dir_d = f'{office_dir}{d}/images'
    for c in listdir(dir_d):
        files = listdir(f'{dir_d}/{c}/')
        for f in files:
            img  = Image.open(f'{dir_d}/{c}/{f}')
            img_r = np.array(img.resize((100,100)))
            X_d.append(img_r)
            y_d.append(c)

    X_d = np.array(X_d)
    y_d = np.array(y_d)
    X[d] = X_d
    y[d] = y_d

X[].shape