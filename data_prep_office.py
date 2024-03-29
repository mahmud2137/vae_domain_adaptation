import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import image
from os import listdir
from PIL import Image
from sklearn.preprocessing import OneHotEncoder
from torchvision.datasets import ImageFolder

office_dir = 'office_data/'
domains = ['amazon', 'dslr', 'webcam']
classes = listdir(f'{office_dir}{domains[0]}/images')
classes.sort()
fs = listdir(f'{office_dir}{domains[0]}/images/{classes[0]}/')

img = Image.open(f'{office_dir}{domains[0]}/images/{classes[0]}/{fs[14]}')
img_r = img.resize((100,100))
img_np = np.array(img_r)


def load_office_data(data_dir = 'office_data/', domains = ['amazon', 'dslr', 'webcam']):
    '''
    Loading office data
    data_dir: directory address of office data
    domains: different domains of data, which are also the name of the folders
    return: X, dictionary containing feature data
            y, dictionary containing labels
    '''
    X = {}
    y = {}
    classes = listdir(f'{data_dir}{domains[0]}/images')
    classes.sort()
    enc = OneHotEncoder()
    enc.fit(np.reshape(classes, (-1,1)))

    for d in domains:

        X_d = []
        y_d = []
        # Loading Amazon data
        dir_d = f'{data_dir}{d}/images'
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
        y[d] =  enc.transform(y_d.reshape(-1,1)).toarray()

    return X, y


# X, y = load_office_data()
# # y_ohe =  enc.transform(y['amazon'].reshape(-1,1)).toarray()
# # enc.categories_

# X['amazon'].shape
# classes
# y['webcam']
# plt.imshow(X['webcam'][150])

if __name__ == '__main__':
    amz = ImageFolder('office_data/amazon/images')
    print(len(amz))
