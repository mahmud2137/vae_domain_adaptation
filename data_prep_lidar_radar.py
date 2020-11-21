from keras.engine.training import Model
from keras.layers.merge import Concatenate
from auto_encoders import AutoEncoder_Lidar, AutoEncoder_Radar
from keras.layers.core import Activation, Flatten
from matplotlib.pyplot import hot
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Softmax,Input, Dropout, Conv3D, Conv2D, MaxPool3D, Flatten, MaxPool2D, Concatenate
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
# from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
import scipy
import re
import os
import cv2
from auto_encoders import AutoEncoder_Lidar, AutoEncoder_Radar
from da_auto_encoder import *

from sklearn.utils import shuffle

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
                ld_sample.append(nums)
    lider_data = np.array(lider_data)
    return lider_data

# path = 'lider_radar_data/sitting/radar/'

def read_radar_data(path):
    radar_files = os.listdir(path)
    radar_data = []
    for rf in radar_files:
        d = loadmat(path+rf, squeeze_me = True)
        img = d['sImage2Save']['images'].item()

        # img = cv2.resize(x, dsize = (20,24), interpolation=cv2.INTER_CUBIC)
        # img = np.pad(img, ((0,0),(0,0),(0,1)))
        # img = np.reshape(img, (60,160))
        radar_data.append(img)

    radar_data = np.array(radar_data)
    return radar_data


# path = 'sandpaper_data/'
def load_sandpaper_data(path, folder = None):
    sp_grades = os.listdir(path)
    X_lidar = np.array([])
    X_radar = np.array([])
    y_lidar = []
    y_radar = []

    if folder:
        lidar_path = f"{path}{folder}/lidar.csv"
        radar_path = f"{path}{folder}/radar/"
        lidar_data = read_lider_data(lidar_path)
        radar_data = read_radar_data(radar_path)
        X_lidar = np.vstack((X_lidar, lidar_data)) if len(X_lidar) else lidar_data
        X_radar = np.vstack((X_radar, radar_data)) if len(X_radar) else radar_data
        return X_lidar, X_radar
    else:
        for spg in sp_grades:
            lidar_path = f"{path}{spg}/lidar.csv"
            radar_path = f"{path}{spg}/radar/"
            lidar_data = read_lider_data(lidar_path)
            radar_data = read_radar_data(radar_path)
            X_lidar = np.vstack((X_lidar, lidar_data)) if len(X_lidar) else lidar_data
            X_radar = np.vstack((X_radar, radar_data)) if len(X_radar) else radar_data
            y_lidar.extend([int(spg)] * lidar_data.shape[0])
            y_radar.extend([int(spg)] * radar_data.shape[0])

        return X_lidar, np.array(y_lidar), X_radar, np.array(y_radar)

def accuracy_score_one_hot(y_true, y_pred):
    y_pred = np.argmax(y_pred, axis = 1)
    y_true = np.argmax(y_true, axis = 1)
    test_score = sum(y_true == y_pred)/len(y_true)
    return test_score

if __name__ == "__main__":
    X_lider, y_lider, X_radar, y_radar = load_sandpaper_data("sandpaper_data/")
    
    # np.where(y_radar=="1200")
    # a = X_radar[937,:]
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.scatter(a[:,0], a[:,1], a[:,2])
    # ax.set_xlim([0,0.6])
    # ax.set_ylim([0,3])
    # ax.set_zlim([0,1])
    # plt.show()

    # x = X_radar[2,:]


    CGrade_to_particle_size = {120:116, 150:93, 180:78, 220:66, 240:53.5, 320:36, 400:23.6, 600:16, 800:12.2, 1000:9.2, 1200:6.5}
    X_radar = np.expand_dims(X_radar, axis=-1)
    X_lider = np.expand_dims(X_lider, axis=-1)

    # le  = LabelEncoder()
    # y_radar_ = le.fit_transform(y_radar)
    # y_radar_ = to_categorical(y_radar_)

    # y_lider_ = le.fit_transform(y_lider)
    # y_lider_ = to_categorical(y_lider_)


    y_radar_ = np.array([CGrade_to_particle_size[x] for x in y_radar])
    y_lider_ = np.array([CGrade_to_particle_size[x] for x in y_lider])

    # X_lider[1]
    # plt.imshow(X_lider[300])
    n_classes = len(np.unique(y_radar))
    X_r_train, X_r_test, y_r_train, y_r_test = train_test_split(X_radar, y_radar_, test_size = 0.4, shuffle= True)
    X_l_train, X_l_test, y_l_train, y_l_test = train_test_split(X_lider, y_lider_, test_size = 0.4, shuffle= True)

    # Classifier with Radar data
    opt = Adam(lr= 0.0001)
    input_shape = X_radar.shape[1:]


    model_r = Sequential()
    model_r.add(Conv3D(16, (3,3,3), padding = 'same' , strides = 1, activation = 'relu' , input_shape= input_shape))
    model_r.add(Conv3D(16, (3,3,3), padding = 'same' , strides = 2, activation = 'relu'))
    model_r.add(MaxPool3D(pool_size = (2,2,2)))
    model_r.add(Dropout(0.4))

    model_r.add(Conv3D(32, (3,3,3), padding = 'same' , strides = 1, activation = 'relu'))
    model_r.add(Conv3D(32, (3,3,3), padding = 'same' , strides = 2, activation = 'relu'))
    model_r.add(MaxPool3D(pool_size = (2,2,1)))
    model_r.add(Dropout(0.4))
    model_r.add(Flatten())

    model_r.add(Dense(100, activation = 'relu'))
    model_r.add(Dense(1, activation = 'linear'))
    model_r.compile(optimizer=opt, loss='mse')
    model_r.summary()
    model_r.fit(X_r_train, y_r_train, batch_size = 64, epochs = 300, validation_split = 0.1)
    model_r.save_weights("model_weights/radar_reg.h5")
    y_pred = model_r.predict(X_r_test)

    # accuracy_score_one_hot(y_r_test, y_pred)
    # y_pred = np.argmax(y_pred, axis = 1)
    # y_true = np.argmax(y_r_test, axis = 1)

    # confusion_matrix(y_true, y_pred)*100/len(y_true)

    y_pred = y_pred.flatten()
    plt.scatter(y_r_test, y_pred)
    plt.title("Average Particle size in microns")
    plt.xlabel("True Size")
    plt.ylabel("Predicted Size")

    r2_radar = r2_score(y_r_test, y_pred)
    print('test score:', r2_radar)

    # Loading Cheetah data
    grass1_l, grass1_r  = load_sandpaper_data(path='cheetah_data/', folder='grass1')
    grass1_r = np.expand_dims(grass1_r, axis=-1)
    grass1_pred = model_r.predict(grass1_r)

    grass2_l, grass2_r  = load_sandpaper_data(path='cheetah_data/', folder='grass2')
    grass2_r = np.expand_dims(grass2_r, axis=-1)
    grass2_pred = model_r.predict(grass2_r)
    plt.plot(grass2_pred)
    
    


    #########################################
    #### Lider model
    ########################################

    n_classes = len(np.unique(y_radar))
    # Classifier with Radar data
    opt = Adam(lr= 0.0001)
    input_shape = X_lider.shape[1:]


    model = Sequential()
    model.add(Conv2D(16, (3,3), padding = 'same' , strides = 1, activation = 'relu' , input_shape= input_shape))
    model.add(Conv2D(16, (3,3), padding = 'same' , strides = 2, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size = (2,2)))
    # model.add(Dropout(0.4))

    model.add(Conv2D(32, (3,3), padding = 'valid' , strides = 1, activation = 'relu'))
    model.add(Conv2D(32, (3,3), padding = 'same' , strides = 2, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size = (2,2)))
    # model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(100, activation = 'relu'))
    model.add(Dense(1, activation = 'linear'))
    model.compile(optimizer=opt, loss='mse')
    model.summary()
    clbck = EarlyStopping(patience=100)
    model.fit(X_l_train, y_l_train, batch_size = 64, callbacks = [clbck],
            epochs = 100, validation_split = 0.1, verbose = 2)
    model.save_weights("model_weights/lidar_reg.h5")
    y_pred = model.predict(X_l_test)
    # accuracy_score_one_hot(y_l_test, y_pred)


    # y_pred = np.argmax(y_pred, axis = 1)
    # y_true = np.argmax(y_l_test, axis = 1)
    # confusion_matrix(y_true, y_pred)

    y_pred = y_pred.flatten()
    plt.scatter(y_l_test, y_pred)
    plt.title("Average Particle size in microns")
    plt.xlabel("True Size")
    plt.ylabel("Predicted Size")
    r_lidar = r2_score(y_l_test, y_pred)

    print('test score:', r_lidar)
    X_l_test.shape 
    ## Cheetah data
    grass1_l = np.expand_dims(grass1_l, axis=-1)
    grass1_l_pred = model.predict(grass1_l)

    grass2_l = np.expand_dims(grass2_l, axis=-1)
    grass2_l_pred = model.predict(grass2_l)

    np.save("grass2_lidar.npy",grass2_l_pred)
     

    #######################################################
    # Lidar and Radar Multimodal Fused Model
    ###################################################
    np.unique(y_radar_)

    X_r_train, y_r_train, X_l_train, y_l_train = upsample_by_class(X_r_train, y_r_train, X_l_train, y_l_train)
    X_r_test, y_r_test, X_l_test, y_l_test = upsample_by_class(X_r_test, y_r_test, X_l_test, y_l_test)   


    input_shape_r = X_radar.shape[1:]
    input_shape_l = X_lider.shape[1:]

    input_layer_radar = Input(shape=input_shape_r, name = 'radar_input')
    input_layer_lidar = Input(shape=input_shape_l, name = 'lidar_input')

    def enc_radar(x):
    
        x = Conv3D(16, (3,3,3), padding = 'same' , strides = 1, activation = 'relu')(x)
        x = Conv3D(16, (3,3,3), padding = 'same' , strides = 2, activation = 'relu')(x)
        x = BatchNormalization()(x)
        x = MaxPool3D(pool_size = (2,2,2))(x)
        # x = Dropout(0.4)(x)

        x = Conv3D(32, (3,3,3), padding = 'same' , strides = 1, activation = 'relu')(x)
        x = Conv3D(32, (3,3,3), padding = 'same' , strides = 2, activation = 'relu')(x)
        x = BatchNormalization()(x)
        x = MaxPool3D(pool_size = (2,2,1))(x)
        # x = Dropout(0.4)(x)

        # x = Conv3D(32, (3,3,3), padding = 'same' , strides = 1, activation = 'relu')(x)
        # x = Conv3D(32, (3,3,3), padding = 'same' , strides = 1, activation = 'relu')(x)
        # x = BatchNormalization()(x)
        # x = MaxPool3D(pool_size = (2,2,1))(x)
        # x = Dropout(0.4)(x)

        x = Flatten()(x)
        return x

    def enc_lidar(x):
    
        x = Conv2D(16, (3,3), padding = 'same' , strides = 1, activation = 'relu')(x)
        x = Conv2D(16, (3,3), padding = 'same' , strides = 2, activation = 'relu')(x)
        x = BatchNormalization()(x)
        x = MaxPool2D(pool_size = (2,2))(x)
        # x = Dropout(0.4)(x)

        x = Conv2D(32, (3,3), padding = 'same' , strides = 1, activation = 'relu')(x)
        x = Conv2D(32, (3,3), padding = 'same' , strides = 2, activation = 'relu')(x)
        x = BatchNormalization()(x)
        x = MaxPool2D(pool_size = (2,2))(x)
        # x = Dropout(0.4)(x)

        # x = Conv2D(32, (3,3), padding = 'same' , strides = 1, activation = 'relu')(x)
        # x = Conv2D(32, (3,3), padding = 'same' , strides = 1, activation = 'relu')(x)
        # x = BatchNormalization()(x)
        # x = MaxPool2D(pool_size = (1,2))(x)
        # # x = Dropout(0.4)(x)
        
        x = Flatten()(x)
        return x


    enc_radar = enc_radar(input_layer_radar)
    enc_lidar = enc_lidar(input_layer_lidar)
    merged = Concatenate()([enc_radar, enc_lidar])
    cls = Sequential([
        # Dense(50, activation = 'relu'),
        # BatchNormalization(),
        Dense(1, activation = 'linear')
    ])

    Fused_Model = Model([input_layer_radar, input_layer_lidar], cls(merged))
    Fused_Model.summary()
    opt = Adam(lr = 0.0001)
    Fused_Model.compile(optimizer=opt, loss = 'mse')
    Fused_Model.fit([X_r_train, X_l_train], y_l_train, batch_size = 128, epochs = 100, validation_split = 0.1, shuffle=True)
    
    np.count_nonzero(y_r_train==116)

    
      
    y_pred = Fused_Model.predict([X_r_test, X_l_test])
    y_pred = y_pred.flatten()
    plt.scatter(y_l_test, y_pred)
    plt.title("Average Particle size in microns")
    plt.xlabel("True Size")
    plt.ylabel("Predicted Size")
    r_fused = r2_score(y_r_test, y_pred)

    print('test score:', r_fused)



