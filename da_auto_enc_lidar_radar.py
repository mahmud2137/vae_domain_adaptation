from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from keras.layers import Lambda, Reshape, Input, Dense, Conv2D, MaxPool2D, Flatten, UpSampling2D, concatenate
from keras.models import Model, Sequential
from keras.datasets import mnist
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import EarlyStopping
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model, to_categorical
from keras import backend as K
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from scipy.io import loadmat
from itertools import permutations

import utils
from functools import partial
import h5py

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from auto_encoders import *
from data_prep_lidar_radar import *
from da_auto_encoder import *
os.environ["CUDA_VISIBLE_DEVICES"]="0"

X_lidar, y_lidar, X_radar, y_radar = load_sandpaper_data("sandpaper_data/")
CGrade_to_particle_size = {120:116, 150:93, 180:78, 220:66, 240:53.5, 320:36, 400:23.6, 600:16, 800:12.2, 1000:9.2, 1200:6.5}
X_radar = np.expand_dims(X_radar, axis=-1)
X_lidar = np.expand_dims(X_lidar, axis=-1)

# le  = LabelEncoder()
# y_radar_ = le.fit_transform(y_radar)
# y_radar_ = to_categorical(y_radar_)

y_radar_ = np.array([CGrade_to_particle_size[x] for x in y_radar])
y_lidar_ = np.array([CGrade_to_particle_size[x] for x in y_lidar])

domains = ['radar', 'lidar']
X, y = {}, {}
X[domains[0]] = X_radar
X[domains[1]] = X_lidar
y[domains[0]] = y_radar_
y[domains[1]] = y_lidar_

src = domains[0]
tgt = domains[1]

x_s_train, x_s_test, y_s_train, y_s_test = train_test_split(X[src], y[src], stratify = y[src], test_size = 0.2)
x_t_train, x_t_test, y_t_train, y_t_test = train_test_split(X[tgt], y[tgt], stratify = y[tgt], test_size = 0.2)


#upsampling by class to match the number of samples in both source and target
x_s_train, y_s_train, x_t_train, y_t_train = upsample_by_class(x_s_train, y_s_train, x_t_train, y_t_train)
x_s_test, y_s_test, x_t_test, y_t_test = upsample_by_class(x_s_test, y_s_test, x_t_test, y_t_test)       

# network parameters
s_input_shape = x_s_train.shape[1:]
t_input_shape = x_t_train.shape[1:]

batch_size = 64
latent_dim = 100

s_ae = AutoEncoder_Radar(s_input_shape, output_layer_name= f'{src}_output', latent_dim=latent_dim)
t_ae = AutoEncoder_Lidar(t_input_shape, output_layer_name= f'{tgt}_output', latent_dim= latent_dim)

class Ae_Radar_Lidar(S_T_AE):
    def __init__(self, s_ae, t_ae, n_classes, latent_dim=100, beta=0.5):
        super().__init__(s_ae, t_ae, n_classes, latent_dim, beta)

    def build_classifier(self):
        model = Sequential()
        model.add(Dense(self.latent_dim, activation='relu', input_dim=self.latent_dim))
        model.add(Dense(1, activation='linear'))

        feats = Input(shape=(self.latent_dim,))
        class_label = model(feats)
        self.classifier = Model(feats, class_label)

# stae = S_T_AE(s_ae, t_ae, n_classes=1, beta = 0.004)
stae = Ae_Radar_Lidar(s_ae, t_ae, n_classes=1, beta = 0.004)
stae.t_encoder.summary()
stae.ae_source_target.summary()

epochs = 10
losses = {f'{src}_output': stae.s_ae_loss, f'{tgt}_output': stae.t_ae_loss}
opt = Adam(lr=0.0001)
stae.ae_source_target.compile(loss = losses, optimizer=opt)
plot_model(stae.ae_source_target, to_file='ae_enc_dec.png', show_shapes=True)


hist = stae.ae_source_target.fit([x_s_train, x_t_train], [x_s_train, x_t_train],
                            epochs=epochs,
                            verbose=2,
                            batch_size=batch_size
                            #validation_data=([x_s_test, x_t_test], [x_s_test, x_t_test])
                            )

# plt.plot(hist.history['loss'])
stae.ae_source_target.save_weights(f'model_weights/ae_{src}_to_{tgt}.h5')

stae.ae_source_target.load_weights(f'model_weights/ae_{src}_to_{tgt}.h5')


stae.build_s_enc_cls_model()
for layer in stae.s_enc_cls_model.layers:
    layer.trainable = False

stae.classifier.trainable = True
opt = Adam(lr = 0.00001)
stae.s_enc_cls_model.compile(loss='mse', optimizer=opt)
# stae.s_enc_cls_model.summary()
clbck = EarlyStopping(patience=20)
stae.s_enc_cls_model.fit(x_s_train, y_s_train,
                    epochs = 30,
                    verbose=2,
                    callbacks= [clbck],
                    batch_size=64,
                    validation_split = 0.1)

s_pred = stae.s_enc_cls_model.predict(x_s_test)
s_r2_score = r2_score(y_s_test, s_pred)
print(f'Source R2 Score {src} to {tgt}: {s_r2_score}')


stae.build_t_enc_cls_model()
stae.t_enc_cls_model.summary()
stae.t_enc_cls_model.compile(loss='mse', optimizer= 'adam')



stae.t_enc_cls_model.fit(x_t_test, y_t_test,
                    epochs=30,
                    verbose = 2,
                    batch_size=64,
                    validation_data=(x_t_train, y_t_train))

t_pred = stae.t_enc_cls_model.predict(x_t_train)
t_r2_score = r2_score(y_t_train, t_pred)
print(f"Target score {src} to {tgt}, After Fine tune: {t_r2_score}")
