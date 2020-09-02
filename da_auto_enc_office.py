from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from keras.layers import Lambda, Reshape, Input, Dense, Conv2D, MaxPool2D, Flatten, UpSampling2D, concatenate
from keras.models import Model, Sequential
from keras.datasets import mnist
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model, to_categorical
from keras import backend as K
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
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
from data_prep_office import *
from da_auto_encoder import *
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#Loading Office dataß
X, y = load_office_data()
domains = ['amazon', 'dslr', 'webcam']
source = domains[0]
target = domains[1]

x_s_train, x_s_test, y_s_train, y_s_test = train_test_split(X[source], y[source], stratify = y[source], test_size = 0.2)
x_t_train, x_t_test, y_t_train, y_t_test = train_test_split(X[target], y[target], stratify = y[target], test_size = 0.2)


#upsampling by class to match the number of samples in both source and target
x_s_train, y_s_train, x_t_train, y_t_train = upsample_by_class(x_s_train, y_s_train, x_t_train, y_t_train)
x_s_test, y_s_test, x_t_test, y_t_test = upsample_by_class(x_s_test, y_s_test, x_t_test, y_t_test)       

y_s_train.shape
x_s_test.shape
y_t_test.shape

n_classes = y[source].shape[1]
y_s_train = to_categorical(y_s_train, num_classes=n_classes)
y_s_test = to_categorical(y_s_test, num_classes=n_classes)
y_t_train = to_categorical(y_t_train, num_classes=n_classes)
y_t_test = to_categorical(y_t_test, num_classes=n_classes)

# plt.imshow(x_t_train[-4])
#################

# network parameters
s_input_shape = x_s_train.shape[1:]
t_input_shape = x_t_train.shape[1:]
n_samples = x_s_train.shape[0]


# intermediate_dim = 512
batch_size = 64
latent_dim = 100



s_ae = AutoEncoder_Office(s_input_shape,  output_layer_name = f'{source}_output' , latent_dim = latent_dim)
t_ae = AutoEncoder_Office(t_input_shape, output_layer_name = f'{target}_output' , latent_dim = latent_dim)
t_ae.input_shape
stae = S_T_AE(s_ae, t_ae, n_classes= n_classes, latent_dim=latent_dim)
stae.s_encoder.summary()
stae.ae_source_target.summary()


# stae.ae_source_target.summary()
epochs = 200
losses = {f'{source}_output': stae.s_ae_loss, f'{target}_output': stae.t_ae_loss}
opt = Adam(lr=0.00001)
stae.ae_source_target.compile(loss = losses, optimizer=opt)
plot_model(stae.ae_source_target, to_file='ae_enc_dec.png', show_shapes=True)


hist = stae.ae_source_target.fit([x_s_train, x_t_train], [x_s_train, x_t_train],
                            epochs=epochs,
                            verbose=2,
                            batch_size=batch_size
                            #validation_data=([x_s_test, x_t_test], [x_s_test, x_t_test])
                            )

# plt.plot(hist.history['loss'])
stae.ae_source_target.save_weights(f'model_weights/ae_{source}_to_{target}.h5')

stae.ae_source_target.load_weights(f'model_weights/ae_{source}_to_{target}.h5')

# recons = stae.ae_source_target.predict([x_s_test, x_t_test])[0]
# plt.imshow(recons[16,:,:,0])
# plt.savefig('sample.png')
# plt.show()


stae.build_s_enc_cls_model()
for layer in stae.s_enc_cls_model.layers:
    layer.trainable = False

stae.classifier.trainable = True
opt = Adam(lr = 0.0001)
stae.s_enc_cls_model.compile(loss='categorical_crossentropy', optimizer=opt)
stae.s_enc_cls_model.summary()
clbck = EarlyStopping(patience=5)
stae.s_enc_cls_model.fit(x_s_train, y_s_train,
                    epochs = 50,
                    verbose=2,
                    callbacks= [clbck],
                    batch_size=64,
                    validation_data=(x_s_test, y_s_test))

s_pred = stae.s_enc_cls_model.predict(x_s_test)
s_score = accuracy_score(y_s_test.argmax(1), s_pred.argmax(1))
print(f'Source Score {source} to {target}: {s_score}')



stae.build_t_enc_cls_model()
stae.t_enc_cls_model.summary()
stae.t_enc_cls_model.compile(loss='categorical_crossentropy', optimizer='adam')

# t_pred = stae.t_enc_cls_model.predict(x_t_train)
# t_score = accuracy_score(y_t_train.argmax(1), t_pred.argmax(1))
# print(f"Target sore, Before Fine tune: {t_score}")

stae.t_enc_cls_model.fit(x_t_test, y_t_test,
                    epochs=5,
                    verbose = 2,
                    batch_size=64,
                    validation_data=(x_t_train, y_t_train))

t_pred = stae.t_enc_cls_model.predict(x_t_train)
t_score = accuracy_score(y_t_train.argmax(1), t_pred.argmax(1))
print(f"Target score {source} to {target}, After Fine tune: {t_score}")
