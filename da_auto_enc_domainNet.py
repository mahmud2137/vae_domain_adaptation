from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from keras.layers import Lambda, Reshape, Input, Dense, Conv2D, MaxPool2D, Flatten, UpSampling2D, concatenate
from keras.preprocessing.image import ImageDataGenerator, DirectoryIterator

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
import random

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from auto_encoders_domainNet import *
from da_auto_encoder import *
os.environ["CUDA_VISIBLE_DEVICES"]="2"

#data generators
class Generator_SML(ImageDataGenerator):
    def __init__(self, featurewise_center=False, samplewise_center=False, featurewise_std_normalization=False, samplewise_std_normalization=False, zca_whitening=False, zca_epsilon=0.000001, rotation_range=0, width_shift_range=0, height_shift_range=0, brightness_range=None, shear_range=0, zoom_range=0, channel_shift_range=0, fill_mode='nearest', cval=0, horizontal_flip=False, vertical_flip=False, rescale=None, preprocessing_function=None, data_format='channels_last', validation_split=0, interpolation_order=1, dtype='float32', class_list = None):
        super().__init__(featurewise_center=featurewise_center, samplewise_center=samplewise_center, featurewise_std_normalization=featurewise_std_normalization, samplewise_std_normalization=samplewise_std_normalization, zca_whitening=zca_whitening, zca_epsilon=zca_epsilon, rotation_range=rotation_range, width_shift_range=width_shift_range, height_shift_range=height_shift_range, brightness_range=brightness_range, shear_range=shear_range, zoom_range=zoom_range, channel_shift_range=channel_shift_range, fill_mode=fill_mode, cval=cval, horizontal_flip=horizontal_flip, vertical_flip=vertical_flip, rescale=rescale, preprocessing_function=preprocessing_function, data_format=data_format, validation_split=validation_split, interpolation_order=interpolation_order, dtype=dtype)
        if class_list is not None:
            print("Class list given")
            self._class_list = class_list
            
            # self.index_generator=self.index_generator_by_classes
            # print("Length of Class List not equal to the batch size")
        
    def index_generator(self):
        self.reset()
        while 1:
            if self.seed is not None:
                np.random.seed(self.seed + self.total_batches_seen)
            if self.batch_index == 0:
                self._set_index_array()

            cls_idx = [random.choice(np.where(self.classes == x))[0] for x in self._class_list] 
            yield cls_idx

def data_generators(data_dir, batch_size = 16, img_size = (224,224), seed = 1):
    datagen = Generator_SML(rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.05) # set validation split

    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        seed = seed,
        subset='training') # set as training data

    test_generator = datagen.flow_from_directory(
        data_dir, # same directory as training data
        target_size=img_size,
        batch_size=batch_size,
        # save_format='jpg',
        seed = seed,
        subset='validation') # set as validation data
    return train_generator, test_generator

def generator_combined(gen_s, gen_t):
    while True:
        X_s_i = gen_s.next()
        X_t_i = gen_t.next()
        if X_s_i[0].shape[0] == X_t_i[0].shape[0]:
            yield [X_s_i[0], X_t_i[0]], [X_s_i[0], X_t_i[0]]
        else:
            continue

src = 'clipart'
tgt = 'real'
s_data_dir = f'/data1/mrahman/DomainNet/{src}'
s_train_generator , s_test_generator = data_generators(s_data_dir, seed=15)

t_data_dir = f'/data1/mrahman/DomainNet/{tgt}'
t_train_generator , t_test_generator = data_generators(t_data_dir, seed=15)

# x = s_train_generator.next()
# x[1].argmax(axis=1)


latent_dim = 100
s_ae = AutoEncoder_vgg16(output_layer_name= f'{src}_output', latent_dim=latent_dim)
t_ae = AutoEncoder_vgg16(output_layer_name= f'{tgt}_output', latent_dim=latent_dim)

stae = S_T_AE(s_ae, t_ae, n_classes=345, beta = 0.004)
stae.t_encoder.summary()
stae.ae_source_target.summary()

epochs = 2
losses = {f'{src}_output': stae.s_ae_loss, f'{tgt}_output': stae.t_ae_loss}
# losses = {f'{src}_output': binary_crossentropy, f'{tgt}_output': binary_crossentropy}
opt = Adam(lr=0.001)
stae.ae_source_target.compile(loss = losses, optimizer=opt)

# plot_model(stae.ae_source_target, to_file='ae_enc_dec.png', show_shapes=True)

hist = stae.ae_source_target.fit_generator(generator_combined(s_train_generator, t_train_generator), 
                                            steps_per_epoch=512,
                                            epochs = epochs,
                                            verbose = 1,
                                            use_multiprocessing=True)

stae.ae_source_target.save_weights(f'model_weights/ae_{src}_to_{tgt}.h5')

stae.ae_source_target.load_weights(f'model_weights/ae_{src}_to_{tgt}.h5')


stae.build_s_enc_cls_model()

# for layer in stae.s_enc_cls_model.layers:
#     layer.trainable = False

# stae.classifier.trainable = True

opt = Adam(lr = 0.001)

stae.s_enc_cls_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
stae.s_enc_cls_model.summary()
clbck = EarlyStopping(patience=5)

stae.s_enc_cls_model.fit_generator(s_train_generator,
                                    epochs = 10,
                                    steps_per_epoch=5012,
                                    verbose=1,
                                    callbacks= [clbck],
                                    validation_data=s_test_generator) 

s_score = stae.s_enc_cls_model.evaluate_generator(s_test_generator, verbose=1)
# s_score = accuracy_score(y_s_test.argmax(1), s_pred.argmax(1))
print(f'Source Score {src} to {tgt}: {s_score[1]}')                    



stae.build_t_enc_cls_model()
stae.t_enc_cls_model.summary()
stae.t_enc_cls_model.compile(loss='categorical_crossentropy', optimizer='adam')

# t_pred = stae.t_enc_cls_model.predict(x_t_train)
# t_score = accuracy_score(y_t_train.argmax(1), t_pred.argmax(1))
# print(f"Target sore, Before Fine tune: {t_score}")

stae.t_enc_cls_model.fit_generator(t_test_generator,
                    epochs=2,
                    steps_per_epoch=512,
                    verbose = 1,
                    validation_data=t_train_generator)

t_pred = stae.t_enc_cls_model.predict_generator(t_train_generator)
t_score = accuracy_score(y_t_train.argmax(1), t_pred.argmax(1))
print(f"Target score {source} to {target}, After Fine tune: {t_score}")