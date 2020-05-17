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
import tensorflow as tf
import utils
from functools import partial
import h5py

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

class AutoEncoder_Mnist():
    def __init__(self, input_shape, latent_dim=100):
        self.input_shape = input_shape
        self.latent_dim = latent_dim

    def encoder_layers(self,x):

        x = Conv2D(16, (3,3), padding = 'same', strides = (1,1), activation = 'relu')(x)
        x = MaxPool2D(pool_size=(2,2))(x)
        x = Conv2D(32, (2,2), padding='same', strides=(1,1), activation='relu')(x)
        x = MaxPool2D(pool_size=(2,2))(x)
        x = Flatten()(x)
        self.encoded = Dense(self.latent_dim, activation='relu')(x)
        return self.encoded

    def decoder_layers(self,x):
    
        x = Dense(7*7*32, activation='relu')(x)
        x = Reshape((7,7,32))(x)
        x = Conv2D(32, (2,2), padding = 'same', strides = (1,1), activation = 'relu')(x)
        x = UpSampling2D((2,2))(x)
        x = Conv2D(16, (3,3), padding = 'same', strides = (1,1), activation = 'relu')(x)
        x = UpSampling2D((2,2))(x)
        outputs = Conv2D(self.s_input_shape[2], (3,3), padding = 'same', strides = (1,1), activation = 'sigmoid', name ='s_output')(x)
        return outputs

class AutoEncoder_USPS():
    def __init__(self, input_shape, latent_dim=100):
        self.input_shape = input_shape
        self.latent_dim

    def encoder_layers(self,x):

        x = Conv2D(16, (3,3), padding = 'same', strides = (1,1), activation = 'relu')(x)
        x = MaxPool2D(pool_size=(2,2))(x)
        x = Conv2D(32, (2,2), padding='same', strides=(1,1), activation='relu')(x)
        x = MaxPool2D(pool_size=(2,2))(x)
        x = Flatten()(x)
        # self.t_z_mean = Dense(self.latent_dim, name='t_z_mean')(x)
        # self.t_z_log_var = Dense(self.latent_dim, name='t_z_log_var')(x) 
        # # use reparameterization trick to push the sampling out as input
        # # note that "output_shape" isn't necessary with the TensorFlow backend
        # self.t_z = Lambda(sampling, output_shape=(self.latent_dim,), name='z')([self.t_z_mean, self.t_z_log_var])
        self.encoded  = Dense(self.latent_dim, activation='relu')(x)
        return self.encoded

    def decoder_layers(self,x):

        x = Dense(4*4*32, activation='relu')(x)
        x = Reshape((4,4,32))(x)
        x = Conv2D(32, (2,2), padding = 'same', strides = (1,1), activation = 'relu')(x)
        x = UpSampling2D((2,2))(x)
        x = Conv2D(16, (3,3), padding = 'same', strides = (1,1), activation = 'relu')(x)
        x = UpSampling2D((2,2))(x)
        outputs = Conv2D(self.t_input_shape[2], (3,3), padding = 'same', strides = (1,1), activation = 'linear', name = 't_output')(x)
        return outputs


