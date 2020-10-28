from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from keras.layers import Lambda, Dropout, ZeroPadding2D, ZeroPadding3D, Reshape, Input, Dense, Conv2D, MaxPool2D, Flatten, UpSampling2D, concatenate, Conv3D, MaxPool3D, UpSampling3D, BatchNormalization
# from keras.layers.core import Dropout
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
        outputs = Conv2D(self.input_shape[2], (3,3), padding = 'same', strides = (1,1), activation = 'sigmoid', name ='M_output')(x)
        return outputs

class AutoEncoder_USPS():
    def __init__(self, input_shape, latent_dim=100):
        self.input_shape = input_shape
        self.latent_dim = latent_dim

    def encoder_layers(self,x):

        x = Conv2D(16, (3,3), padding = 'same', strides = (1,1), activation = 'relu')(x)
        x = MaxPool2D(pool_size=(2,2))(x)
        x = Conv2D(32, (2,2), padding='same', strides=(1,1), activation='relu')(x)
        x = MaxPool2D(pool_size=(2,2))(x)
        x = Flatten()(x)

        self.encoded  = Dense(self.latent_dim, activation='relu')(x)
        return self.encoded

    def decoder_layers(self,x):

        x = Dense(4*4*32, activation='relu')(x)
        x = Reshape((4,4,32))(x)
        x = Conv2D(32, (2,2), padding = 'same', strides = (1,1), activation = 'relu')(x)
        x = UpSampling2D((2,2))(x)
        x = Conv2D(16, (3,3), padding = 'same', strides = (1,1), activation = 'relu')(x)
        x = UpSampling2D((2,2))(x)
        outputs = Conv2D(self.input_shape[2], (3,3), padding = 'same', strides = (1,1), activation = 'linear', name = 'U_output')(x)
        return outputs


class AutoEncoder_SVHN():
    def __init__(self, input_shape, latent_dim=100):
        self.input_shape = input_shape
        self.latent_dim = latent_dim

    def encoder_layers(self,x):

        x = Conv2D(16, (3,3), padding = 'same', strides = (1,1), activation = 'relu')(x)
        x = MaxPool2D(pool_size=(2,2))(x)
        x = Conv2D(32, (2,2), padding='same', strides=(1,1), activation='relu')(x)
        x = MaxPool2D(pool_size=(2,2))(x)
        x = Conv2D(32, (2,2), padding='same', strides=(1,1), activation='relu')(x)
        x = MaxPool2D(pool_size=(2,2))(x)
        x = Flatten()(x)
        self.encoded  = Dense(self.latent_dim, activation='relu')(x)
        return self.encoded

    def decoder_layers(self,x):

        x = Dense(4*4*32, activation='relu')(x)
        x = Reshape((4,4,32))(x)
        x = Conv2D(32, (2,2), padding = 'same', strides = (1,1), activation = 'relu')(x)
        x = UpSampling2D((2,2))(x)
        x = Conv2D(32, (2,2), padding = 'same', strides = (1,1), activation = 'relu')(x)
        x = UpSampling2D((2,2))(x)
        x = Conv2D(16, (3,3), padding = 'same', strides = (1,1), activation = 'relu')(x)
        x = UpSampling2D((2,2))(x)
        outputs = Conv2D(self.input_shape[2], (3,3), padding = 'same', strides = (1,1), activation = 'linear', name = 'S_output')(x)
        return outputs

class AutoEncoder_SYN():
    def __init__(self, input_shape, latent_dim=100):
        self.input_shape = input_shape
        self.latent_dim = latent_dim

    def encoder_layers(self,x):

        x = Conv2D(16, (3,3), padding = 'same', strides = (1,1), activation = 'relu')(x)
        x = MaxPool2D(pool_size=(2,2))(x)
        x = Conv2D(32, (2,2), padding='same', strides=(1,1), activation='relu')(x)
        x = MaxPool2D(pool_size=(2,2))(x)
        x = Conv2D(32, (2,2), padding='same', strides=(1,1), activation='relu')(x)
        x = MaxPool2D(pool_size=(2,2))(x)
        x = Flatten()(x)
        self.encoded  = Dense(self.latent_dim, activation='relu')(x)
        return self.encoded

    def decoder_layers(self,x):

        x = Dense(4*4*32, activation='relu')(x)
        x = Reshape((4,4,32))(x)
        x = Conv2D(32, (2,2), padding = 'same', strides = (1,1), activation = 'relu')(x)
        x = UpSampling2D((2,2))(x)
        x = Conv2D(32, (2,2), padding = 'same', strides = (1,1), activation = 'relu')(x)
        x = UpSampling2D((2,2))(x)
        x = Conv2D(16, (3,3), padding = 'same', strides = (1,1), activation = 'relu')(x)
        x = UpSampling2D((2,2))(x)
        outputs = Conv2D(self.input_shape[2], (3,3), padding = 'same', strides = (1,1), activation = 'linear', name = 'Sy_output')(x)
        return outputs


class AutoEncoder_Office():
    def __init__(self, input_shape, output_layer_name, latent_dim=100):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.output_layer_name = output_layer_name

    def encoder_layers(self,x):

        x = Conv2D(32, (3,3), padding = 'same', strides = (1,1), activation = 'relu')(x)
        x = Conv2D(32, (3,3), padding = 'same', strides = (1,1), activation = 'relu')(x)
        x = MaxPool2D(pool_size=(2,2))(x)
        x = Conv2D(64, (2,2), padding='same', strides=(1,1), activation='relu')(x)
        x = Conv2D(64, (2,2), padding='same', strides=(1,1), activation='relu')(x)
        x = MaxPool2D(pool_size=(2,2))(x)
        x = Conv2D(128, (2,2), padding='same', strides=(1,1), activation='relu')(x)
        x = Conv2D(128, (2,2), padding='same', strides=(1,1), activation='relu')(x)
        x = Conv2D(128, (2,2), padding='same', strides=(1,1), activation='relu')(x)
        x = MaxPool2D(pool_size=(2,2))(x)
        x = Conv2D(256, (2,2), padding='same', strides=(1,1), activation='relu')(x)
        x = Conv2D(256, (2,2), padding='same', strides=(1,1), activation='relu')(x)
        x = Conv2D(256, (2,2), padding='same', strides=(1,1), activation='relu')(x)
        x = MaxPool2D(pool_size=(2,2))(x)
        x = Flatten()(x)
        self.encoded  = Dense(self.latent_dim, activation='relu')(x)
        return self.encoded

    def decoder_layers(self,x):

        x = Dense(6*6*256, activation='relu')(x)
        x = Reshape((6,6,256))(x)
        x = Conv2D(256, (2,2), padding = 'same', strides = (1,1), activation = 'relu')(x)
        x = Conv2D(256, (2,2), padding = 'same', strides = (1,1), activation = 'relu')(x)
        x = Conv2D(256, (2,2), padding = 'same', strides = (1,1), activation = 'relu')(x)
        x = UpSampling2D((2,2))(x)
        x = Conv2D(128, (2,2), padding = 'same', strides = (1,1), activation = 'relu')(x)
        x = Conv2D(128, (2,2), padding = 'same', strides = (1,1), activation = 'relu')(x)
        x = Conv2D(128, (2,2), padding = 'same', strides = (1,1), activation = 'relu')(x)
        x = UpSampling2D((2,2))(x)
        x = Conv2D(64, (2,2), padding = 'same', strides = (1,1), activation = 'relu')(x)
        x = ZeroPadding2D(padding=((0,1),(0,1)))(x)
        x = UpSampling2D((2,2))(x)
        x = Conv2D(64, (3,3), padding = 'same', strides = (1,1), activation = 'relu')(x)
        x = UpSampling2D((2,2))(x)
        x = Conv2D(32, (3,3), padding = 'same', strides = (1,1), activation = 'relu')(x)
        x = Conv2D(32, (3,3), padding = 'same', strides = (1,1), activation = 'relu')(x)
        outputs = Conv2D(self.input_shape[2], (3,3), padding = 'same', strides = (1,1), activation = 'linear', name = self.output_layer_name)(x)
        return outputs


class AutoEncoder_Radar():
    def __init__(self, input_shape, output_layer_name, latent_dim=100):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.output_layer_name = output_layer_name

    def encoder_layers(self,x):

        x = Conv3D(16, (3,3,3), padding = 'same' , strides = 1, activation = 'relu')(x)
        x = Conv3D(16, (3,3,3), padding = 'same' , strides = 1, activation = 'relu')(x)
        x = BatchNormalization()(x)
        x = MaxPool3D(pool_size = (2,2,2))(x)
        # x = Dropout(0.4)(x)

        x = Conv3D(32, (3,3,3), padding = 'same' , strides = 1, activation = 'relu')(x)
        x = Conv3D(32, (3,3,3), padding = 'same' , strides = 1, activation = 'relu')(x)
        x = BatchNormalization()(x)
        x = MaxPool3D(pool_size = (2,2,1))(x)
        # x = Dropout(0.4)(x)

        x = Conv3D(32, (3,3,3), padding = 'same' , strides = 1, activation = 'relu')(x)
        x = Conv3D(32, (3,3,3), padding = 'same' , strides = 1, activation = 'relu')(x)
        x = BatchNormalization()(x)
        x = MaxPool3D(pool_size = (2,2,1))(x)
        # x = Dropout(0.4)(x)

        x = Flatten()(x)
        self.encoded  = Dense(self.latent_dim, activation='relu')(x)
        return self.encoded

    def decoder_layers(self,x):

        x = Dense(10*7*9*32, activation='relu')(x)
        x = Reshape((10,7,9,32))(x)
        x = UpSampling3D((2,2,1))(x)
        # x = ZeroPadding3D(padding=((0,0),(0,1),(0,0)))(x)
        x = Conv3D(32, (3,3,3), padding = 'same' , strides = 1, activation = 'relu')(x)
        x = Conv3D(32, (3,3,3), padding = 'same' , strides = 1, activation = 'relu')(x)
        x = UpSampling3D((2,2,1))(x)
        # x = ZeroPadding3D(padding=((0,1),(0,1),(0,0)))(x)

        x = Conv3D(32, (3,3,3), padding = 'same' , strides = 1, activation = 'relu')(x)
        x = Conv3D(32, (3,3,3), padding = 'same' , strides = 1, activation = 'relu')(x)
        x = UpSampling3D((2,2,2))(x)

        x = Conv3D(16, (3,3,3), padding = 'same' , strides = 1, activation = 'relu')(x)
        x = ZeroPadding3D(padding=((0,1),(0,1),(0,1)))(x)
        outputs = Conv3D(1, (3,3,3), padding = 'same', strides = (1,1,1), activation = 'linear', name = self.output_layer_name)(x)
        return outputs

class AutoEncoder_Lidar():
    def __init__(self, input_shape, output_layer_name, latent_dim=100):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.output_layer_name = output_layer_name

    def encoder_layers(self,x):

        x = Conv2D(16, (3,3), padding = 'same' , strides = 1, activation = 'relu')(x)
        x = Conv2D(16, (3,3), padding = 'same' , strides = 1, activation = 'relu')(x)
        x = BatchNormalization()(x)
        x = MaxPool2D(pool_size = (2,2))(x)
        # x = Dropout(0.4)(x)

        x = Conv2D(32, (3,3), padding = 'same' , strides = 1, activation = 'relu')(x)
        x = Conv2D(32, (3,3), padding = 'same' , strides = 1, activation = 'relu')(x)
        x = BatchNormalization()(x)
        x = MaxPool2D(pool_size = (2,2))(x)
        # x = Dropout(0.4)(x)

        x = Conv2D(32, (3,3), padding = 'same' , strides = 1, activation = 'relu')(x)
        x = Conv2D(32, (3,3), padding = 'same' , strides = 1, activation = 'relu')(x)
        x = BatchNormalization()(x)
        x = MaxPool2D(pool_size = (1,2))(x)
        # x = Dropout(0.4)(x)

        x = Flatten()(x)
        self.encoded  = Dense(self.latent_dim, activation='relu')(x)
        return self.encoded

    def decoder_layers(self,x):

        x = Dense(15*20*32, activation='relu')(x)
        x = Reshape((15,20,32))(x)
        x = UpSampling2D((1,2))(x)
        x = Conv2D(32, (3,3), padding = 'same' , strides = 1, activation = 'relu')(x)
        # x = BatchNormalization()(x)
        x = Conv2D(32, (3,3), padding = 'same' , strides = 1, activation = 'relu')(x)
        # x = ZeroPadding2D(padding=((0,1),(1,1)))(x)
        x = UpSampling2D((2,2))(x)

        x = Conv2D(32, (3,3), padding = 'same' , strides = 1, activation = 'relu')(x)
        x = Conv2D(32, (3,3), padding = 'same' , strides = 1, activation = 'relu')(x)
        x = UpSampling2D((2,2))(x)
        x = Conv2D(16, (3,3), padding = 'same' , strides = 1, activation = 'relu')(x)
        outputs = Conv2D(1, (3,3), padding = 'same', strides = (1,1), activation = 'linear', name = self.output_layer_name)(x)
        return outputs