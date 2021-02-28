from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from keras.layers import Lambda, MaxPooling2D, Dropout, ZeroPadding2D, ZeroPadding3D, Reshape, Input, Dense, Conv2D, MaxPool2D, Flatten, UpSampling2D, concatenate, Conv3D, MaxPool3D, UpSampling3D, BatchNormalization
# from keras.layers.core import Dropout
from keras.models import Model, Sequential

from keras.applications import VGG16

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

class AutoEncoder_vgg16():
    def __init__(self, output_layer_name ,input_shape = (224,224,3), latent_dim=100):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        # self.layer_no = layer_no
        self.output_layer_name = output_layer_name

    def encoder_layers(self, x):
        #################################
        # Encoder
        #################################
        # inputs = Input(self.input_shape, name = 'input')

        # conv1 = Conv2D(64, (3, 3), activation = 'relu', padding = 'same', name ='conv1_1')(inputs)
        # conv1 = Conv2D(64, (3, 3), activation = 'relu', padding = 'same', name ='conv1_2')(conv1)
        # pool1 = MaxPooling2D(pool_size = (2,2), strides = (2,2), name = 'pool_1')(conv1)

        # conv2 = Conv2D(128, (3, 3), activation = 'relu', padding = 'same', name ='conv2_1')(pool1)
        # conv2 = Conv2D(128, (3, 3), activation = 'relu', padding = 'same', name ='conv2_2')(conv2)
        # pool2 = MaxPooling2D(pool_size = (2,2), strides = (2,2), name = 'pool_2')(conv2)
        
        # conv3 = Conv2D(256, (3, 3), activation = 'relu', padding = 'same', name ='conv3_1')(pool2)
        # conv3 = Conv2D(256, (3, 3), activation = 'relu', padding = 'same', name ='conv3_2')(conv3)
        # conv3 = Conv2D(256, (3, 3), activation = 'relu', padding = 'same', name ='conv3_3')(conv3)
        # pool3 = MaxPooling2D(pool_size = (2,2), strides = (2,2), name = 'pool_3')(conv3)
        
        # conv4 = Conv2D(512, (3, 3), activation = 'relu', padding = 'same', name ='conv4_1')(pool3)
        # conv4 = Conv2D(512, (3, 3), activation = 'relu', padding = 'same', name ='conv4_2')(conv4)
        # conv4 = Conv2D(512, (3, 3), activation = 'relu', padding = 'same', name ='conv4_3')(conv4)
        # pool4 = MaxPooling2D(pool_size = (2,2), strides = (2,2), name = 'pool_4')(conv4)

        # conv5 = Conv2D(512, (3, 3), activation = 'relu', padding = 'same', name ='conv5_1')(pool4)
        # conv5 = Conv2D(512, (3, 3), activation = 'relu', padding = 'same', name ='conv5_2')(conv5)
        # conv5 = Conv2D(512, (3, 3), activation = 'relu', padding = 'same', name ='conv5_3')(conv5)
        # pool5 = MaxPooling2D(pool_size = (2,2), strides = (2,2), name = 'pool_5')(conv5)
        # # return pool5
        # model = Model(inputs = inputs, outputs = pool5, name = 'vgg16_encoder')
        vgg_model = VGG16(include_top = False ,input_shape= self.input_shape)
        # for layer in vgg_model.layers:
        #     layer.name = layer.name + str(self.layer_no)

        f = Flatten()(vgg_model(x))
        self.encoded = Dense(self.latent_dim, activation='relu')(f)
        # model = Model( inputs=vgg_model.input, outputs = self.encoded)
        return self.encoded



    def decoder_layers(self,x):
    
        x = Dense(7*7*512, activation='relu')(x)
        x = Reshape((7,7,512))(x)
        x = UpSampling2D((2,2))(x)

        x = Conv2D(512, (3,3), padding = 'same', strides = 1, activation = 'relu')(x)
        x = UpSampling2D((2,2))(x)

        x = Conv2D(512, (3,3), padding = 'same', strides = 1, activation = 'relu')(x)
        x = UpSampling2D((2,2))(x)

        x = Conv2D(256, (3,3), padding = 'same', strides = 1, activation = 'relu')(x)
        x = UpSampling2D((2,2))(x)

        x = Conv2D(256, (3,3), padding = 'same', strides = 1, activation = 'relu')(x)
        # x = UpSampling2D((2,2))(x)

        x = Conv2D(128, (3,3), padding = 'same', strides = 1, activation = 'relu')(x)
        x = UpSampling2D((2,2))(x)

        x = Conv2D(64, (3,3), padding = 'same', strides = 1, activation = 'relu')(x)
        # x = UpSampling2D((2,2))(x)

        outputs = Conv2D(self.input_shape[2], (3,3), padding = 'same', strides = (1,1), activation = 'sigmoid', name = self.output_layer_name)(x)
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

if __name__ == '__main__':

    ae_vgg16 = AutoEncoder_vgg16(output_layer_name='a')
    enc_model = ae_vgg16.encoder_layers()
    enc_model.summary()
