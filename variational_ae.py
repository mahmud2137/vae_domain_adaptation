from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from keras.layers import Lambda, Reshape, Input, Dense, Conv2D, MaxPool2D, Flatten, UpSampling2D, concatenate
from keras.models import Model, Sequential
from keras.datasets import mnist
from keras.optimizers import Adam
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model, to_categorical
from keras import backend as K
from sklearn.decomposition import PCA
import tensorflow as tf
import utils
from functools import partial
import h5py

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    # return z_mean + K.exp(0.5 * z_log_var) * epsilon
    return z_mean + 0*K.exp(0.5 * z_log_var) +  0*epsilon

def maximum_mean_discrepancy(x, y, kernel=utils.gaussian_kernel_matrix):

    """Computes the Maximum Mean Discrepancy (MMD) of two samples: x and y.
    Maximum Mean Discrepancy (MMD) is a distance-measure between the samples of
    the distributions of x and y. Here we use the kernel two sample estimate
    using the empirical mean of the two distributions.
    MMD^2(P, Q) = || \E{\phi(x)} - \E{\phi(y)} ||^2
                = \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) },
    where K = <\phi(x), \phi(y)>,
    is the desired kernel function, in this case a radial basis kernel.
    Args:
        x: a tensor of shape [num_samples, num_features]
        y: a tensor of shape [num_samples, num_features]
        kernel: a function which computes the kernel in MMD. Defaults to the
                GaussianKernelMatrix.
    Returns:
        a scalar denoting the squared maximum mean discrepancy loss.
    """
    with tf.name_scope('MaximumMeanDiscrepancy'):
        # \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) }
        cost = tf.reduce_mean(kernel(x, x))
        cost += tf.reduce_mean(kernel(y, y))
        cost -= 2 * tf.reduce_mean(kernel(x, y))

        # We do not allow the loss to become negative.
        cost = tf.where(cost > 0, cost, 0, name='value')
    return cost

def plot_results(models,
                 data,
                 batch_size=128,
                 model_name="vae_mnist"):
    """Plots labels and MNIST digits as a function of the 2D latent vector

    # Arguments
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(x_test,
                                   batch_size=batch_size)
    pca = PCA(n_components=2)
    z_mean_pc2 = pca.fit_transform(z_mean)
    # print("z_mean shape: ", z_mean.shape, "z_mean_pc2 shape: ", z_mean_pc2.shape)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean_pc2[:, 0], z_mean_pc2[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    plt.show()

def zero_activation(x):
    return x * 0

def sort_by_class_label(x,y):
    sorted_args = np.argsort(y)
    y_sorted = y[sorted_args]
    x_sorted = x[sorted_args]
    return x_sorted, y_sorted

def upsample_by_class(x_s,y_s,x_t,y_t):
    x_s_ups = np.array([])
    y_s_ups = np.array([])
    x_t_ups = np.array([])
    y_t_ups = np.array([])
    for cl in np.unique(y_s):
        s_args = np.argwhere(y_s == cl).flatten()
        t_args = np.argwhere(y_t == cl).flatten()
        if len(s_args) >= len(t_args):
            x_s_cl_ups = x_s[s_args]
            y_s_cl_ups = y_s[s_args]

            y_t_cl_ups = y_t[t_args]
            x_t_cl_ups = x_t[t_args]
            
            n_diff = s_args.shape[0] - t_args.shape[0]
            sampled_args =  np.random.choice(t_args.flatten(), n_diff)
            y_t_cl_ups = np.append(y_t_cl_ups, y_t[sampled_args], axis=0)
            x_t_cl_ups = np.append(x_t_cl_ups, x_t[sampled_args], axis=0)
            

        elif len(s_args) < len(t_args):
            y_s_cl_ups = y_s[s_args]
            x_s_cl_ups = x_s[s_args]

            y_t_cl_ups = y_t[t_args]
            x_t_cl_ups = x_t[t_args]
            
            n_diff = t_args.shape[0] - s_args.shape[0]
            sampled_args =  np.random.choice(s_args.flatten(), n_diff)
            y_s_cl_ups = np.append(y_s_cl_ups, y_s[sampled_args], axis=0)
            x_s_cl_ups = np.append(x_s_cl_ups, x_s[sampled_args], axis=0)

        if x_s_ups.shape[0] == 0:
            x_s_ups = x_s_cl_ups
            y_s_ups = y_s_cl_ups
            x_t_ups = x_t_cl_ups
            y_t_ups = y_t_cl_ups
        else:
            x_s_ups = np.append(x_s_ups, x_s_cl_ups, axis=0)
            y_s_ups = np.append(y_s_ups, y_s_cl_ups, axis=0)
            x_t_ups = np.append(x_t_ups, x_t_cl_ups, axis=0)
            y_t_ups = np.append(y_t_ups, y_t_cl_ups, axis=0)

    return x_s_ups, y_s_ups, x_t_ups, y_t_ups

class VAE():
    def __init__(self, s_input_shape, t_input_shape, n_classes, latent_dim=3):
        self.s_input_shape = s_input_shape
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.s_total_pixel = s_input_shape[0] * s_input_shape[1]

        self.t_input_shape = t_input_shape
        self.latent_dim = latent_dim
        self.t_total_pixel = t_input_shape[0] * t_input_shape[1]
        self.classifier = None

        # self.s_build_model()
        # self.t_build_model()
        self.build_st_model()

    def s_encoder_layers(self,x):
        
        x = Conv2D(16, (3,3), padding = 'same', strides = (1,1), activation = 'relu')(x)
        x = MaxPool2D(pool_size=(2,2))(x)
        x = Conv2D(32, (2,2), padding='same', strides=(1,1), activation='relu')(x)
        x = MaxPool2D(pool_size=(2,2))(x)
        x = Flatten()(x)
        self.s_z_mean = Dense(self.latent_dim, name='s_z_mean')(x)
        self.s_z_log_var = Dense(self.latent_dim, name='s_z_log_var')(x) 
        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        self.s_z = Lambda(sampling, output_shape=(self.latent_dim,), name='s_z')([self.s_z_mean, self.s_z_log_var])
        return self.s_z 

    def s_decoder_layers(self,x):
        # self.s_latent_inputs = Input(shape=(self.latent_dim,), name='s_z_sampling')
        x = Dense(7*7*32, activation='relu')(x)
        x = Reshape((7,7,32))(x)
        x = Conv2D(32, (2,2), padding = 'same', strides = (1,1), activation = 'relu')(x)
        x = UpSampling2D((2,2))(x)
        x = Conv2D(16, (3,3), padding = 'same', strides = (1,1), activation = 'relu')(x)
        x = UpSampling2D((2,2))(x)
        s_outputs = Conv2D(self.s_input_shape[2], (3,3), padding = 'same', strides = (1,1), activation = 'linear', name ='s_output')(x)
        return s_outputs

    
    def t_encoder_layers(self,x):
        
        x = Conv2D(16, (3,3), padding = 'same', strides = (1,1), activation = 'relu')(x)
        x = MaxPool2D(pool_size=(2,2))(x)
        x = Conv2D(32, (2,2), padding='same', strides=(1,1), activation='relu')(x)
        x = MaxPool2D(pool_size=(2,2))(x)
        x = Flatten()(x)
        self.t_z_mean = Dense(self.latent_dim, name='t_z_mean')(x)
        self.t_z_log_var = Dense(self.latent_dim, name='t_z_log_var')(x) 
        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        self.t_z = Lambda(sampling, output_shape=(self.latent_dim,), name='z')([self.t_z_mean, self.t_z_log_var])
        return self.t_z 

    def t_decoder_layers(self,x):
        # self.t_latent_inputs = Input(shape=(self.latent_dim,), name='z_sampling')
        x = Dense(4*4*32, activation='relu')(x)
        x = Reshape((4,4,32))(x)
        x = Conv2D(32, (2,2), padding = 'same', strides = (1,1), activation = 'relu')(x)
        x = UpSampling2D((2,2))(x)
        x = Conv2D(16, (3,3), padding = 'same', strides = (1,1), activation = 'relu')(x)
        x = UpSampling2D((2,2))(x)
        t_outputs = Conv2D(self.t_input_shape[2], (3,3), padding = 'same', strides = (1,1), activation = 'linear', name = 't_output')(x)
        return t_outputs

    def build_classifier(self):

        model = Sequential()
        model.add(Dense(self.latent_dim, activation='relu', input_dim=self.latent_dim))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(self.n_classes, activation='softmax'))

        feats = Input(shape=(self.latent_dim,))
        class_label = model(feats)
        self.classifier = Model(feats, class_label)

    # def s_build_model(self):
    #     s_input_layer = Input(shape=self.s_input_shape, name = 's_input')
    #     latent_space = self.s_encoder_layers(s_input_layer)
    #     s_output_layer = self.s_decoder_layers()
    #     self.s_encoder = Model(s_input_layer, [self.s_z_mean, self.s_z_log_var, self.s_z], name = 's_encoder')
    #     self.s_decoder = Model(self.s_latent_inputs, s_output_layer, name = 's_decoder')
    #     self.s_v_autoencoder = Model(s_input_layer, self.s_decoder(self.s_encoder(s_input_layer)[2]), name='v_autoencoder')

    # def t_build_model(self):
    #     t_input_layer = Input(shape=self.t_input_shape, name = 't_input')
    #     latent_space = self.t_encoder_layers(t_input_layer)
    #     t_output_layer = self.t_decoder_layers()
    #     self.t_encoder = Model(t_input_layer, [self.t_z_mean, self.t_z_log_var, self.t_z], name = 't_encoder')
    #     self.t_decoder = Model(self.t_latent_inputs, t_output_layer, name = 't_decoder')
    #     self.t_v_autoencoder = Model(t_input_layer, self.t_decoder(self.t_encoder(t_input_layer)[2]), name='v_autoencoder')

    def build_st_model(self):
        #building source vae
        s_input_layer = Input(shape=self.s_input_shape, name = 's_input')
        s_latent_space = self.s_encoder_layers(s_input_layer)
        s_output_layer = self.s_decoder_layers(s_latent_space)

        self.s_encoder = Model(s_input_layer, [self.s_z_mean, self.s_z_log_var, self.s_z], name = 's_encoder')
        # self.s_decoder = Model(s_latent_space, s_output_layer, name = 's_decoder')
        # self.s_v_autoencoder = Model(s_input_layer, self.s_decoder(self.s_encoder(s_input_layer)[2]), name='v_autoencoder')
        self.dummy_layer = Dense(self.latent_dim, activation=zero_activation)(s_latent_space)
        
        #building target vae
        t_input_layer = Input(shape=self.t_input_shape, name = 't_input')
        t_latent_space = self.t_encoder_layers(t_input_layer)
        
        self.t_encoder = Model(t_input_layer, [self.t_z_mean, self.t_z_log_var, self.t_z], name = 't_encoder')
        conc = concatenate([self.dummy_layer, t_latent_space])
        t_output_layer = self.t_decoder_layers(conc)

        # self.t_decoder = Model(conc, t_output_layer, name = 't_decoder')
        # self.t_v_autoencoder = Model(t_input_layer, self.t_decoder(self.t_encoder(t_input_layer)[2]), name='v_autoencoder')
        self.vae_source_target = Model([s_input_layer, t_input_layer], [s_output_layer, t_output_layer])

    def build_s_enc_cls_model(self):
        # Building The model with Source Encoder and Classifier
        if self.classifier is None:
            self.build_classifier()
        self.s_enc_cls_model = Model(input=self.s_encoder.input, output=self.classifier(self.s_encoder.output[2]))
        

    def build_t_enc_cls_model(self):
        # Building The model with Target Encoder and Classifier
        # Assuming The classifier is already built and trained by the source data
        self.t_enc_cls_model = Model(input=self.t_encoder.input, output=self.classifier(self.t_encoder.output[2]))
        # return t_enc_cls_model

    def s_vae_loss(self, y_true, y_pred):
        xent_loss = self.s_total_pixel * binary_crossentropy(K.flatten(y_true), K.flatten(y_pred))
        kl_loss = - 0 * K.sum(1 + self.s_z_log_var - K.square(self.s_z_mean) - K.exp(self.s_z_log_var), axis=-1)
        vae_loss = K.mean(xent_loss + kl_loss)
        return vae_loss

    def t_vae_loss(self, y_true, y_pred):
        sigmas = [
            1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
            1e3, 1e4, 1e5, 1e6
        ]
        gaussian_kernel = partial(
        utils.gaussian_kernel_matrix, sigmas=tf.constant(sigmas))

        xent_loss = self.t_total_pixel * binary_crossentropy(K.flatten(y_true), K.flatten(y_pred))
        # kl_loss = - 0.25 * K.sum(1 + self.t_z_log_var - K.square(self.t_z_mean) - K.exp(self.t_z_log_var), axis=-1)
        mmd_loss = maximum_mean_discrepancy(self.s_z_mean, self.t_z_mean, kernel=gaussian_kernel)
        vae_loss = K.mean(xent_loss +  mmd_loss)
        return vae_loss




if __name__ == '__main__':

    #Loading Mnist data
    (x_s_train, y_s_train), (x_s_test, y_s_test) = mnist.load_data()

    original_dim = x_s_train.shape[1] * x_s_train.shape[2]
    # x_s_train = np.reshape(x_s_train, [-1, original_dim])
    # x_s_test = np.reshape(x_s_test, [-1, original_dim])
    x_s_train = np.expand_dims(x_s_train, axis=-1)
    x_s_test = np.expand_dims(x_s_test, axis=-1)
    x_s_train = x_s_train.astype('float32') / 255
    x_s_test = x_s_test.astype('float32') / 255

    #Loading USPS dataset
    
    with h5py.File('usps.h5', 'r') as hf:
            train = hf.get('train')
            x_t_train = train.get('data')[:].reshape(-1,16,16,1)
            y_t_train = train.get('target')[:]
            test = hf.get('test')
            x_t_test = test.get('data')[:].reshape(-1,16,16,1)
            y_t_test = test.get('target')[:]

    x_s_train , y_s_train = sort_by_class_label(x_s_train, y_s_train)
    x_t_train , y_t_train = sort_by_class_label(x_t_train, y_t_train)

    x_s_test , y_s_test = sort_by_class_label(x_s_test, y_s_test)
    x_t_test , y_t_test = sort_by_class_label(x_t_test, y_t_test)

    x_s_train, y_s_train, x_t_train, y_t_train = upsample_by_class(x_s_train, y_s_train, x_t_train, y_t_train)
    x_s_test, y_s_test, x_t_test, y_t_test = upsample_by_class(x_s_test, y_s_test, x_t_test, y_t_test)       

    y_s_train = to_categorical(y_s_train)
    y_s_test = to_categorical(y_s_test)
    y_t_train = to_categorical(y_t_train)
    y_t_test = to_categorical(y_t_test)

    # network parameters
    s_input_shape = x_s_train.shape[1:]
    t_input_shape = x_t_train.shape[1:]
    n_samples = x_s_train.shape[0]

    # intermediate_dim = 512
    batch_size = 128
    latent_dim = 10
    epochs = 400
    n_classes = 10

    vae = VAE(s_input_shape=s_input_shape, t_input_shape=t_input_shape, n_classes= n_classes, latent_dim=latent_dim)
    vae.s_encoder.summary()
    vae.t_encoder.summary()
    # vae.t_v_autoencoder.summary()
    vae.vae_source_target.summary()
    losses = {'s_output': vae.s_vae_loss, 't_output': vae.t_vae_loss}
    opt = Adam(lr=0.00001)
    vae.vae_source_target.compile(loss = losses, optimizer=opt)
    
    hist = vae.vae_source_target.fit([x_s_train, x_t_train], [x_s_train, x_t_train],
                                epochs=epochs,
                                verbose=2,
                                batch_size=batch_size,
                                validation_data=([x_s_test, x_t_test], [x_s_test, x_t_test]))
    # plot_model(vae.vae_source_target, to_file='vae_enc_dec.png', show_shapes=True)

    vae.vae_source_target.save_weights('model_weights/vae.h5')



    vae.vae_source_target.load_weights('model_weights/vae.h5')

    vae.build_s_enc_cls_model()
    for layer in vae.s_enc_cls_model.layers:
        layer.trainable = False

    vae.classifier.trainable = True
    vae.s_enc_cls_model.compile(loss='categorical_crossentropy', optimizer='adam')
    vae.s_enc_cls_model.summary()
    vae.s_enc_cls_model.fit(x_s_train, y_s_train,
                        epochs = 100,
                        verbose=2,
                        batch_size=64,
                        validation_data=(x_s_test, y_s_test))

    s_score = vae.s_enc_cls_model.evaluate(x_s_test, y_s_test)
    print(f'Source Score: {s_score}')

    vae.build_t_enc_cls_model()
    vae.t_enc_cls_model.summary()
    vae.t_enc_cls_model.compile(loss='categorical_crossentropy', optimizer='adam')
    vae.t_enc_cls_model.fit(x_t_test, y_t_test,
                        epochs=5,
                        verbose = 2,
                        batch_size=64,
                        validation_data=(x_t_train, y_t_train))

    t_score = vae.t_enc_cls_model.evaluate(x_t_train, y_t_train)
    print(f"sore: {t_score}")


