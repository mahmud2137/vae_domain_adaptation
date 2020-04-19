from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from keras.layers import Lambda, Reshape, Input, Dense, Conv2D, MaxPool2D, Flatten, UpSampling2D, concatenate
from keras.models import Model, Sequential
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
from sklearn.decomposition import PCA
import tensorflow as tf
import utils
from functools import partial


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
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

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


class VAE():
    def __init__(self, s_input_shape, t_input_shape, n_classes, latent_dim=3):
        self.s_input_shape = s_input_shape
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.s_total_pixel = s_input_shape[0] * s_input_shape[1]

        self.t_input_shape = t_input_shape
        self.latent_dim = latent_dim
        self.t_total_pixel = t_input_shape[0] * t_input_shape[1]

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
        s_outputs = Conv2D(input_shape[2], (3,3), padding = 'same', strides = (1,1), activation = 'linear', name ='s_output')(x)
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
        x = Dense(7*7*32, activation='relu')(x)
        x = Reshape((7,7,32))(x)
        x = Conv2D(32, (2,2), padding = 'same', strides = (1,1), activation = 'relu')(x)
        x = UpSampling2D((2,2))(x)
        x = Conv2D(16, (3,3), padding = 'same', strides = (1,1), activation = 'relu')(x)
        x = UpSampling2D((2,2))(x)
        t_outputs = Conv2D(input_shape[2], (3,3), padding = 'same', strides = (1,1), activation = 'linear', name = 't_output')(x)
        return t_outputs

    def build_classifier(self):

        model = Sequential()
        model.add(Dense(self.latent_dim, activation='relu', input_dim=self.latent_dim))
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


    def s_vae_loss(self, y_true, y_pred):
        xent_loss = self.s_total_pixel * binary_crossentropy(K.flatten(y_true), K.flatten(y_pred))
        kl_loss = - 0.25 * K.sum(1 + self.s_z_log_var - K.square(self.s_z_mean) - K.exp(self.s_z_log_var), axis=-1)
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
        kl_loss = - 0.25 * K.sum(1 + self.t_z_log_var - K.square(self.t_z_mean) - K.exp(self.t_z_log_var), axis=-1)
        mmd_loss = maximum_mean_discrepancy(self.s_z_mean, self.t_z_mean, kernel=gaussian_kernel)
        vae_loss = K.mean(xent_loss + mmd_loss)
        return vae_loss


    # def train_target(self, X_s, X_t, n_samples, epochs = 10, batch_size = 128):

    #     # self.s_encoder.trainable = False
    #     # for l in range(len(self.s_encoder.layers)-1):
    #     #     w = self.s_encoder.layers[l].get_weights()
    #     #     self.t_encoder.layers[l].set_weights(w)
        
    #     n_batches = n_samples//batch_size
    #     for e in range(epochs):
    #         s_l_sum = 0
    #         t_l_sum = 0
    #         for batch in range(n_batches):
    #             idx = np.random.randint(0, X_s.shape[0], batch_size)
    #             batch_s = X_s[idx]

    #             idx = np.random.randint(0, X_t.shape[0], batch_size)
    #             batch_t = X_t[idx]

    #             self.s_v_autoencoder.compile(loss = self.s_vae_loss, optimizer='adam')
    #             s_l = self.s_v_autoencoder.train_on_batch(batch_s, batch_s)

    #             self.t_v_autoencoder.compile(loss= self.t_vae_loss,  optimizer = 'adam')
    #             t_l = self.t_v_autoencoder.train_on_batch(batch_t, batch_t)
    #             s_l_sum += s_l
    #             t_l_sum += t_l

    #             # if batch % 50 == 0:
    #             print(f"batch: {batch}, source loss: {s_l}, target loss: {t_l} \n")

    #         print(f"\tEpoch: {e}, Avg Source Loss: {s_l_sum/n_batches}, Avg Target Loss: {t_l_sum/n_batches}")


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    original_dim = x_train.shape[1] * x_train.shape[2]
    # x_train = np.reshape(x_train, [-1, original_dim])
    # x_test = np.reshape(x_test, [-1, original_dim])
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # network parameters
    input_shape = x_train.shape[1:]
    n_samples = x_train.shape[0]

    # intermediate_dim = 512
    batch_size = 128
    latent_dim = 3
    epochs = 50
    n_classes = len(np.unique(y_train))
    vae = VAE(s_input_shape=input_shape, t_input_shape=input_shape, n_classes= n_classes)
    vae.s_encoder.summary()
    vae.t_encoder.summary()
    # vae.t_v_autoencoder.summary()
    vae.vae_source_target.summary()
    losses = {'s_output': vae.s_vae_loss, 't_output': vae.t_vae_loss}
    vae.vae_source_target.compile(loss = losses, optimizer='adam')
    vae.vae_source_target.fit([x_train, x_train], [x_train, x_train],
                                epochs=epochs,
                                batch_size=batch_size,
                                validation_data=([x_test, x_test], [x_test, x_test]))
    # plot_model(vae.vae_source_target, to_file='vae_enc_dec.png', show_shapes=True)

    # vae.s_v_autoencoder.compile(loss = vae.s_vae_loss, optimizer='adam')
    # # vae.s_v_autoencoder.fit(x_train, x_train,
    # #                         epochs= epochs,
    # #                         batch_size= batch_size,
    # #                         validation_data = (x_test, x_test)
    # #                         )
    # # vae.s_v_autoencoder.save_weights('model_weights/source_vae.h5')

    # vae.s_v_autoencoder.load_weights('model_weights/source_vae.h5')
    # vae.train_target(x_train, x_train, n_samples)


