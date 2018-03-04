from __future__ import print_function
from utils.utils import z_noise, c_noise, make_trainable, ups_conv_bn
from utils.visualization import plot_results_InfoGAN
from keras.models import Model
from keras.layers import *
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from models.gan import GAN
from tqdm import tqdm

import numpy as np
import tensorflow as tf

class InfoGAN(GAN):
    """ InfoGAN, as per https://arxiv.org/abs/1606.03657
    We base our GAN architecture on a DCGAN model
    """

    def __init__(self, args):
        GAN.__init__(self)
        self.build_model()

    def build_model(self):
        # Input Tensors
        self.input_G = Input(shape=(self.noise_dim,)) # Noise Vector
        self.input_D = Input(shape=self.img_shape) # Image Tensor
        self.conditioning_label = Input(shape=(self.class_dim,))  # One-hot encoded latent code

        # Assemble InfoGAN Model using the functional API
        self.G = self.generator(self.input_G, self.conditioning_label)
        self.D = self.discriminator(self.input_D)
        self.Q = self.auxiliary(self.input_D)
        self.G_and_D = Model([self.input_G, self.conditioning_label], self.D(self.output_G)) # D attached to G
        self.G_and_Q = Model([self.input_G, self.conditioning_label], self.Q(self.output_G)) # Q attached to G

        # Compile models
        self.D.compile(Adam(10 * self.lr), "binary_crossentropy")
        self.G_and_D.compile(Adam(self.lr), "binary_crossentropy")
        self.G_and_Q.compile(Adam(self.lr), self.custom_objective_Q(self.input_G, self.conditioning_label))

    def train(self, X_train, nb_epoch=10, nb_iter=250, bs=128, y_train=None, save_path='../models/'):
        """ Train InfoGAN:
            - Train D to discriminate G results, conditioned on label
            - Train G to fool D, conditioned on label
        """
        for e in range(nb_epoch):
            print("Epoch " + str(e+1) + "/" + str(nb_epoch))
            for i in tqdm(range(nb_iter)):
                # Retrieve discriminator and auxiliary network training data
                X, y, random_z, random_c = self.mixed_data(bs//2, X_train, y_train, self.G)
                # Train discriminator
                self.D.train_on_batch(X,y)
                # Freeze discriminator
                make_trainable(self.D, False)
                make_trainable(self.Q, False)
                # Train generator i.e. whole model (G + frozen D)
                self.G_and_D.train_on_batch([z_noise(bs), c_noise(bs)], np.zeros([bs]))
                # Unfreeze discriminator
                make_trainable(self.D, True)
                make_trainable(self.Q, True)
                # Train Auxiliary Network
                self.G_and_Q.train_on_batch([random_z, random_c], np.zeros([bs//2]))
            self.G_and_Q.save_weights(save_path +'InfoGAN_Q' + str(e+1) + '.h5')
            self.G_and_D.save_weights(save_path +'InfoGAN_D' + str(e+1) + '.h5')

    def pre_train(self, X_train, y_train):
        """ Pre-train D for a couple of iterations
        """
        print("Pre-training D for a couple of iterations...", end='')
        sz = X_train.shape[0]//200
        # Concatenate real and fake images
        real_images = np.random.permutation(X_train)[:sz]
        fake_images = self.G.predict([z_noise(sz), c_noise(sz)])
        x1 = np.concatenate([real_images, fake_images])
        # Train D
        self.D.fit(x1, [0]*sz + [1]*sz, batch_size=128, nb_epoch=1, verbose=2)
        print("done.")

    def mixed_data(self, sz, X_train, y_train, G):
        """ Generate fake and real data to train D and Q
        """
        # Pre-compute random vectors
        permutations = np.random.randint(0,X_train.shape[0],size=sz)
        random_z = z_noise(sz) # Noise input
        random_c = c_noise(sz) # Latent code
        # Sample real images and fake images
        X = np.concatenate((X_train[permutations[:sz]], G.predict([random_z, random_c])))
        return X, [0]*sz + [1]*sz, random_z, random_c

    def generator(self, input_G, conditioning_label):
        """ InfoGAN Generator, small neural network with upsampling and ReLU
        """
        # Feed conditioning input into a Dense unit
        x_noise = Dense(128, activation='relu')(input_G)
        x_label = Dense(128, activation='relu')(conditioning_label)

        # Concatenate the units and feed to the shared branch
        x = merge([x_noise, x_label], mode='concat')
        x = Dense(512*7*7, activation='relu')(x)
        x = BatchNormalization(mode=2)(x)
        x = Reshape((7, 7, 512))(x)
        # 2 x (UpSampling + Conv2D + BatchNorm) blocks
        x = ups_conv_bn(x, 64, 'relu')
        x = ups_conv_bn(x, 32, 'relu')
        self.output_G = Convolution2D(1, 1, 1, border_mode='same', activation='tanh')(x)
        # Assemble the model
        return Model([input_G, conditioning_label], self.output_G)

    def discriminator(self, input_D):
        """ InfoGAN Discriminator, small neural network with upsampling
        (nb: D is unconditional)
        """
        # Create a shared core for D and Q
        x = Convolution2D(256, 5, 5, subsample=(2,2), border_mode='same', input_shape=(28,28,1), activation=LeakyReLU())(input_D)
        self.shared_D_Q = Convolution2D(512, 5, 5, subsample=(2,2), border_mode='same', activation=LeakyReLU())(x)
        x = Flatten()(self.shared_D_Q)
        x = Dense(256, activation=LeakyReLU())(x)
        output_D = Dense(1, activation = 'sigmoid')(x)
        # Assemble the model
        return Model(input_D, output_D)

    def auxiliary(self, input_D):
        """ Auxiliary network Q, to maximize mutual information in latent code
        """
        x = Flatten()(self.shared_D_Q)
        x = Dense(256, activation='relu')(x)
        output_Q = Dense(10, activation = 'softmax')(x) # Nb: softmax to match c prior (categorical)
        # Assemble the model
        return Model(input_D, output_Q)

    def custom_objective_Q(self, z, c):
        """ Define proxy objective function, using nested loss function (unused args)
        """
        def loss(y_true, y_pred):
            # Conditional entropy Q(c'|z)
            cond_entropy = tf.reduce_mean(-tf.reduce_sum(tf.log(self.G_and_Q([z,c]) + 1e-8) * c, 1))
            # Entropy of latent code H(c)
            lat_entropy = tf.reduce_mean(-tf.reduce_sum(tf.log(c + 1e-8) * c, 1))
            # Total Entropy
            total_entropy = tf.cast(cond_entropy, tf.float32) + tf.cast(lat_entropy, tf.float32)
            return total_entropy
        return loss

    def visualize(self):
        plot_results_InfoGAN(self.G, )

    def load_weights(self,path):
        self.G_and_D.load_weights(path)
        self.G_and_Q.load_weights(path.replace('_D', '_Q'))
