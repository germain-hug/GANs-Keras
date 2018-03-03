from __future__ import print_function
from utils.utils import z_noise, make_trainable
from utils.visualization import plot_results_GAN
from keras.models import Model, Sequential
from keras.layers import *
from keras.optimizers import Adam
from models.gan import GAN
from tqdm import tqdm

import numpy as np

class DCGAN(GAN):
    """ Deep Convolutional GAN, as per https://arxiv.org/abs/1511.06434
    """

    def __init__(self, args):
        GAN.__init__(self)
        self.build_model()

    def build_model(self):
        self.G = self.generator()
        self.D = self.discriminator()
        self.m = Sequential([self.G, self.D])
        self.D.compile(Adam(10 * self.lr), "binary_crossentropy")
        self.m.compile(Adam(self.lr), "binary_crossentropy")

    def train(self, X_train, nb_epoch=10, nb_iter=450, bs=128, y_train=None, save_path='../models/'):
        """ Train DCGAN:
            - Train D to discriminate G results
            - Train G to fool D (D is frozen)
        """
        for e in range(nb_epoch):
            print("Epoch " + str(e+1) + "/" + str(nb_epoch))
            for i in tqdm(range(nb_iter)):
                # Get real and fake data + labels
                X,y = self.mixed_data(bs//2, X_train)
                # Train D
                self.D.train_on_batch(X,y)
                # Freeze D
                make_trainable(self.D, False)
                # Train G
                self.m.train_on_batch(z_noise(bs), np.zeros([bs]))
                # Unfreeze D
                make_trainable(self.D, True)
            self.m.save_weights(save_path +'DCGAN_' + str(e+1) + '.h5')

    def pre_train(self, X_train, y_train=None):
        """ Pre-train D for a couple of iterations
        """
        print("Pre-training D for a couple of iterations...", end='')
        sz = X_train.shape[0]//200
        x1 = np.concatenate([np.random.permutation(X_train)[:sz], self.G.predict(z_noise(sz))])
        self.D.fit(x1, [0]*sz + [1]*sz, batch_size=128, nb_epoch=1, verbose=0)
        print("done.")

    def mixed_data(self, sz, X_train):
        """ Generate fake and real data to train D
        """
        N = X_train.shape[0]
        sz = N//200
        real_img = X_train[np.random.randint(0,N,size=sz)]
        X = np.concatenate((real_img, self.G.predict(z_noise(sz))))
        return X, [0]*sz + [1]*sz

    def generator(self):
        """ DCGAN Generator, small neural network with Upsampling and ReLU
        """
        return Sequential([
            Dense(512*7*7, input_dim=self.noise_dim, activation='relu'),
            BatchNormalization(mode=2),
            Reshape((7, 7, 512)),
            UpSampling2D(),
            Convolution2D(64, 3, 3, border_mode='same', activation='relu'),
            BatchNormalization(mode=2),
            UpSampling2D(),
            Convolution2D(32, 3, 3, border_mode='same', activation='relu'),
            BatchNormalization(mode=2),
            Convolution2D(1, 1, 1, border_mode='same', activation='tanh')
        ])

    def discriminator(self):
        """ DCGAN Discriminator, small neural network with upsampling
        """
        return Sequential([
            Convolution2D(256, 5, 5, subsample=(2,2), border_mode='same', input_shape=self.img_shape, activation=LeakyReLU()),
            Convolution2D(512, 5, 5, subsample=(2,2), border_mode='same', activation=LeakyReLU()),
            Flatten(),
            Dense(256, activation=LeakyReLU()),
            Dense(1, activation = 'sigmoid')
        ])
