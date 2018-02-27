from __future__ import print_function
from utils.utils import z_noise, make_trainable
from utils.visualization import plot_results_CGAN
from keras.models import Model
from keras.layers import *
from keras.optimizers import Adam, RMSprop
from keras.utils.np_utils import to_categorical
from tqdm import tqdm

import numpy as np

class CGAN(object):
    """ Conditional GAN, as per https://arxiv.org/abs/1411.1784
    """

    def __init__(self, args):
        self.build_model()
        self.preprocess = True

    def build_model(self):
        # Input Tensors
        self.input_G = Input(shape=(100,)) # Noise Vector
        self.input_D = Input(shape=(28,28,1)) # Image Tensor
        self.conditioning_label = Input(shape=(10,))  # One-hot encoded label
        # Assemble CGAN Model using the **functional** API
        self.G = self.generator(self.input_G, self.conditioning_label)
        self.D = self.discriminator(self.input_D, self.conditioning_label)
        self.D.compile(RMSprop(0.5e-4), "mean_squared_error")
        self.m = Model([self.input_G, self.conditioning_label], self.D([self.output_G, self.conditioning_label]))
        self.m.compile(RMSprop(0.5e-5), "mean_squared_error")

    def train(self, X_train, nb_epoch=10, nb_iter=250, bs=128, y_train=None, save_path='../models/'):
        """ Train CGAN:
            - Train D to discriminate G results, conditioned on label
            - Train G to fool D, conditioned on label
        """
        for e in range(nb_epoch):
            print("Epoch " + str(e+1) + "/" + str(nb_epoch))
            for i in tqdm(range(nb_iter)):
                # Get real and fake data + labels
                X, y, label_onehot = self.mixed_data(bs//2, X_train, y_train)
                # Train discriminator
                self.D.train_on_batch([X, label_onehot],y)
                # Clip discriminator weights
                for l in self.D.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -0.01, 0.01) for w in weights]
                    l.set_weights(weights)
                # Freeze discriminator
                make_trainable(self.D, False)
                # Train generator i.e. whole model (G + frozen D)
                self.m.train_on_batch([z_noise(bs), label_onehot], np.zeros([bs]))
                # Unfreeze discriminator
                make_trainable(self.D, True)
            self.m.save_weights(save_path +'CGAN_' + str(e) + '.h5')

    def pre_train(self, X_train, y_train):
        """ Pre-train D for a couple of iterations
        """
        print("Pre-training D for a couple of iterations...", end='')
        sz = X_train.shape[0]//200
        # Random labels to condition on
        permutations  = np.random.randint(0,X_train.shape[0],size=sz)[:sz]
        random_labels = to_categorical(y_train[permutations[:sz]])
        random_images = X_train[permutations[:sz]]
        fake_pred = self.G.predict([z_noise(sz), random_labels])
        # Train D for a couple of iterations
        x1_D = np.concatenate([fake_pred, random_images])
        x2_D = np.concatenate([random_labels, random_labels])
        self.D.fit([x1_D, x2_D], [0]*sz + [1]*sz, batch_size=128, nb_epoch=1, verbose=2)
        print("done.")

    def mixed_data(self, sz, X_train, y_train):
        """ Generate fake and real data to train D. Both real and fake data
        are conditioned on a one-hot encoded vector c.
        """
        permutations = np.random.randint(0,X_train.shape[0],size=sz)
        real_images = X_train[permutations[:sz]]
        label_onehot = to_categorical(y_train[permutations[:sz]], 10)
        X = np.concatenate((real_images, self.G.predict([z_noise(sz),label_onehot])))
        label_onehot = np.concatenate((label_onehot, label_onehot))
        return X, [0]*sz + [1]*sz, label_onehot

    def generator(self, input_G, conditioning_label):
        """ CGAN Generator, small neural network with upsampling and LeakyReLU()
        """
        # Feed each input into a maxout unit
        x_noise = Dense(256, input_dim=100)(input_G)
        x_label = Dense(256, input_dim=10)(conditioning_label)

        # Concatenate the units and feed to the shared branch
        x = merge([x_noise, x_label], mode='concat')
        x = Dense(512*7*7, activation='relu')(x)
        x = BatchNormalization(mode=2)(x)
        x = Reshape((7, 7, 512))(x)
        x = UpSampling2D()(x)
        x = Convolution2D(64, 3, 3, border_mode='same', activation='relu')(x)
        x = BatchNormalization(mode=2)(x)
        x = UpSampling2D()(x)
        x = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(x)
        x = BatchNormalization(mode=2)(x)
        self.output_G = Convolution2D(1, 1, 1, border_mode='same', activation='tanh')(x)

        # Assemble the model
        return Model([input_G, conditioning_label], self.output_G)

    def discriminator(self, input_D, conditioning_label):
        """ CGAN Discriminator, small neural network with upsampling
        """
        # Concatenate the units and feed to the shared branch
        x = Convolution2D(128, 5, 5, subsample=(2,2), border_mode='same', input_shape=(28,28,1), activation=LeakyReLU())(input_D)
        x = Convolution2D(256, 5, 5, subsample=(2,2), border_mode='same', activation=LeakyReLU())(x)
        x = Flatten()(x)
        x = merge([x, conditioning_label], mode='concat')
        x = Dense(512, activation=LeakyReLU())(x)
        x = Dense(256, activation=LeakyReLU())(x)
        output_D = Dense(1, activation = None)(x)

        # Assemble the model
        return Model([input_D, conditioning_label], output_D)

    def load_weights(self,path):
        self.m.load_weights(path)

    def visualize(self):
        plot_results_CGAN(self.G)
