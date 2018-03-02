from __future__ import print_function
from keras.layers import UpSampling2D, Convolution2D, BatchNormalization
import numpy as np

def z_noise(bs):
    """ Random noise of size bs, to serve as generator noise input
    """
    return np.random.rand(bs, 100)


def c_noise(sz):
    """ Generator latent code prior, using a categorical distribution as a prior
    """
    return np.random.multinomial(1, 10*[0.1], size=sz).astype(np.float)

def pre_process(X):
    """ Go from [0,255] range to [-1,1] (to match tanh activation domain)
    """
    X = np.divide(X.astype(np.float), 255.0/2.0)
    return np.subtract(X, 1.0)

def import_mnist(preprocess=True):
    """ Import and pre-process mnist dataset
    """
    print("Downloading MNIST data...", end='')
    from keras.datasets import mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test  = X_test.reshape(X_test.shape[0], 28, 28, 1)
    if(preprocess):
        X_train = pre_process(X_train)
        X_test  = pre_process(X_test)
    print("done.")
    return X_train, y_train, X_test, y_test, X_train.shape[0]

def make_trainable(net, val):
    """ Freeze or unfreeze layers
    """
    net.trainable = val
    for l in net.layers: l.trainable = val

def ups_conv_bn(x, dim, act):
    x = UpSampling2D()(x)
    x = Convolution2D(dim, 3, 3, border_mode='same', activation=act)(x)
    return BatchNormalization(mode=2)(x)
