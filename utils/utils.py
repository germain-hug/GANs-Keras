import numpy as np
import keras.backend as K

from PIL import Image
from tqdm import tqdm
from keras import initializations
from keras.models import Model, Sequential
from keras.layers import *
from keras.optimizers import Adam, RMSprop
from keras.regularizers import l2

def plot_multi(im, dim=(4,4), figsize=(6,6), **kwargs ):
    plt.figure(figsize=figsize)
    for i,img in enumerate(im):
        plt.subplot(*((dim)+(i+1,)))
        plt.imshow(img, **kwargs)
        plt.axis('off')
    plt.tight_layout()

"""
Random noise of size bs, to serve as generator noise input
"""
def noise(bs):
    return np.random.rand(bs, 100)

"""
Generator latent code prior, using a categorical distribution as a prior
"""
def c_prior(sz):
    return np.random.multinomial(1, 10*[0.1], size=sz).astype(np.float)


"""
Go from [0,255] range to [-1,1] (to match tanh activation domain)
"""
def pre_process(X):
    X_train = np.divide(X_train.astype(np.float), 255.0/2.0)
    return np.subtract(X_train, 1.0)

"""
Import and pre-process mnist dataset
"""
def import_mnist(preprocess=True):
    from keras.datasets import mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test  = X_test.reshape(X_test.shape[0], 28, 28, 1)
    if(preprocess):
        X_train = pre_process(X_train)
        X_test  = pre_process(X_test)
    return X_train, y_train, X_test, y_test, X_train_w.shape[0]

"""
Freeze or unfreeze layers
"""
def make_trainable(net, val):
    net.trainable = val
    for l in net.layers: l.trainable = val
