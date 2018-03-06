import matplotlib.pyplot as plt
from utils.utils import z_noise, c_noise
from keras.utils.np_utils import to_categorical
import numpy as np


def plot_large(img):
    """ Custom sized image
    """
    fig1 = plt.figure(figsize = (4,4))
    ax1 = fig1.add_subplot(1,1,1)
    ax1.axes.get_xaxis().set_visible(False)
    ax1.axes.get_yaxis().set_visible(False)
    ax1.imshow(img, cmap='gray')
    plt.show()


def plot_results_GAN(G,n=5):
    """ Plots n x n windows from DCGAN and WGAN generator
    """
    img = np.zeros((n * 28,1))
    for i in range(n):
        col = np.multiply(np.add(G.predict(z_noise(n)).reshape(n * 28,28), 1.0), 255.0/2.0)
        img = np.concatenate((img,col), axis=1)
    plot_large(img)

def plot_results_CGAN(G):
    """ Plots n x n windows from CGAN generator
    """
    labels = np.arange(0, 10)

    n = len(labels)
    img = np.zeros((n * 28,1))
    for i in range(n):
        # Remap from tanh range [-1, 1] to image range [0, 255]
        col = np.multiply(np.add(G.predict([z_noise(n), \
            to_categorical(labels,n)]).reshape(n * 28,28), 1.0), 255.0/2.0)
        img = np.concatenate((img,col), axis=1)
    plot_large(img)

def plot_results_InfoGAN(G):
    """ Plots 10x10 windows from InfoGAN generator
    """
    upper = 0.0
    lower = 2.0
    latent_code =  np.arange(lower, upper, (upper-lower)/10.0)
    latent_code = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    z = z_noise(10)

    img = np.zeros((10*28,1))
    for i in range(10):
        x = np.zeros((10,10))
        x[i,:] = 1.0#latent_code
        # Convert tanh range [-1; 1] to [0; 255]
        col = np.multiply(np.add(G.predict([z,x]).reshape(10*28,28), 1.0), 255.0/2.0)
        img = np.concatenate((col,img), axis=1)
    plot_large(img)
