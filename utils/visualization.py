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
    # Latent code
    lower = 0.0
    upper = 10.0
    latent_code =  np.arange(lower, upper, (upper-lower)/10.0)

    # Fixed input
    n_rows = 10
    z = z_noise(n_rows)
    img = np.zeros((n_rows*28,1))

    for i in range(10):
        x = np.zeros((n_rows,10))
        x[:,i] = latent_code
        #print(x)
        #x[:,0] = 1.0
        # x[:,3] = 1.0
        # x[:,2] = 1.0
        # x[:,1] = 1.0
        #x[:,0] = latent_code
        print(x)

        # Convert tanh range [-1; 1] to [0; 255]
        col = np.multiply(np.add(G.predict([z,x]).reshape(n_rows*28,28), 1.0), 255.0/2.0)
        img = np.concatenate((col,img), axis=1)
    plot_large(img)
