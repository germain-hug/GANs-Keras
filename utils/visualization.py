import matplotlib.pyplot as plt
import numpy as np

"""
Plots 10x10 windows from InfoGAN generator
"""
def plot_results_InfoGAN(G, c):
    preds = np.zeros((10*28,1))
    # Convert tanh range [-1; 1] to [0; 255]
    for i in range(10):
        x = np.zeros((10,10))
        x[:,2] = c
        p = np.multiply(np.add(G.predict([noise(10),x]).reshape(10*28,28), 1.0), 255.0/2.0)
        preds = np.concatenate((p,preds), axis=1)

    fig1 = plt.figure(figsize = (10,10))
    ax1 = fig1.add_subplot(1,1,1)
    ax1.axes.get_xaxis().set_visible(False)
    ax1.axes.get_yaxis().set_visible(False)
    ax1.imshow(preds, cmap='gray')
