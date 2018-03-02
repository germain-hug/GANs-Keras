from utils.visualization import plot_results_GAN

class GAN(object):
    """ Generic GAN Class
    """

    def __init__(self):
        self.img_shape = (28, 28, 1)
        self.noise_dim = 100
        self.class_dim = 10
        self.lr = 1e-4

    def load_weights(self,path):
        self.m.load_weights(path)

    def visualize(self):
        plot_results_GAN(self.G)
