import os
import sys
import keras
import argparse
import tensorflow as tf

from utils.utils import import_mnist
from models.dcgan import DCGAN
from models.wgan import WGAN
#, wgan, cgan, infogan as DCGAN, WGAN, CGAN, InfoGAN

def get_session():
    """ Limit session memory usage
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def parse_args(args):
    """ Parse arguments from command line input
    """
    parser     = argparse.ArgumentParser(description='Training and testing scripts for various types of GAN Architectures')
    parser.add_argument('--type', type=str, default='DCGAN',  help='Choose from {DCGAN, WGAN, CGAN, InfoGAN}')
    parser.add_argument('--nb_epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--visualize', type=bool, default=True, help="Results visualization")
    parser.add_argument('--model', type=str, help="Pre-trained weights path")
    parser.add_argument('--gpu', type=int, help='GPU ID')
    parser.add_argument('--train', dest='train', action='store_true')
    parser.add_argument('--no-train', dest='train', action='store_false')
    parser.set_defaults(train=True)
    return parser.parse_args(args)

def main(args=None):
    # Parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # Check if a GPU Id was set
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    keras.backend.tensorflow_backend.set_session(get_session())

    # Load appropriate model:
    if args.type=='DCGAN': # Deep Convolutional GAN
        model = DCGAN(args)
    elif(args.type=='WGAN'): # Wasserstein GAN
        model = WGAN(args)
    elif(args.type=='CGAN'): # Conditional GAN
        model = CGAN(args)
    elif(args.type=='InfoGAN'): # InfoGAN
        model = InfoGAN(args)

    # Load pre-trained weights
    if args.model:
        model.load_weights(args.model)

    # Load MNIST Data
    if args.train:
        # TODO Pretrain on a couple of iterations !!!!!
        X_train, _, _, _, N = import_mnist(preprocess=model.preprocess)
        model.train(X_train, nb_epoch=args.nb_epochs, nb_iter=X_train.shape[0])

    if args.visualize:
        model.visualize()

if __name__ == '__main__':
    main()
