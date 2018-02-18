import os
import sys
import keras
import argparse
import tensorflow as tf

from utils import *
from models import *

"""
Limit memory usage
"""
def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

"""
Parse arguments from command line input
"""
def parse_args(args):
    parser     = argparse.ArgumentParser(description='Training and testing scripts for various types of GAN Architectures')
    parser.add_argument('--type', type=str, default='DCGAN',  help='Choose from {DCGAN, WGAN, CGAN, InfoGAN}')
    parser.add_argument('--nb_epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--visualization', type=bool, default=True, help="Results visualization")
    parser.add_argument('--train', type=bool, default=True, help="Training or testing")
    parser.add_argument('--model', type=str, default=True, help="Pre-trained weights path")
    parser.add_argument('--gpu', type=int, help='GPU ID')
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

    # Load MNIST Data
    if arg.train:
        X_train, _, _, _ = import_mnist(preprocess=False)
        model.train(X_train, nb_epoch=args.nb_epochs)
    else:
        pass # TODO Load weights

if __name__ == '__main__':
    main()
