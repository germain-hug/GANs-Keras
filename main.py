import os
import sys
import keras
import argparse
import tensorflow as tf

from utils.utils import import_mnist
from models.dcgan import DCGAN
from models.wgan import WGAN
from models.cgan import CGAN

def get_session():
    """ Limit session memory usage
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

def parse_args(args):
    """ Parse arguments from command line input
    """
    parser = argparse.ArgumentParser(description='Training and testing scripts for various types of GAN Architectures')
    parser.add_argument('--type', type=str, default='DCGAN',  help='Choose from {DCGAN, WGAN, CGAN, InfoGAN}')
    parser.add_argument('--nb_epochs', type=int, default=100, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size")
    parser.add_argument('--visualize', type=bool, default=True, help="Results visualization")
    parser.add_argument('--model', type=str, help="Pre-trained weights path")
    parser.add_argument('--save_path', type=str, default='weights/',help="Pre-trained weights path")
    parser.add_argument('--gpu', type=int, help='GPU ID')
    parser.add_argument('--train', dest='train', action='store_true', help="Retrain model (default)")
    parser.add_argument('--no-train', dest='train', action='store_false', help="Test model")
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
    elif not args.train:
        raise Exception('Please specify path to pretrained model')

    # Load MNIST Data, pre-train D for a couple of iterations and train model
    if args.train:
        X_train, y_train, _, _, N = import_mnist()
        model.pre_train(X_train, y_train)
        model.train(X_train,
            bs=args.batch_size,
            nb_epoch=args.nb_epochs,
            nb_iter=X_train.shape[0]//args.batch_size,
            y_train=y_train,
            save_path=args.save_path)

    # (Optional) Visualize results
    if args.visualize:
        model.visualize()

if __name__ == '__main__':
    main()
