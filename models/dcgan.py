from .. import utils

class DCGAN(object):
    """
    Deep Convolutional GAN, as per https://arxiv.org/abs/1511.06434
    """

    def __init__(self, args):
        self.G, self.D, self.m = build_model()

    def build_model(self):
        G = generator()
        D = discriminator()
        m = Sequential([G, D])
        D.compile(Adam(1e-3), "binary_crossentropy")
        m.compile(Adam(1e-4), "binary_crossentropy")
        return G, D, m

    def train(self, X_train, nb_epoch=10000, bs=128):
        """
        Train DCGAN:
            - Train D to discriminate G results
            - Train G to fool D (D is frozen)
        """
        dl,gl=[],[] # Training history
        for e in tqdm(range(nb_epoch)):
            X,y = data_D(bs//2, self.G, X_train) # Get real and fake data + labels
            dl.append(self.D.train_on_batch(X,y)) # Train D
            make_trainable(self.D, False) # Freeze D
            gl.append(m.train_on_batch(noise(bs), np.zeros([bs]))) # Train G
            make_trainable(self.D, True) # Unfreeze D
        return dl,gl

    def data_D(self, sz, G, X_train):
        """
        Generate fake and real data to train D
        """
        real_img = X_train[np.random.randint(0,n,size=sz)]
        X = np.concatenate((real_img, G.predict(noise(sz))))
        return X, [0]*sz + [1]*sz

     def generator(self):
        """
        DCGAN Generator, small neural network with upsampling and LeakyReLU()
        """
        return Sequential([
            Dense(512*7*7, input_dim=100, activation=LeakyReLU()),
            BatchNormalization(mode=2),
            Reshape((7, 7, 512)),
            UpSampling2D(),
            Convolution2D(64, 3, 3, border_mode='same', activation=LeakyReLU()),
            BatchNormalization(mode=2),
            UpSampling2D(),
            Convolution2D(32, 3, 3, border_mode='same', activation=LeakyReLU()),
            BatchNormalization(mode=2),
            Convolution2D(1, 1, 1, border_mode='same', activation='sigmoid')
        ])

    def discriminator(self):
        """
        DCGAN Discriminator, small neural network with upsampling
        """
        return Sequential([
            Convolution2D(256, 5, 5, subsample=(2,2), border_mode='same',
            input_shape=(28, 28, 1), activation=LeakyReLU()),
            Convolution2D(512, 5, 5, subsample=(2,2), border_mode='same', activation=LeakyReLU()),
            Flatten(),
            Dense(256, activation=LeakyReLU()),
            Dense(1, activation = 'sigmoid')
        ])
