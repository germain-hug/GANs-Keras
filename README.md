# GANs Implementations in Keras  

Keras implementation of:  
- [Deep Convolutional GAN](https://arxiv.org/abs/1511.06434)  
- [Wasserstein GAN](https://arxiv.org/abs/1701.07875)  
- [Conditional GAN](https://arxiv.org/abs/1411.1784)  
- [InfoGAN](https://arxiv.org/abs/1606.03657)  

The DCGAN code was inspired by Jeremy Howard's [course](http://course.fast.ai/)  
**TODO** -> Introduction


### Requirements:  

You will need [Keras 1.2.2](https://pypi.python.org/pypi/Keras/1.2.2) with a Tensorflow backend.  
To install dependencies, run `pip install -r requirements.txt`  

### DCGAN  
Deep Convolutional GANs was one of the first modifications made to the original GAN architecture to avoid mode collapsing. Theses improvements include:  
- Replacing pooling with strided convolutions
- Using Batch-Normalization in both G and D
- Starting G with a single Fully-Connected layer, end D with a flattening layer. The rest should be Fully-Convolutional
- Using LeakyReLU activations in D, ReLU in G, with the exception of the last layer of G which should be tanh  

**TODO** -> Add picture
Running pretrained model: `python main.py --type DCGAN --no-train --model weights/DCGAN.h5`  
Retraining: `python main.py --type DCGAN`  

### WGAN  
Following up on the DCGAN architecture, the Wasserstein GAN aims at leveraging another distance metric between distribution to train G and D. More specifically, they use the EM distance, which has the nice property of being continuous and differentiable for feed-forward networks. In practice, computing the EM distance is intractable, but we can approximate it by clipping the discriminator weights. The insures that D learns a K-Lipschitz function to compute the EM distance. Additionally, we:  
- Remove the sigmoid activation from D, leaving no constraint to its output range
- Use RMSprop optimizer over Adam  

**TODO** -> Add picture + explanations  
Running pretrained model: `python main.py --type WGAN --no-train --model weights/WGAN.h5`  
Retraining: `python main.py --type WGAN`  

### cGAN  

**TODO** -> Add picture + explanations  
Running pretrained model: `python main.py --type CGAN --no-train --model weights/cGAN.h5`  
Retraining: `python main.py --type CGAN`  

### InfoGAN  

**TODO** -> Add picture + explanations  
Running pretrained model: `python main.py --type InfoGAN --no-train --model weights/InfoGAN.h5`  
Retraining: `python main.py --type InfoGAN`  
