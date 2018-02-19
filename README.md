# GANs Implementations in Keras  

Implementation of:  
- [Deep Convolutional GAN](https://arxiv.org/abs/1511.06434)  
- [Wasserstein GAN](https://arxiv.org/abs/1701.07875)  
- [Conditional GAN](https://arxiv.org/abs/1411.1784)  
- [InfoGAN](https://arxiv.org/abs/1606.03657)  

### Requirements:
You will need Keras 1.2.2 with a Tensorflow backend.
To install dependencies, run `pip install -r requirements.txt`  

### DCGAN  

Running pretrained model: `python main.py --type DCGAN --no-train --model weights/DCGAN.h5`  
Retraining: `python main.py --type DCGAN`  

### WGAN  

Running pretrained model: `python main.py --type WGAN --no-train --model weights/WGAN.h5`  
Retraining: `python main.py --type WGAN`  

### cGAN  

Running pretrained model: `python main.py --type CGAN --no-train --model weights/cGAN.h5`  
Retraining: `python main.py --type CGAN`  
