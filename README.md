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

**TODO** -> Add picture + explanations
Running pretrained model: `python main.py --type DCGAN --no-train --model weights/DCGAN.h5`  
Retraining: `python main.py --type DCGAN`  

### WGAN  

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
