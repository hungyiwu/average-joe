# Image convergence: regularize by data

Leveraging similar concept as in N2N [1], variational autoencoder (VAE) [2], and Siamese network [3]. The regularization effect is similar to VAE: in VAE, the model aims to reconstruct the input image from an off-by-a-bit latent vector (sampled from the latent space), while here the model aims to reconstruct an off-by-a-bit target image (shuffled with label preserved) from the latent vector. The net effect is the model converged to the typical image(s) within the label group. The latent space from models like this allows new image generation, clustering, and other applications.

## Example training data
![digit_2](figures/train_2.png)  
![digit_3](figures/train_3.png)  
![digit_4](figures/train_4.png)  

## Example test result
![digit_2](figures/eval_2.png)  
![digit_3](figures/eval_3.png)  
![digit_4](figures/eval_4.png)  

## Reference
[1] Noise2Noise: Learning Image Restoration without Clean Data ([arxiv](https://arxiv.org/abs/1803.04189))  
[2] Variational autoencoder ([Wikipedia](https://en.wikipedia.org/wiki/Autoencoder#Variational_autoencoder_(VAE)))  
[3] Siamese neural network ([Wikipedia](https://en.wikipedia.org/wiki/Siamese_neural_network))  
