# Deeper
A view of the deep learning landscape; implementations of old and new deep learning architectures. I aim to implement state of the art methods in machine learning using the latest implementations of Tensorflow (2.0) or PyTorch. 

## Models Implemneted

### Autoencoders
- VAE
- Gaussian Mixture VAE
    - [x] Marginalised VAE
    - [x] Gumble Softmax discrete sampling
- Gaussian Mixture VAE with GAN Error
    - [] Matginalised VAE
    - [x] Gumble softmax discrete sampling

### Attention Models
- [] Attention Graph
- [] BERT (pretraining): https://arxiv.org/pdf/1903.10145.pdf


## Extra Notes

### Optimization
TODO: learnable dropout http://proceedings.mlr.press/v108/boluki20a/boluki20a.pdf


### Variational Inference / latent spaces
- Beta VAE. Penalising the KL divergence (Beta > 1) term within the loss can force the latent space representation to be encoded in a more efficient manner; forcing the model to encode the most efficient representation of the data (https://openreview.net/references/pdf?id=Sy2fzU9gl). Further extended by (https://arxiv.org/pdf/1802.04942.pdf) and explained by (https://arxiv.org/abs/1711.00464).
- 
- Using Cyclic beta constants with KL divergence term within VAE enables better reconstruction learning (for the same loss) at the cost of latent space divergence. Follows on from Beta-VAE results (https://arxiv.org/pdf/1903.10145.pdf).