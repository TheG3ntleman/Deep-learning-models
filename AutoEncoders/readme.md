# Autoencoders

AUtoencoders are unsupervised deep learning models. They take in the features our input and try to reconstruct them. There are many applications of autoencoders. If you don't know how they work you should learn that before you try to understand the code.

Learning Resources:
    - http://ufldl.stanford.edu/tutorial/unsupervised/Autoencoders/
    -

## Reconstructive Auto Encoders
1] [Basic autoencoder](https://github.com/TheG3ntleman/Deep-learning-models/blob/master/AutoEncoders/Autoencoder.py): A simple 
    implementation of a autoencoder. That reconstructs the MNIST dataset.
    
2] [Prob Autoencoder](https://github.com/TheG3ntleman/Deep-learning-models/blob/master/AutoEncoders/ProbAutoencoder.py): My own
    creation it uses a softmax function after the encoder generates the latent space allowing us to later use the decoder by making
    our own latent space probability vectors. Trained on the MNIST dataset.
