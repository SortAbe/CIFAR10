# CIFAR10
These are entry level projects to help me understand machine learning.

This particualr neural network uses convolution layers. 

This particual project is focused on the cifar10 dataset:

https://www.cs.toronto.edu/~kriz/cifar.html

The architecture is as follows:
Input: 32x32x3
HiddenLayer1: Convolution(in: 3, out: 4, kernel: 5, stride 1, pad: 0)
MaxPool downscale half resolution.
HiddenLayer3:  Convolution(in: 6, out: 16, kernel: 4, stride 1, pad: 0)
MaxPool downscale half resolution.
HiddenLayer2: Linear + LeakyReLU (400,200)
HiidenLayer4: Linear + LeakyReLU (200,200)
Output: Linear (200,10)
Loss function: SoftMmax + Cross Entropy Loss
Optimizer: ADAM

I have not been able to get beyond 75% accuracy.
