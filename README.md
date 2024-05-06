# TNNLS 2024 Multichannel Orthogonal Transform-Based Perceptron Layers for Efficient ResNets

Paper link: https://ieeexplore.ieee.org/abstract/document/10506207

The folder "layers" contains the implementation of the proposed layers. 
For example, if the input tensor is 3x16x32x32 and the output is 3x16x32x32, 
the single-path DCT-perceptron layer: 

    DCTConv2D(32, 32, 16, 16, 1, residual=True)
3-path DCT-perceptron layer: 

    DCTConv2D(32, 32, 16, 16, 3, residual=False)
The parameter "pod" in the function "DCTConv2D" stands for the number of paths.

More examples can be found in the folder CIFAR10 and ImageNet1K.
