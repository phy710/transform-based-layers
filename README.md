# TNNLS 2024 Multichannel Orthogonal Transform-Based Perceptron Layers for Efficient ResNets

Paper link: https://ieeexplore.ieee.org/abstract/document/10506207

The folder "layers" contains the implementation of the proposed layers. 
For example, if the input tensor is 3x16x32x32 and the output is 3x16x32x32, 
the single-path DCT-perceptron layer: 

    from layers.DCT import DCTConv2D
    DCTConv2D(32, 32, 16, 16, 1, residual=True)
3-path DCT-perceptron layer: 

    DCTConv2D(32, 32, 16, 16, 3, residual=False)
The parameter "pod" in the function "DCTConv2D" stands for the number of paths.

More examples can be found in the folder CIFAR10 and ImageNet1K.

To cite this work:

    @article{pan2024multichannel,
      title={Multichannel Orthogonal Transform-Based Perceptron Layers for Efficient ResNets},
      author={Pan, Hongyi and Hamdan, Emadeldeen and Zhu, Xin and Atici, Salih and Cetin, Ahmet Enis},
      journal={IEEE Transactions on Neural Networks and Learning Systems},
      year={2024},
      publisher={IEEE}
    }
