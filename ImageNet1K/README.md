These experiments are based on PyTorch's official ImageNet training:

    https://github.com/pytorch/examples/tree/main/imagenet

DCTResNet50x3 is the 3-path DCT-ResNet50, and the input size is 224x224.

DCTResNet50x3_256 is the 3-path DCTResNet50, but the input size is 256x256.

To train the network:
        
    python main.py -a dct_resnet50 -b 128 --lr 0.05

To test the network:

    python test.py -a dct_resnet50 -b 128 -b10 32

Other networks's training is similar. Please change "dct" to "wht" or "bwt" or change "50" to "101".

The test code contains a 10-fold test, and the 10-fold test batch size is 32. We reduce this size to avoid the memory issue. 
