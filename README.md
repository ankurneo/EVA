  Write a custom ResNet architecture for CIFAR10 that has the following architecture:

    PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]

    Layer1 -
    X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
    R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] 
    Add(X, R1)

    Layer 2 -
    Conv 3x3 [256k]
    MaxPooling2D
    BN
    ReLU

    Layer 3 -
    X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
    R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
    Add(X, R2)

    MaxPooling with Kernel Size 4

    FC Layer

    SoftMax

    Uses One Cycle Policy such that:

    Total Epochs = 24

    Max at Epoch = 5

    LRMIN = FIND

    LRMAX = FIND

    NO Annihilation

    Uses this transform -RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8)

    Batch size = 512
    
 Logs-:
    
    Epoch 0 : 
    Train set: Average loss: 1.4403, Accuracy: 41.00

    Test set: Average loss: 0.004, Accuracy: 41.30

    Epoch 1 : 
    Train set: Average loss: 1.2742, Accuracy: 48.88

    Test set: Average loss: 0.003, Accuracy: 53.71

    Epoch 2 : 
    Train set: Average loss: 1.3152, Accuracy: 60.43

    Test set: Average loss: 0.002, Accuracy: 61.22

    Epoch 3 : 
    Train set: Average loss: 0.8507, Accuracy: 63.69

    Test set: Average loss: 0.002, Accuracy: 71.81

    Epoch 4 : 
    Train set: Average loss: 0.8564, Accuracy: 73.83

    Test set: Average loss: 0.001, Accuracy: 76.79

    Epoch 5 : 
    Train set: Average loss: 0.7402, Accuracy: 77.91

    Test set: Average loss: 0.002, Accuracy: 75.58

    Epoch 6 : 
    Train set: Average loss: 0.4352, Accuracy: 80.57

    Test set: Average loss: 0.001, Accuracy: 81.34

    Epoch 7 : 
    Train set: Average loss: 0.3789, Accuracy: 83.87

    Test set: Average loss: 0.001, Accuracy: 82.24

    Epoch 8 : 
    Train set: Average loss: 0.2953, Accuracy: 86.83

    Test set: Average loss: 0.001, Accuracy: 85.68

    Epoch 9 : 
    Train set: Average loss: 0.2981, Accuracy: 89.23

    Test set: Average loss: 0.001, Accuracy: 84.72

    Epoch 10 : 
    Train set: Average loss: 0.2272, Accuracy: 90.51

    Test set: Average loss: 0.001, Accuracy: 85.78

    Epoch 11 : 
    Train set: Average loss: 0.2021, Accuracy: 92.28

    Test set: Average loss: 0.001, Accuracy: 86.94

    Epoch 12 : 
    Train set: Average loss: 0.1481, Accuracy: 93.19

    Test set: Average loss: 0.001, Accuracy: 87.48

    Epoch 13 : 
    Train set: Average loss: 0.2045, Accuracy: 94.48

    Test set: Average loss: 0.001, Accuracy: 87.24

    Epoch 14 : 
    Train set: Average loss: 0.1044, Accuracy: 95.62

    Test set: Average loss: 0.001, Accuracy: 88.64

    Epoch 15 : 
    Train set: Average loss: 0.1349, Accuracy: 96.51

    Test set: Average loss: 0.001, Accuracy: 88.62

    Epoch 16 : 
    Train set: Average loss: 0.0990, Accuracy: 97.26

    Test set: Average loss: 0.001, Accuracy: 89.00

    Epoch 17 : 
    Train set: Average loss: 0.0383, Accuracy: 97.57

    Test set: Average loss: 0.001, Accuracy: 88.98

    Epoch 18 : 
    Train set: Average loss: 0.0791, Accuracy: 98.01

    Test set: Average loss: 0.001, Accuracy: 89.42

    Epoch 19 : 
    Train set: Average loss: 0.0403, Accuracy: 98.29

    Test set: Average loss: 0.001, Accuracy: 89.38

    Epoch 20 : 
    Train set: Average loss: 0.0615, Accuracy: 98.42

    Test set: Average loss: 0.001, Accuracy: 89.69

    Epoch 21 : 

    Test set: Average loss: 0.001, Accuracy: 89.61

    Epoch 22 : 
    Train set: Average loss: 0.0519, Accuracy: 98.74

    Test set: Average loss: 0.001, Accuracy: 89.62

    Epoch 23 : 
    Train set: Average loss: 0.0207, Accuracy: 98.91

    Test set: Average loss: 0.001, Accuracy: 89.81
