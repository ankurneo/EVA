 Build the following network:

    
    That takes a CIFAR10 image (32x32x3)

    Add 3 Convolutions to arrive at AxAx48 dimensions (e.g. 32x32x3 |    3x3x3x16 >> 3x3x16x32 >> 3x3x32x48)

    Apply GAP and get 1x1x48, call this X

    Create a block called ULTIMUS that:

    Creates 3 FC layers called K, Q and V such that:

    X*K = 48*48x8 > 8

    X*Q = 48*48x8 > 8 

    X*V = 48*48x8 > 8 

    then create AM = SoftMax(QTK)/(8^0.5) = 8*8 = 8
    then Z = V*AM = 8*8 > 8
    then another FC layer called Out that:
    Z*Out = 8*8x48 > 48

    Repeat this Ultimus block 4 times

    Then add final FC layer that converts 48 to 10 and sends it to the loss function.

    Model would look like this 

    C>C>C>U>U>U>U>FFC>Loss

    Train the model for 24 epochs using the OCP that I wrote in class. Use ADAM as an optimizer. **

        Batch size = 512
    
 Logs-:
    
      Epoch 0 : 
      Train set: Average loss: 2.3020, Accuracy: 13.16

      Test set: Average loss: 0.005, Accuracy: 10.00

      Epoch 1 : 
      Train set: Average loss: 2.3049, Accuracy: 10.63

      Test set: Average loss: 0.005, Accuracy: 10.00

      Epoch 2 : 
      Train set: Average loss: 2.3163, Accuracy: 9.95

      Test set: Average loss: 0.005, Accuracy: 10.00

      Epoch 3 : 
      Train set: Average loss: 56678.2969, Accuracy: 10.07

      Test set: Average loss: 368.239, Accuracy: 10.00

      Epoch 4 : 
      Train set: Average loss: 65302.3750, Accuracy: 10.02

      Test set: Average loss: 884.942, Accuracy: 10.00

      Epoch 5 : 
      Train set: Average loss: 1764.8209, Accuracy: 9.93

      Test set: Average loss: 17.722, Accuracy: 10.00

      Epoch 6 : 
      Train set: Average loss: 23000.9414, Accuracy: 10.06

      Test set: Average loss: 12.444, Accuracy: 10.00

      Epoch 7 : 
      Train set: Average loss: 538.4319, Accuracy: 9.99

      Test set: Average loss: 1.217, Accuracy: 10.00

      Epoch 8 : 
      Train set: Average loss: 64.2081, Accuracy: 9.79

      Test set: Average loss: 0.209, Accuracy: 10.00

      Epoch 9 : 
      Train set: Average loss: 11.8270, Accuracy: 9.96

      Test set: Average loss: 0.096, Accuracy: 10.00

      Epoch 10 : 
      Train set: Average loss: 7.6269, Accuracy: 9.84

      Test set: Average loss: 0.020, Accuracy: 10.00

      Epoch 11 : 
      Train set: Average loss: 6.4092, Accuracy: 9.85

      Test set: Average loss: 0.014, Accuracy: 10.00

      Epoch 12 : 
      Train set: Average loss: 8.5962, Accuracy: 9.94

      Test set: Average loss: 0.024, Accuracy: 10.00

      Epoch 13 : 
      Train set: Average loss: 8.0253, Accuracy: 9.97

      Test set: Average loss: 0.021, Accuracy: 10.00

      Epoch 14 : 
      Train set: Average loss: 8.9160, Accuracy: 9.95

      Test set: Average loss: 0.022, Accuracy: 10.00

      Epoch 15 : 
      Train set: Average loss: 3.2180, Accuracy: 10.04

      Test set: Average loss: 0.009, Accuracy: 10.00

      Epoch 16 : 
      Train set: Average loss: 3.5376, Accuracy: 9.99

      Test set: Average loss: 0.010, Accuracy: 10.00

      Epoch 17 : 
      Train set: Average loss: 2.5492, Accuracy: 9.80

      Test set: Average loss: 0.005, Accuracy: 10.00

      Epoch 18 : 
      Train set: Average loss: 2.4133, Accuracy: 10.01

      Test set: Average loss: 0.005, Accuracy: 10.00

      Epoch 19 : 
      Train set: Average loss: 2.3110, Accuracy: 10.01

      Test set: Average loss: 0.005, Accuracy: 10.00

      Epoch 20 : 
      Train set: Average loss: 2.3217, Accuracy: 9.96

      Test set: Average loss: 0.005, Accuracy: 10.00

      Epoch 21 : 
      Train set: Average loss: 2.3014, Accuracy: 9.90

      Test set: Average loss: 0.005, Accuracy: 10.00

      Epoch 22 : 
      Train set: Average loss: 2.3024, Accuracy: 9.87

      Test set: Average loss: 0.005, Accuracy: 10.00

      Epoch 23 : 
      Train set: Average loss: 2.3029, Accuracy: 10.05

      Test set: Average loss: 0.005, Accuracy: 10.00
