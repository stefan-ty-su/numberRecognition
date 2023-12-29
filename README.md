# numberRecognition
## Description
Digit recognition fully connected neural network, with the option of creating a neural network with 1 or 2 hidden layers and the customisation of the number of nodes in each layer.
- All combinations implement ReLU as the activation function

Repo offers interface to create custom number image for predictions.

## Usage
### Drawing Custom Image for prediction
To create custom image to test the neural network, run the following:

```console
> python interaction.py
```
This will create a window which enables 'drawing' on it.
Once the drawing of a number is complete, press the corresponding button in which was draw (0-9), the image is then converted to MNIST format, and the canvas is reset, allowing for continual drawing.

### Creating and Training a Neural Network
To create a neural network, the following arguments are required:
- -d: the dataset used for training (mnist or custom)
- -l: the number of hidden layers in the neural network (1 or 2)
- -i: the number of iterations (integer)
- -a: the learning rate of the neural network (alpha)

Example:
The following example creates a 1 layer network which is trained off the mnist dataset at a learning rate of 0.1 and 500 iterations.
```
> python createNetwork.py -d mnist -l 1 -i 500 -a 0.1
```

The program will then require user input on the number of nodes for each layer.

### Output
For every 50 iterations during training, the accuracy of the neural network is printed.
After training, the neural network is run on the validation set, then the accuracy of said set is outputted.
The neural network is also run on the custom test data, which also the accuracy.
