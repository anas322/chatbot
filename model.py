import torch
import torch.nn as nn


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.l2 = nn.Linear(hidden_size, hidden_size) 
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # no activation and no softmax at the end
        return out  

"""
This code defines a class called NeuralNet, which represents a neural network model. The NeuralNet class is a subclass of PyTorch's nn.Module class, which is a base class for creating neural network modules in PyTorch.

The NeuralNet class has the following features:

It has an __init__ method that initializes the model and defines its layers. The __init__ method takes in three arguments: input_size, hidden_size, and num_classes, which represent the size of the input layer, the size of the hidden layers, and the number of output classes, respectively. It uses these arguments to instantiate three linear layers (l1, l2, l3) using PyTorch's nn.Linear class and a ReLU activation layer (relu) using PyTorch's nn.ReLU class.
It has a forward method that defines the forward pass of the model. The forward method takes in an input tensor x and applies the linear layers and the ReLU activation function to it in sequence, returning the final output of the model. It's important to note that the forward method does not apply an activation function or a softmax function at the end, which means that the output of the model is not normalized and does not represent a probability distribution.

1.input_size represents the size of the input layer of the model. It specifies the number of input features that the model expects. For example, if the model is a classifier that takes in a sentence as input and the sentence is represented as a vector of word embeddings, then input_size would be the length of the word embedding vector.

2.hidden_size represents the size of the hidden layers of the model. It specifies the number of units in the hidden layers. The hidden layers are the layers in the model that come between the input and output layers and are not visible to the user. The number of hidden layers and their sizes are typically determined through experimentation and are important for the model's ability to learn complex patterns in the data.


3.output_size represents the size of the output layer of the model. It specifies the number of output classes that the model predicts. For example, if the model is a classifier that predicts the sentiment of a sentence (positive, negative, neutral), then output_size would be 3.

4.model_state is a variable that contains the saved state of a trained NeuralNet model. It is typically a dictionary that holds the weights and biases of the model's layers, as well as other information such as the model's hyperparameters and the state of the optimizer used to train the model.


The NeuralNet class has the following layers:

l1: This is a linear layer that takes in the input tensor and applies a linear transformation to it using a set of weights and biases. The output of this layer is passed through a ReLU activation function.

l2: This is a linear layer that takes in the output of the first linear layer (l1) and applies another linear transformation to it using a different set of weights and biases. The output of this layer is also passed through a ReLU activation function.

l3: This is a linear layer that takes in the output of the second linear layer (l2) and applies a final linear transformation to it using a third set of weights and biases. The output of this layer is the final output of the model.


Overall, these layers form a feedforward neural network with three linear layers and ReLU activation. The input is passed through the layers in sequence, and the final output is produced by the third linear layer (l3).
"""