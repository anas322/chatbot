import random
import json

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]
# print(data)

"""
instantiates a neural network model, loads a saved model state into it, and sets it to evaluation mode, which is typically done when we want to use the model to make predictions on new data.

1.It loads the saved model state into the model using the load_state_dict method and the model_state variable, which should contain a dictionary with the model's state. This is typically used to load a previously trained model from a file.

2.the eval method. This changes the behavior of certain layers in the model, such as setting the dropout layers to evaluate mode, which disables dropout and uses the average of the input units. Evaluation mode is typically used when evaluating the model on a test dataset or when making predictions on new data.
"""
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()


bot_name = "Sam"

def get_response(msg):
    sentence = tokenize(msg)
    # print("Tokenization output example: ",sentence)
    X = bag_of_words(sentence, all_words)
    # print('bag of words output example',X)
    """
    reshape method is a function that is used to reshape the shape of an array. It takes in an array and a new shape as input, and returns a new array with the same data as the original array but with the specified shape.
    x.shape[0] ->  give the lengths of the corresponding array dimensions.
    """
    # print(X.shape[0])
    
    X = X.reshape(1, X.shape[0]) 
    # print("reshape array example: ",X,'\n')
    """
    A tensor is a multi-dimensional array that is used to represent data in machine learning. It is a generalization of a matrix, which is a two-dimensional array. Tensors can have any number of dimensions, but the most common ones are one-dimensional (vectors), two-dimensional (matrices), and three-dimensional (3D tensors).

    Tensors have several properties, including a rank (the number of dimensions), a shape (the size of each dimension), and a data type (e.g., float32, int64). They can also be manipulated using various operations, such as element-wise addition and multiplication, matrix multiplication, and slicing.
    """
    # NumPy arrays are similar to PyTorch tensors, but they are stored and manipulated differently under the hood.
    X = torch.from_numpy(X).to(device)
    # print("tensor array example: ",X,'\n')

    output = model(X)
    # print("output model",output,'\n')
    """
    torch.max is a function in the PyTorch library that returns the maximum value of a tensor along a specified dimension. It takes a tensor and an optional dimension as input and returns a tuple containing the maximum value of the tensor along the specified dimension and the corresponding indices of the maximum value.

    for example: 
        # Create a tensor
        a = torch.tensor([[1, 2, 3], [4, 5, 6]])

        # Get the maximum value along the rows (dimension 1)
        max_value, indices = torch.max(a, dim=1)

        print(max_value)  # Output: tensor([3, 6])
        print(indices)    # Output: tensor([2, 2])
    """
    _, predicted = torch.max(output, dim=1)
    # print("the predicted index: ",predicted,'\n')

    tag = tags[predicted.item()]
    # print("all the tags we have: ",tags,'\n')
    # print("the choosing tag: ",tag,'\n')
    
    """
    The softmax function is a mathematical function that takes in a vector of real numbers and returns a vector of the same size where each element is the exponentiated value of the corresponding element in the input vector, normalized so that the sum of all the elements is 1. The softmax function is often used in machine learning to convert a set of raw predictions into a ""probability distribution"", where the sum of the probabilities is 1.
    """
    probs = torch.softmax(output, dim=1)
    # print("all probabilities: ",probs,'\n')

    # take the propability of the predicted number 
    prob = probs[0][predicted.item()]
    # print("the choosing probability",prob,'\n')

    # check if the propability of this predcited number is greater than 75%
    if prob.item() > 0.45:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    
    return "I do not understand..."


if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = get_response(sentence)
        print(resp)

