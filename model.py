import numpy as np
import matplotlib.pyplot as plt
from activation import *
from testCases_v4 import *


plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


np.random.seed(1)


def model_init(model_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """
    np.random.seed(3)
    dic = {}
    L = len(model_dims)
    for l in range(1, L):
        dic['W' + str(l)] = np.random.randn(model_dims[l], model_dims[l-1]) * 0.01
        dic['b' + str(l)] = np.zeros((model_dims[l], 1)) * 0.01

    return dic


def layer_forward(A, W, b):
    """
    Arguments:
    A -- np array containing the input to the this layer
    W -- the parameters of this layer
    b -- the bias to this layer
 
    Returns:
    Z -- input to the activation function
    cache -- the cached A, W, b values as a dictionary
    """

    Z = np.dot(W, A) + b

    cache = {'A': A, 'W': W, 'b': b}
    
    return (Z, cache) 


def layer_forward_activation(A_prev, W, b, activation):

    """
    Arguments:
    A_prev -- np array containing the input to the this layer
    W -- the parameters of this layer
    b -- the bias to this layer
 
    Returns:
    A -- output of the activation function
    cache -- the cached linear_cache, activation_cache values as a dictionary
    """

    A = None 
    linear_cache = None
    activation_cache = None
   
    if (activation == "relu"):
        (Z, linear_cache) = layer_forward(A_prev, W, b)
        A = relu(Z)
        activation_cache = {"Z": Z}        
    elif (activation == "sigmoid"):
        (Z, linear_cache) = layer_forward(A_prev, W, b)
        A = sigmoid(Z)
        activation_cache = {"Z": Z}


    cache = (linear_cache, activation_cache)

    return (A, cache) 




def L_model_forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
    
    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()
    
    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)
    """

    caches = []
    L = len(parameters) // 2 # divide by 2 because we have two variable for each parameter W and b
    A = X

    for l in range(1, L):
        A_prev = A
        (A, cache) =  layer_forward_activation(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = 'relu')
        caches.append(cache)

    
    (AL, cache) = layer_forward_activation(A, parameters['W' + str(L)], parameters['b' + str(L)], activation = 'sigmoid')
    caches.append(cache)

    return (AL, caches)




def compute_cost(AL, Y):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """
    
    m = Y.shape[1]

    
    cost = 1 / m * (- np.sum((Y * np.log(AL)) + (1 - Y) * np.log(1 - AL)))

    cost = np.squeeze(cost)
    return cost



def layer_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """


    (A_prev, W, b) = cache

    m = A_prev.shape[1]

    dW = (1 / m) * np.dot(dZ, A_prev.T)

    db = (1 / m) * np.sum(dZ, axis = 1, keepdims = True)

    dA_prev = np.dot(W.T, dZ)
   
    
    return (dA_prev, dW, db)



def layer_backward_activation(dA, cache, activation):

    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.
    
    Arguments:
    dA -- post-activation gradient for current layer l 
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """

    (linear_cache, activation_cache) = cache

    dA_prev = None
    dW = None
    db = None


    if activation == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)
        (dA_prev, dW, db) = layer_backward(dZ, linear_cache)

    elif activation == 'relu':
        dZ = relu_backward(dA, activation_cache)
        (dA_prev, dW, db) = layer_backward(dZ, linear_cache)        


    return (dA_prev, dW, db)



def L_model_backward(AL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    
    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
    
    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ... 
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ... 
    """

    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL


    dA =  - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    current_cache = caches[L-1]
    (grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)]) = layer_backward_activation(dA, current_cache, activation = 'sigmoid')

    for l in reversed(range(L-1)):

        current_cache = caches[l]
        (dATemp, dWTemp, dbTemp) = layer_backward_activation(grads["dA" + str(l+1)], current_cache, activation = 'relu')
        grads["dA" + str(l)] = dATemp
        grads["dW" + str(l+1)] = dWTemp
        grads["db" + str(l+1)] = dbTemp

    return grads
        



def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients, output of L_model_backward
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    """

    L = len(parameters) // 2

    for l in range(1, L+1):
        parameters["W" + str(l)] = parameters["W" + str(l)] - (learning_rate) * grads["dW" + str(l)]
        parameters["b" + str(l)] = parameters["b" + str(l)] - (learning_rate) * grads["db" + str(l)]

    return parameters    



def nn_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Arguments:
    X -- data, numpy array of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """


    np.random.seed(1)


    costs = []


    parameters = model_init(layers_dims)

    for i in range(num_iterations):

        (AL, caches) = L_model_forward(X, parameters) # Forward pass
        cost = compute_cost(AL, Y) # compute cost function
        grads = L_model_backward(AL, Y, caches) # Backward pass
        parameters = update_parameters(parameters, grads, learning_rate) # update our parameters


        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()


def predict(X, parameters):
    """ 
    Given a neural network classifies the input as 0 or 1
    Parameters:
    X -- the Input to the model
    Y -- Expected Output
    parameters -- the neural network parameters

    Returns:
    AL -- predictions of the model
    """
    (AL, caches) = L_model_forward(X, parameters)

    return AL

def calculate_accuracy(X, Y, parameters):

    """ 
       Given a neural network classifies the input as 0 or 1
       Parameters:
       X -- the Input to the model
       Y -- Expected Output
       parameters -- the neural network parameters
       
       Returns:
       AL -- predictions of the model
    """
    AL = predict(X, parameters)

    misclassified = np.abs(AL - Y)
    
    acc = (1 - np.mean(misclassified)) * 100

    print("Accuracy: {} %".format(acc))




