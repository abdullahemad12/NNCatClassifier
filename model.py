import numpy as np
import matplotlib.pyplot as plt

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


def sigmoid(A):
    """
    Arguments:
    A -- np array containing the input to the sigmoid function
    
    Returns:
    parameters -- np array containing the output of the sigmoid function
    """
    Z = 1 / (1 + np.exp(-A))
    return Z

def relu(A):
    """
    Arguments:
    A -- np array containing the input to the relu function
    
    Returns:
    parameters -- np array containing the output of the relu function
    """
     
    A[A < 0] = 0
    return A

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

dZ, linear_cache = linear_backward_test_case()

dA_prev, dW, db = layer_backward(dZ, linear_cache)
print ("dA_prev = "+ str(dA_prev))
print ("dW = " + str(dW))
print ("db = " + str(db))

    
