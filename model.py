import numpy as np
import matplotlib.pyplot as plt



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
    Z = 1 / (1 + np.exp(A))
    return Z

def relu(A):
    """
    Arguments:
    A -- np array containing the input to the relu function
    
    Returns:
    parameters -- np array containing the output of the relu function
    """
     
    return np.max(0, A)


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



A = np.array([[ 1.62434536, -0.61175641], [-0.52817175, -1.07296862], [ 0.86540763, -2.3015387 ]])
W = np.array([[ 1.74481176, -0.7612069,  0.3190391 ]])
b = np.array([[-0.24937038]])

(Z, cache) = layer_forward(A, W, b)
print(Z)


