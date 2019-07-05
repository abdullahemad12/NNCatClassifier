import numpy as np


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
     
    Z = np.maximum(0, A)
    return Z

def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache["Z"]
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ



def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache["Z"]

    dZ = dA * sigmoid(Z) * (1 - sigmoid(Z))

    return dZ

