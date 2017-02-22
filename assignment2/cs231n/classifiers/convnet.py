import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ConvNet(object):
  """
  A convolutional network with the following architecture:
  
  [conv - relu - 2x2 max pool] x 2 - [affine - relu] x 2 - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3,
               use_batchnorm=False, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    self.use_batchnorm = use_batchnorm

    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    C, H, W = input_dim
    F = num_filters
    HH = filter_size
    WW = filter_size
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
    H_after_conv = 1 + (H + 2 * conv_param['pad'] - HH) / conv_param['stride']
    W_after_conv = 1 + (W + 2 * conv_param['pad'] - WW) / conv_param['stride']
    H_after_pool = 1 + (H_after_conv  - pool_param['pool_height']) / pool_param['stride']
    W_after_pool = 1 + (W_after_conv  - pool_param['pool_width']) / pool_param['stride']

    self.params['W1'] = weight_scale * np.random.normal(size=(F, C, HH, WW))
    self.params['W2'] = weight_scale * np.random.normal(size=(H_after_pool*W_after_pool*num_filters, hidden_dim)) 
    self.params['W3'] = weight_scale * np.random.normal(size=(hidden_dim, num_classes) )
    self.params['b1'] = weight_scale * np.random.normal(size=F)
    self.params['b2'] = weight_scale * np.random.normal(size=hidden_dim)
    self.params['b3'] = weight_scale * np.random.normal(size=num_classes)
    
    if self.use_batchnorm:
        self.params['gamma1'] = np.ones(F)
        self.params['beta1'] = np.zeros(F)
        self.params['gamma2'] = np.ones(hidden_dim)
        self.params['beta2'] = np.zeros(hidden_dim)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    if self.use_batchnorm:
        self.bn_params1 = {'mode':'train'}
        self.bn_params2 = {'mode':'train'}

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}


    mode = 'test' if y is None else 'train'

    if self.use_batchnorm: 
        self.bn_params1['mode'] = mode


    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    if self.use_batchnorm:
        gamma1 = self.params['gamma1']
        beta1 = self.params['beta1']
        out, cache1 = conv_bn_relu_pool_forward(X, gamma1, beta1, conv_param, self.bn_params1, pool_param, W1, b1) 
        gamma2 = self.params['gamma2']
        beta2 = self.params['beta2']
        out, cache2 = affine_bn_relu_forward(out,gamma2, beta2, self.bn_params2, W2,b2)
    else:
        out, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param) 
        out, cache2 = affine_relu_forward(out,W2,b2)
    scores, cache3 = affine_forward(out,W3,b3)


    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if mode == 'test':
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    reg = self.reg
    loss, dout = softmax_loss(scores,y)
    loss += 0.5 * reg * ( np.sum(W1*W1) + np.sum(W2*W2) + np.sum(W3*W3) )
    dx, dW3, db3 = affine_backward(dout, cache3)
    grads['W3'] = dW3 + reg * W3
    grads['b3'] = db3
    if self.use_batchnorm:
        dx, dgamma2, dbeta2, dW2, db2 = affine_bn_relu_backward(dx, cache2)
        grads['gamma2'] = dgamma2
        grads['beta2'] = dbeta2
    else:
        dx, dW2, db2 = affine_relu_backward(dx, cache2)
    grads['W2'] = dW2 + reg * W2
    grads['b2'] = db2
    if self.use_batchnorm:
        dx, dgamma1, dbeta1, dW1, db1 = conv_bn_relu_pool_backward(dx, cache1)
        grads['gamma1'] = dgamma1
        grads['beta1'] = dbeta1
    else:
        dx, dW1, db1 = conv_relu_pool_backward(dx, cache1)
    grads['W1'] = dW1 + reg * W1
    grads['b1'] = db1
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
   
    return loss, grads
  
  
pass


def conv_bn_relu_pool_forward(x, gamma, beta, conv_param, bn_param, pool_param, w, b):
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    a_norm, bn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(a_norm)
    out, pool_cache = max_pool_forward_fast(out, pool_param)
    cache = (conv_cache, bn_cache, relu_cache, pool_cache)
    return out, cache

def conv_bn_relu_pool_backward(dout, cache):
    conv_cache, bn_cache, relu_cache, pool_cache = cache
    ds = max_pool_backward_fast(dout, pool_cache)
    da = relu_backward(ds, relu_cache)
    daa, dgamma, dbeta = spatial_batchnorm_backward(da, bn_cache)
    dx, dw, db = conv_backward_fast(daa, conv_cache)
    return dx, dgamma, dbeta, dw, db

def affine_bn_relu_forward(x, gamma, beta, bn_param, w, b):
    a, fc_cache = affine_forward(x, w, b)
    a_norm, bn_cache = batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(a_norm)
    cache = (fc_cache, bn_cache, relu_cache)
    return out, cache

def affine_bn_relu_backward(dout, cache):
    fc_cache, bn_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    daa, dgamma, dbeta = batchnorm_backward(da, bn_cache)
    dx, dw, db = affine_backward(daa, fc_cache)
    return dx, dgamma, dbeta, dw, db

