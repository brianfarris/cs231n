import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  n_samples = X.shape[0]
  n_features = X.shape[1]
  n_classes = W.shape[1]
  f = np.zeros((n_samples,n_classes))
  f_y = np.zeros(n_samples)
  loss = 0.
  y_p = np.zeros((n_features,n_classes))
  for i in range(n_samples):
      for j in range(n_features):
          f_y[i] += X[i][j] * W[j][y[i]]
          for k in range(n_classes):
              f[i][k] += X[i][j]*W[j][k]
      
      #y_pi =  np.exp(f_y[i] + np.max(f[i,:])) / np.sum(np.exp(f[i,:] + np.max(f[i,:]) ) ) 
      loss += -np.log( np.exp(f_y[i] + np.max(f[i,:])) / np.sum(np.exp(f[i,:] + np.max(f[i,:]) ) ) ) / n_samples

  loss += 0.5 * reg * np.sum(W*W)
  yhat = np.zeros((n_samples, n_classes))
  yhat[np.arange(n_samples),y] = 1
  
  f = X.dot(W)
  y_p = np.exp(f) / np.sum(np.exp(f),axis=1).reshape(n_samples,1) 
  dW = -X.T.dot(yhat - y_p) / n_samples + reg * W
  

  #dW = -1./n_samples * X.dot(yhat - y_p)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  n_samples = X.shape[0]
  n_classes = W.shape[1]

  yhat = np.zeros((n_samples, n_classes))
  yhat[np.arange(n_samples),y] = 1.

  f = X.dot(W)
  f -= np.max(f, axis=1, keepdims=True)
  fexp = np.exp(f)
  fexpsum = np.sum(np.exp(f),axis=1, keepdims=True)

  y_p = fexp / fexpsum
  
  logy_p = f - np.log(fexpsum)

  loss = - (yhat * logy_p).sum() / np.float(n_samples)
  loss += 0.5 * reg * np.sum(W*W)

  dW = - X.T.dot( yhat - y_p ) / np.float(n_samples) + reg * W
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

