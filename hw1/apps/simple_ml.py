"""hw1/apps/simple_ml.py"""

from audioop import cross
from socket import AF_X25
import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")
import needle as ndl


def parse_mnist(image_filesname, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    with gzip.open(image_filesname, "rb") as image, gzip.open(label_filename, "rb") as label:
      image_magic, image_number, image_rows, image_cols = struct.unpack(">IIII", image.read(16))
      image_pixels = struct.unpack("B" * (image_number * image_rows * image_cols), image.read())
      image_pixels_array = np.array(image_pixels, dtype=np.float32).reshape(image_number, image_rows * image_cols) / 255.0
        
      label_magic, label_number = struct.unpack(">II", label.read(8))
      label_values = struct.unpack("B" * label_number, label.read())
      label_values_array = np.array(label_values, dtype=np.uint8)

    return image_pixels_array, label_values_array
    ### END YOUR SOLUTION


def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    batch_size = Z.shape[0]
    _lhs = ndl.log(ndl.summation(ndl.exp(Z), axes=1))
    _rhs = ndl.summation(ndl.multiply(Z, y_one_hot), axes=1)

    _loss = ndl.summation((_lhs - _rhs))
 
    return _loss / batch_size
    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION
    num_examples, num_classes = X.shape[0], W2.shape[1]
    for i in range(0, num_examples, batch):
        sample = ndl.Tensor(X[i:i+batch, :])
        label = y[i:i+batch]

        Z = ndl.matmul(ndl.relu(ndl.matmul(sample, W1)), W2)
        I_y = np.zeros((batch, num_classes))
        I_y[np.arange(batch), label] = 1

        loss = softmax_loss(Z, ndl.Tensor(I_y))
        loss.backward()
        W1 = ndl.Tensor(W1.realize_cached_data() - lr * W1.grad.realize_cached_data())
        W2 = ndl.Tensor(W2.realize_cached_data() - lr * W2.grad.realize_cached_data())

    return W1, W2
    ### END YOUR SOLUTION


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
