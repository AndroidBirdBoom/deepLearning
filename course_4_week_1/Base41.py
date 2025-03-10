import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)


def zero_pad(X, pad):
    """
    Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image,
    as illustrated in Figure 1.

    Argument:
    X -- python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
    pad -- integer, amount of padding around each image on vertical and horizontal dimensions

    Returns:
    X_pad -- padded image of shape (m, n_H + 2*pad, n_W + 2*pad, n_C)
    """
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant')
    return X_pad


def conv_single_step(a_slice_prev, W, b):
    """
    Apply one filter defined by parameters W on a single slice (a_slice_prev) of the output activation
    of the previous layer.

    Arguments:
    a_slice_prev -- slice of input data of shape (f, f, n_C_prev)
    W -- Weight parameters contained in a window - matrix of shape (f, f, n_C_prev)
    b -- Bias parameters contained in a window - matrix of shape (1, 1, 1)

    Returns:
    Z -- a scalar value, result of convolving the sliding window (W, b) on a slice x of the input data
    """
    Z = a_slice_prev * W
    Z = np.sum(Z) + b
    return Z


def conv_forward(A_prev, W, b, hparameters):
    """
    Implements the forward propagation for a convolution function

    Arguments:
    A_prev -- output activations of the previous layer, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
    b -- Biases, numpy array of shape (1, 1, 1, n_C)
    hparameters -- python dictionary containing "stride" and "pad"

    Returns:
    Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward() function
    """
    stride = hparameters['stride']
    pad = hparameters['pad']
    n_H_prev = A_prev.shape[1]
    n_W_prev = A_prev.shape[2]
    f = W.shape[0]
    n_C = W.shape[-1]
    n_H = int(np.floor((n_H_prev + 2 * pad - f) / stride) + 1)
    n_W = int(np.floor((n_W_prev + 2 * pad - f) / stride) + 1)
    A_prev_pad = zero_pad(A_prev, pad)
    m = A_prev.shape[0]
    Z = np.zeros((m, n_H, n_W, n_C))

    # 该层是m个数据集分别循环
    for index in range(m):
        # 需要将过滤器分别循环一下求，因为conv_single_step不支持多个过滤器同时求
        for c in range(n_C):
            # 列固定，行进行循环
            for j in range(n_W):
                for i in range(n_H):
                    Z_single = conv_single_step(
                        A_prev_pad[index, i * stride:i * stride + f, j * stride:j * stride + f, :], W[:, :, :, c],
                        b[:, :, :, c])
                    Z[index, i, j, c] = Z_single

    cache = (A_prev, W, b, hparameters)
    return Z, cache


def pool_forward(A_prev, hparameters, mode="max"):
    """
    Implements the forward pass of the pooling layer

    Arguments:
    A_prev -- Input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    hparameters -- python dictionary containing "f" and "stride"
    mode -- the pooling mode you would like to use, defined as a string ("max" or "average")

    Returns:
    A -- output of the pool layer, a numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache used in the backward pass of the pooling layer, contains the input and hparameters
    """
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    f = hparameters['f']
    stride = hparameters['stride']

    n_H = int(np.floor((n_H_prev - f) / stride)) + 1
    n_W = int(np.floor((n_W_prev - f) / stride)) + 1
    n_C = n_C_prev

    A = np.zeros((m, n_H, n_W, n_C))

    for index in range(m):
        for c in range(n_C):
            for i in range(n_H):
                for j in range(n_W):
                    A_slice = A_prev[index, i * stride:i * stride + f, j * stride:j * stride + f, c]
                    if mode == 'max':
                        A_target_data = np.max(A_slice)
                    elif mode == 'average':
                        A_target_data = np.mean(A_slice)

                    A[index, i, j, c] = A_target_data

    cache = (A_prev, hparameters)
    return A, cache





# z = np.arange(16).reshape(2, 2, 2, 2)
# print(z, '\n\n')
#
# z[:, 0, 0, 1] = 100
# print(z)
