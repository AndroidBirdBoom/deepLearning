import tensorflow as tf
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from tf_utils import *

tf.compat.v1.disable_eager_execution()

np.random.seed(1)


def tf_test():
    y_hat = tf.constant(36, name='y_hat')
    y = tf.constant(39, name='y')
    loss = tf.Variable((y - y_hat) ** 2, name='loss')
    init = tf.compat.v1.global_variables_initializer()

    with tf.compat.v1.Session() as session:
        session.run(init)
        print(session.run(loss))

    a = tf.constant(2)
    b = tf.constant(10)
    c = tf.multiply(a, b)
    print(c)
    sess = tf.compat.v1.Session()
    print(sess.run(c))

    x = tf.compat.v1.placeholder(tf.int64, name='x')
    print(sess.run(2 * x, feed_dict={x: 3}))
    sess.close()


def linear_function():
    """
    Implements a linear function:
            Initializes W to be a random tensor of shape (4,3)
            Initializes X to be a random tensor of shape (3,1)
            Initializes b to be a random tensor of shape (4,1)
    Returns:
    result -- runs the session for Y = WX + b
    """
    np.random.seed(1)
    X = tf.constant(np.random.randn(3, 1), name='X')
    W = tf.constant(np.random.randn(4, 3))
    b = tf.constant(np.random.randn(4, 1))
    Y = tf.add(tf.matmul(W, X), b)
    session = tf.compat.v1.Session()
    result = session.run(Y)
    session.close()
    return result


def sigmoid(z):
    """
    Computes the sigmoid of z

    Arguments:
    z -- input value, scalar or vector

    Returns:
    results -- the sigmoid of z
    """
    x = tf.compat.v1.placeholder(tf.float32, name='x')
    y = tf.sigmoid(x, name='y')
    with tf.compat.v1.Session() as session:
        results = session.run(y, feed_dict={x: z})
    return results


def cost(logits, labels):
    """
    Computes the cost using the sigmoid cross entropy
    Arguments:
    logits -- vector containing z, output of the last linear unit (before the final sigmoid activation)
    labels -- vector of labels y (1 or 0)

    Note: What we've been calling "z" and "y" in this class are respectively called "logits" and "labels"
    in the TensorFlow documentation. So logits will feed into z, and labels into y.

    Returns:
    cost -- runs the session of the cost (formula (2))
    """
    z = tf.constant(logits, dtype=tf.float32, name='z')
    y = tf.constant(labels, dtype=tf.float32, name='y')
    c = tf.nn.sigmoid_cross_entropy_with_logits(y, z, name='c')
    sess = tf.compat.v1.Session()
    cost = sess.run(c)
    sess.close()
    return cost


def one_hot_matrix(labels, C):
    """
    Creates a matrix where the i-th row corresponds to the ith class number and the jth column
                     corresponds to the jth training example. So if example j had a label i. Then entry (i,j)
                     will be 1.

    Arguments:
    labels -- vector containing the labels
    C -- number of classes, the depth of the one hot dimension

    Returns:
    one_hot -- one hot matrix
    """
    # 这两行加不加都行
    C = tf.constant(C)
    x = tf.constant(labels)
    y_hat = tf.one_hot(x, C, axis=0)
    sess = tf.compat.v1.Session()
    result = sess.run(y_hat)
    sess.close()
    return result


def ones(shape):
    """
    Creates an array of ones of dimension shape

    Arguments:
    shape -- shape of the array you want to create

    Returns:
    ones -- array containing only ones
    """
    x = tf.ones(shape)
    sess = tf.compat.v1.Session()
    result = sess.run(x)
    sess.close()
    return result


def model(X, Y, layer_dimes, count=10000, learn_rate=0.0007):
    parameters = init_parameters(layer_dimes)

    seed = 10

    for i in range(count):

        seed = seed + 1
        minibatches = random_mini_batches(X, Y, seed=seed)
        for minibatch in minibatches:
            (minibatch_X, minibatch_Y) = minibatch
            AL, caches = L_forward(minibatch_X, parameters)

            cost = comput_cost(AL, minibatch_Y)

            if i % 100 == 0:
                print('cost = ', cost)

            grids = L_backward(minibatch_Y, AL, caches)

            parameters = update_parameters(grids, parameters, learn_rate)

    return parameters


# 生成多个batch
def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    np.random.seed(seed)
    m = X.shape[1]

    permutation = list(np.random.permutation(m))  # 它会返回一个长度为m的随机数组，且里面的数是0到m-1
    shuffled_X = X[:, permutation]  # 将每一列的数据按permutation的顺序来重新排列。
    shuffled_Y = Y[:, permutation].reshape((1, m))

    count = int(m / mini_batch_size)
    count = count if m % mini_batch_size == 0 else count + 1
    mini_batches = []

    for i in range(count):
        # 最后一项，注意不要越界
        if i == count - 1:
            mini_batch_X = shuffled_X[:, i * mini_batch_size:]
            mini_batch_Y = shuffled_Y[:, i * mini_batch_size:]
        else:
            mini_batch_X = shuffled_X[:, i * mini_batch_size: (i + 1) * mini_batch_size]
            mini_batch_Y = shuffled_Y[:, i * mini_batch_size: (i + 1) * mini_batch_size]

        mini_batches.append([mini_batch_X, mini_batch_Y])

    return mini_batches


def update_parameters(grids, parameters, learn_rate):
    L = len(parameters) // 2

    for i in range(L):
        parameters['W' + str(i + 1)] -= learn_rate * grids['dW' + str(i + 1)]
        parameters['b' + str(i + 1)] -= learn_rate * grids['db' + str(i + 1)]

    return parameters


def L_backward(Y, AL, caches):
    grids = {}
    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    L = len(caches)
    for i in range(L, 0, -1):
        linear_cache, activation_cache = caches[i - 1]
        if i == L:
            dZ = get_dz(dAL, activation_cache, 'sigmoid')
        else:
            dZ = get_dz(dA_p, activation_cache, 'relu')
        dA_p, dW, db = get_dwb(dZ, linear_cache)
        grids['dW' + str(i)] = dW
        grids['db' + str(i)] = db
    return grids


def get_dz(dA, activation_cache, activation='relu'):
    if activation == 'sigmoid':
        sig, cache = sigmoid(activation_cache)
        return dA * sig * (1 - sig)
    elif activation == 'relu':
        dZ = np.array(dA, copy=True)
        dZ[activation_cache <= 0] = 0
        return dZ


def get_dwb(dZ, linear_cache):
    m = dZ.shape[1]
    A_p, W, b = linear_cache
    dA_p = np.dot(W.T, dZ)
    dW = np.dot(dZ, A_p.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    return dA_p, dW, db


def comput_cost(AL, Y):
    m = Y.shape[1]
    cost = -np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL))) / m
    return cost


def init_parameters(layer_dimes):
    np.random.seed(3)
    parameters = {}
    for i, value in enumerate(layer_dimes):
        if i == 0:
            continue
        else:
            parameters['W' + str(i)] = np.random.randn(value, layer_dimes[i - 1]) * np.sqrt(2 / layer_dimes[i - 1])
            parameters['b' + str(i)] = np.zeros((value, 1))
    return parameters


def L_forward(X, parameters):
    L = len(parameters) // 2
    caches = []

    A = X
    for i in range(L):
        Z, linear_cache = forward(A, parameters['W' + str(i + 1)], parameters['b' + str(i + 1)])
        if i != L - 1:
            A, activation_cache = activation(Z, 'relu')
        else:
            AL, activation_cache = activation(Z, 'sigmoid')

        caches.append((linear_cache, activation_cache))

    return AL, caches


def activation(Z, activcation='relu'):
    if activcation == 'relu':
        A, activcation_cache = relu(Z)
    elif activcation == 'sigmoid':
        A, activcation_cache = sigmoid(Z)

    return A, activcation_cache


def relu(Z):
    return np.maximum(0, Z), Z


def sigmoid(Z):
    return 1.0 / (1.0 + np.exp(-Z)), Z


def forward(A_pre, W, b):
    Z = np.dot(W, A_pre) + b
    return Z, (A_pre, W, b)


from course_2_week_2 import opt_utils

if __name__ == "__main__":
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

    # index = 0
    # plt.imshow(X_train_orig[index])
    # plt.show()
    # print("y = " + str(np.squeeze(Y_train_orig[:, index])))

    X_train = X_train_orig.reshape(X_train_orig.shape[0], -1).T / 255.
    X_test = X_test_orig.reshape(X_test_orig.shape[0], -1).T / 255.

    Y_train = one_hot_matrix(np.squeeze(Y_train_orig), 6)
    Y_test = one_hot_matrix(np.squeeze(Y_test_orig), 6)

    train_X, train_Y = opt_utils.load_dataset()
    plt.show()
    layers_dims = [train_X.shape[0], 5, 2, 1]
    parameters = model(train_X, train_Y, layers_dims)

    # Predict
    predictions = opt_utils.predict(train_X, train_Y, parameters)

    # Plot decision boundary
    plt.title("Model with Gradient Descent optimization")
    axes = plt.gca()
    axes.set_xlim([-1.5, 2.5])
    axes.set_ylim([-1, 1.5])
    opt_utils.plot_decision_boundary(lambda x: opt_utils.predict_dec(parameters, x.T), train_X, train_Y)
