import tensorflow as tf
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from tf_utils import *

from matplotlib.font_manager import FontProperties

font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)  # 解决windows环境下画图汉字乱码问题

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


def model_(X, Y, layer_dimes, count=10000, learn_rate=0.0007):
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
    shuffled_Y = Y[:, permutation]

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


def create_placeholders(n_x, n_y):
    """
    Creates the placeholders for the tensorflow session.

    Arguments:
    n_x -- scalar, size of an image vector (num_px * num_px = 64 * 64 * 3 = 12288)
    n_y -- scalar, number of classes (from 0 to 5, so -> 6)

    Returns:
    X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
    Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"

    Tips:
    - You will use None because it let's us be flexible on the number of examples you will for the placeholders.
      In fact, the number of examples during test/train is different.
    """
    X = tf.compat.v1.placeholder(dtype=tf.float32, shape=[n_x, None])
    Y = tf.compat.v1.placeholder(dtype=tf.float32, shape=[n_y, None])

    return X, Y


def initialize_parameters():
    """
    Initializes parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [25, 12288]
                        b1 : [25, 1]
                        W2 : [12, 25]
                        b2 : [12, 1]
                        W3 : [6, 12]
                        b3 : [6, 1]

    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    """
    tf.random.set_seed(1)
    W1 = tf.compat.v1.get_variable('W1', [25, 12288], initializer=tf.keras.initializers.glorot_normal(seed=1))
    b1 = tf.compat.v1.get_variable('b1', [25, 1], initializer=tf.zeros_initializer())
    W2 = tf.compat.v1.get_variable('W2', [12, 25], initializer=tf.keras.initializers.glorot_normal(seed=1))
    b2 = tf.compat.v1.get_variable('b2', [12, 1], initializer=tf.zeros_initializer())
    W3 = tf.compat.v1.get_variable('W3', [6, 12], initializer=tf.keras.initializers.glorot_normal(seed=1))
    b3 = tf.compat.v1.get_variable('b3', [6, 1], initializer=tf.zeros_initializer())
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}

    return parameters


def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX

    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    Z1 = tf.add(tf.matmul(W1, X), b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)
    return Z3


def compute_cost(Z3, Y):
    """
    Computes the cost

    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3

    Returns:
    cost - Tensor of the cost function
    """

    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    return cost


def model(X_train, Y_train, X_test, Y_test, learning_rate=0.0001,
          num_epochs=1500, minibatch_size=32, print_cost=True):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.

    Arguments:
    X_train -- training set, of shape (input size = 12288, number of training examples = 1080)
    Y_train -- test set, of shape (output size = 6, number of training examples = 1080)
    X_test -- training set, of shape (input size = 12288, number of training examples = 120)
    Y_test -- test set, of shape (output size = 6, number of test examples = 120)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    tf.random.set_seed(1)
    seed = 3
    costs = []
    m = X_train.shape[1]

    X, Y = create_placeholders(X_train.shape[0], Y_train.shape[0])

    parameters = initialize_parameters()

    Z3 = forward_propagation(X, parameters)

    cost = compute_cost(Z3, Y)

    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    init = tf.compat.v1.global_variables_initializer()

    with tf.compat.v1.Session() as sess:

        sess.run(init)

        for i in range(num_epochs):
            epoch_cost = 0.
            num_minibatches = int(m / minibatch_size)  # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                epoch_cost += minibatch_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and i % 100 == 0:
                print("Cost after epoch %i: %f" % (i, epoch_cost))
            if print_cost == True and i % 5 == 0:
                costs.append(epoch_cost)

        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        parameters = sess.run(parameters)
        print("Parameters have been trained!")

        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))

        return parameters


def softmax(Z3):
    z = np.exp(Z3)
    z_sum = np.sum(z, axis=0)
    A = z / z_sum
    return A


def predict(my_image, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    Z1 = np.dot(W1, my_image) + b1
    A1, _ = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2, _ = relu(Z2)
    Z3 = np.dot(W3, A2) + b3
    A3 = softmax(Z3)

    m = my_image.shape[1]
    result = []
    for i in range(m):
        result.append(np.argmax(A3[:, i], axis=0))

    return result


import imageio
from PIL import Image
from scipy import ndimage

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

    parameters = model(X_train, Y_train, X_test, Y_test, num_epochs=100)
    # layer_dims = (X_train.shape[0], 25, 12, 6)
    # parameters = model_(X_train, Y_train, layer_dims)

    m = 10
    images = None
    for i in range(m):
        # my_image = str(i + 1) + ".png"
        # frame = "test_imgs/" + my_image
        #
        # image = np.array(imageio.imread(frame))
        plt.subplot(2, m / 2, i + 1)
        image = X_train_orig[i]
        plt.imshow(image)

        my_image = image.reshape(-1, 1) / 255.
        print('shape = ', my_image.shape)
        if my_image.shape[0] == 12288:
            if images is None:
                images = my_image
            else:
                images = np.hstack((images, my_image))

    result = predict(images, parameters=parameters)
    for i, values in enumerate(result):
        plt.subplot(2, m/2, i + 1)
        plt.title('预测结果为：' + str(values), fontproperties=font)
    plt.show()

    # train_X, train_Y = opt_utils.load_dataset()
    # plt.show()
    # layers_dims = [train_X.shape[0], 5, 2, 1]
    # parameters = model(train_X, train_Y, layers_dims)
    #
    # # Predict
    # predictions = opt_utils.predict(train_X, train_Y, parameters)
    #
    # # Plot decision boundary
    # plt.title("Model with Gradient Descent optimization")
    # axes = plt.gca()
    # axes.set_xlim([-1.5, 2.5])
    # axes.set_ylim([-1, 1.5])
    # opt_utils.plot_decision_boundary(lambda x: opt_utils.predict_dec(parameters, x.T), train_X, train_Y)
