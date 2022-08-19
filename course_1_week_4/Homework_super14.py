import numpy as np
import h5py
import matplotlib.pyplot as plt

from course_1_week_4.testCases import linear_forward_test_case, linear_backward_test_case, \
    linear_activation_forward_test_case, linear_activation_backward_test_case, L_model_forward_test_case, \
    compute_cost_test_case, L_model_backward_test_case, update_parameters_test_case


def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


# 初始化 两层网络参数
def initialize_parameters(n_x, n_h, n_y):
    """
        Argument:
        n_x -- size of the input layer
        n_h -- size of the hidden layer
        n_y -- size of the output layer

        Returns:
        parameters -- python dictionary containing your parameters:
                        W1 -- weight matrix of shape (n_h, n_x)
                        b1 -- bias vector of shape (n_h, 1)
                        W2 -- weight matrix of shape (n_y, n_h)
                        b2 -- bias vector of shape (n_y, 1)
    """
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }
    return parameters


# 初始化 深层网络模型参数
def initialize_parameters_deep(layer_dims):
    parameters = {}
    for i in range(len(layer_dims)):
        if i == 0:
            continue
        else:
            parameters[str('W%d' % i)] = np.random.randn(layer_dims[i], layer_dims[i - 1]) * 0.01
            parameters[str('b%d' % i)] = np.zeros((layer_dims[i], 1))
    return parameters


# 前向网络传播 - 浅层
def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter
    cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache


# 激活函数-线性部分
def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value
    cache -- a python dictionary containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """
    Z, linear_cache = linear_forward(A_prev, W, b)

    if activation == "sigmoid":
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        A, activation_cache = relu(Z)

    cache = (linear_cache, activation_cache)
    return A, cache


# 前向传播 - 多个
def L_model_forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()

    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                the cache of linear_sigmoid_forward() (there is one, indexed L-1)
    """
    L = len(parameters) // 2  # 层数
    A = X
    caches = []
    for i in range(L):
        if i == L - 1:  # 最后一层，使用sigmoid
            AL, cache = linear_activation_forward(A, parameters['W%d' % (i + 1)], parameters['b%d' % (i + 1)],
                                                  "sigmoid")
        else:
            A, cache = linear_activation_forward(A, parameters['W%d' % (i + 1)], parameters['b%d' % (i + 1)], "relu")

        caches.append(cache)

    return AL, caches


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z)), z


def relu(z):
    return np.maximum(0, z), z


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
    cost = -np.sum(np.multiply(np.log(AL), Y) + np.multiply(np.log(1 - AL), 1 - Y)) / m

    # 矩阵转数组(单个数字)
    cost = np.squeeze(cost)
    return cost


def linear_backward(dZ, cache):
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
    m = dZ.shape[1]
    A_prev, W, b = cache[0], cache[1], cache[2]
    dA_prev = np.dot(W.T, dZ)
    dW = np.dot(dZ, A_prev.T)
    db = np.sum(dZ, axis=1, keepdims=True)
    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
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
    linear_cache, activation_cache = cache
    if activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def sigmoid_backward(dA, cache):
    z = cache
    sig = 1.0 / (1.0 + np.exp(-z)) * (1 - 1.0 / (1.0 + np.exp(-z)))
    dZ = dA * sig
    return dZ


def relu_backward(dA, cache):
    z = cache
    dZ = np.array(dA, copy=True)
    dZ[z <= 0] = 0
    return dZ


# 反向传播迭代
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
    L = len(caches)
    grads = {}
    m = Y.shape[1]
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    for i in range(L, 0, -1):
        if i == L:
            dA_prev, dW, db = linear_activation_backward(dAL, caches[i - 1], "sigmoid")
        else:
            dA_prev, dW, db = linear_activation_backward(dAL, caches[i - 1], "relu")

        grads["dA" + str(i)] = dAL
        grads["dW" + str(i)] = dW
        grads["db" + str(i)] = db
        dAL = dA_prev

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
    for i in range(L):
        parameters['W' + str(i + 1)] -= learning_rate * grads['dW' + str(i + 1)]
        parameters['b' + str(i + 1)] -= learning_rate * grads['db' + str(i + 1)]
    return parameters


def two_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    """
    Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID.

    Arguments:
    X -- input data, of shape (n_x, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- dimensions of the layers (n_x, n_h, n_y)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- If set to True, this will print the cost every 100 iterations

    Returns:
    parameters -- a dictionary containing W1, W2, b1, and b2
    """
    np.random.seed(1)
    costs = []
    parameters = initialize_parameters_deep(layers_dims)
    for i in range(num_iterations):
        AL, caches = L_model_forward(X, parameters)
        cost = compute_cost(AL, Y)
        if i % 100 == 0:
            costs.append(cost)
        if i % 100 == 0 and print_cost:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        grads = L_model_backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters


def predict(X, y, parameters):
    """
    预测结果
    :param X:
    :param y:
    :param parameters:
    :return:
    """
    m = X.shape[1]
    p = np.zeros((1, m))

    # 根据参数前向传播
    probas, caches = L_model_forward(X, parameters)

    for i in range(0, probas.shape[1]):
        if probas[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    print("准确度为: " + str(float(np.sum((p == y)) / m)))
    return p


def print_mislabeled_images(classes, X, y, p):
    """
	绘制预测和实际不同的图像。
	    X - 数据集
	    y - 实际的标签
	    p - 预测
    """
    a = p + y
    mislabeled_indices = np.asarray(np.where(a == 1))
    plt.rcParams['figure.figsize'] = (40.0, 40.0)  # set default size of plots
    num_images = len(mislabeled_indices[0])
    for i in range(num_images):
        index = mislabeled_indices[1][i]

        plt.subplot(2, num_images, i + 1)
        plt.imshow(X[:, index].reshape(64, 64, 3), interpolation='nearest')
        plt.axis('off')
        plt.title(
            "Prediction: " + classes[int(p[0, index])].decode("utf-8") + " \n Class: " + classes[y[0, index]].decode(
                "utf-8"))
    plt.show()


def load_planar_dataset():
    np.random.seed(1)  # 设置种子，为后续验证使用
    m = 400  # 样本数是400
    N = int(m / 2)  # 此案例分为两个类，每个类的样本数大小
    D = 2  # 维度
    a = 4
    X = np.zeros((m, D))  # x矩阵是一个400*2大小的矩阵
    y = np.zeros((m, 1))

    for i in range(D):
        ix = range(i * N, (i + 1) * N)
        t = np.linspace(i * 3.12, (i + 1) * 3.12, N) + np.random.randn(N) * 0.2  # theta
        r = a * np.sin(4 * t) + np.random.randn(N) * 0.2  # radius
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]  # == np.hstack((r*np.sin(t),r*np.cos(t)))
        y[ix] = i
        # plt.scatter(r * np.sin(t), r * np.cos(t), c='r' if i == 0 else 'b')
    # plt.show()

    X = X.T
    y = y.T

    return X, y


def plot_decision_boundary(model, X, y):
    x1_min, x1_max = X[0, :].min() - 1, X[0, :].max() + 1  # 挑选x1的最大最小值，用于绘制plot的x轴
    x2_min, x2_max = X[1, :].min() - 1, X[1, :].max() + 1  # 挑选x2的最值，用于绘制plot的y轴
    h = 0.01  # 间隔
    xx, yy = np.meshgrid(np.arange(x1_min, x1_max, h), np.arange(x2_min, x2_max, h))  # 生成网格
    # 预测 表格数据
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)


if __name__ == "__main__":
    train_x_orig, train_y, test_x_orig, test_y, classes = load_dataset()

    m_train = train_x_orig.shape[0]
    num_px = train_x_orig.shape[1]
    m_test = test_x_orig.shape[0]

    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0],
                                           -1).T  # The "-1" makes reshape flatten the remaining dimensions
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

    # Standardize data to have feature values between 0 and 1.
    train_x = train_x_flatten / 255.
    test_x = test_x_flatten / 255.

    '''
    单隐藏层的步骤：
    1. 初始化参数
    （多次迭代2-4）
    2. 前向传播
    3. 求代价函数
    4. 后向传播
    5. 预测
    '''
    # 2层网络结构（单隐藏层）
    n_x = train_x_flatten.shape[0]
    n_h = 7
    n_y = 1
    layers_dims = (n_x, n_h, n_y)
    parameters = two_layer_model(train_x, train_y, layers_dims, num_iterations=2500, print_cost=True)
    pred_train = predict(train_x, train_y, parameters)
    pred_test = predict(test_x, test_y, parameters)

    # 深层网络结构
    # layers_dims = [12288, 20, 7, 5, 1]
    # parameters = two_layer_model(train_x, train_y, layers_dims, num_iterations=2500, print_cost=True)
    # pred_train = predict(train_x, train_y, parameters)
    # pred_test = predict(test_x, test_y, parameters)

    # X, y = load_planar_dataset()
    #
    # layers_dims = [X.shape[0], 20, 7, 5, 1]
    # parameters = two_layer_model(X, y, layers_dims, print_cost=True)
    #
    # plot_decision_boundary(lambda x: predict(x.T, y, parameters), X, y)
    # plt.show()
