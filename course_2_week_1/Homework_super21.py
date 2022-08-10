import numpy as np
import matplotlib.pyplot as plt
from scipy import io
from course_2_week_1 import init_utils
from course_2_week_1 import reg_utils, gc_utils
import sklearn
from matplotlib.font_manager import FontProperties
import h5py

font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)  # 解决windows环境下画图汉字乱码问题


def load_dataset(is_plot=True):
    np.random.seed(1)
    train_X, train_Y = sklearn.datasets.make_circles(n_samples=300, noise=0.05)
    np.random.seed(2)
    test_X, test_Y = sklearn.datasets.make_circles(n_samples=100, noise=0.05)
    if is_plot:
        plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y)
    return train_X.T, train_Y.reshape((1, train_Y.shape[0])), test_X.T, test_Y.reshape((1, test_Y.shape[0]))


def model(X, Y, learning_rate=0.01, num_iterations=15000, print_cost=True, initialization="he", is_polt=True):
    '''
    神经网络步骤：
    1. 初始化参数
    2. 前向传播
    3. 计算代价
    4. 后向传播
    5. 更新参数
    6. 预测
    '''
    grads = {}
    costs = []  # to keep track of the loss
    m = X.shape[1]  # number of examples
    layers_dims = [X.shape[0], 10, 5, 1]

    # 初始化参数
    if initialization == "zeros":
        parameters = initialize_parameters_zeros(layers_dims)
    elif initialization == "random":
        parameters = initialize_parameters_random(layers_dims)
    elif initialization == 'he':
        parameters = initialize_parameters_he(layers_dims)
    for i in range(num_iterations):

        # 先前向传播
        AL, caches = L_model_forward(X, parameters)

        # 计算代价函数
        cost = compute_cost(AL, Y)
        if i % 100 == 0 and print_cost:
            print('cost = ', cost)
        if i % 1000 == 0:
            costs.append(cost)

        # 反向传播
        grads = L_model_backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)

    # 学习完毕，绘制成本曲线
    if is_polt:
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("initialization = " + initialization + ",Learning rate =" + str(learning_rate))
        # plt.show()

    # 返回学习完毕后的参数
    return parameters


# 更新参数
def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2

    for i in range(L):
        parameters['W' + str(i + 1)] -= learning_rate * grads['dW' + str(i + 1)]
        parameters['b' + str(i + 1)] -= learning_rate * grads['db' + str(i + 1)]

    return parameters


# 反向传播
def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = Y.shape[1]
    # 最后一层和其他层用的是两个激活函数，需要分别求
    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL)) / m
    for i in range(L, 0, -1):
        linear_cache, activation_cache = caches[i - 1]
        if i == L:
            dZ = backward_dZ(dAL, activation_cache, 'sigmoid')
            # dZ = 1. / m * (AL - Y)
            dA_p, dW, db = backward_dWb(dZ, linear_cache)
        else:
            dZ = backward_dZ(dA_p, activation_cache, 'relu')
            dA_p, dW, db = backward_dWb(dZ, linear_cache)
        grads['dW' + str(i)] = dW
        grads['db' + str(i)] = db
    return grads


# 求dZ
def backward_dZ(dA, activation_cache, activation):
    Z = activation_cache
    if activation == 'relu':
        dZ = backward_relu(dA, Z)
    elif activation == 'sigmoid':
        dZ = backward_sigmoid(dA, Z)

    return dZ


def backward_relu(dA, Z):
    dRelu = np.ones(dA.shape)
    dRelu[Z <= 0] = 0
    dZ = dRelu * dA
    return dZ


def backward_sigmoid(dA, Z):
    sigmoid = 1.0 / (1.0 + np.exp(-Z))
    dSigmoid = sigmoid * (1 - sigmoid)
    dZ = dA * dSigmoid
    return dZ


# 求dw,db,da_p
def backward_dWb(dZ, linear_cache):
    A_p, W, b = linear_cache
    dA_p = np.dot(W.T, dZ)
    dW = np.dot(dZ, A_p.T)
    db = np.sum(dZ, axis=1, keepdims=True)
    return dA_p, dW, db


def backward_dWb_reg(dZ, linear_cache, lamda):
    m = dZ.shape[1]
    A_p, W, b = linear_cache
    dA_p = np.dot(W.T, dZ)
    dW = np.dot(dZ, A_p.T) + lamda / m * W
    db = np.sum(dZ, axis=1, keepdims=True)
    return dA_p, dW, db


# 代价函数
def compute_cost(AL, Y):
    m = Y.shape[1]
    logprobs = np.multiply(-np.log(AL), Y) + np.multiply(-np.log(1 - AL), 1 - Y)
    loss = 1. / m * np.nansum(logprobs)
    return loss


# 前向传播
def L_model_forward(X, parameters):
    caches = []
    L = len(parameters) // 2
    A = X
    for i in range(L):
        if i == L - 1:
            AL, cache = forward_detail(A, parameters['W' + str(i + 1)], parameters['b' + str(i + 1)], "sigmoid")
        else:
            A, cache = forward_detail(A, parameters['W' + str(i + 1)], parameters['b' + str(i + 1)], "relu")
        caches.append(cache)
    return AL, caches


# 前向传播具体实现
def forward_detail(A_prev, W, b, activation='relu'):
    Z, linear_cache = forward_detail_Z(A_prev, W, b)
    if activation == 'relu':
        A, activation_cache = forward_relu(Z)
    elif activation == 'sigmoid':
        A, activation_cache = forward_sigmoid(Z)

    return A, (linear_cache, activation_cache)


def forward_detail_Z(A_prev, W, b):
    Z = np.dot(W, A_prev) + b
    return Z, (A_prev, W, b)


# 前向传播实现
def forward_relu(Z):
    A = np.maximum(0, Z)
    return A, Z


def forward_sigmoid(Z):
    return 1.0 / (1.0 + np.exp(-Z)), Z


# 初始化参数-0
def initialize_parameters_zeros(layers_dims):
    parameters = {}
    for i, e in enumerate(layers_dims):
        if i == 0:
            continue
        else:
            parameters['W' + str(i)] = np.zeros((e, layers_dims[i - 1]))
            parameters['b' + str(i)] = np.zeros((e, 1))
    return parameters


# 初始化参数-随机
def initialize_parameters_random(layers_dims):
    np.random.seed(3)
    parameters = {}
    for i, e in enumerate(layers_dims):
        if i == 0:
            continue
        else:
            parameters['W' + str(i)] = np.random.randn(e, layers_dims[i - 1])
            parameters['b' + str(i)] = np.zeros((e, 1))
    return parameters


# 初始化参数-he
def initialize_parameters_he(layers_dims):
    np.random.seed(3)
    parameters = {}
    for i, e in enumerate(layers_dims):
        if i == 0:
            continue
        else:
            parameters['W' + str(i)] = np.random.randn(e, layers_dims[i - 1]) * np.sqrt(2.0 / layers_dims[i - 1])
            parameters['b' + str(i)] = np.zeros((e, 1))
    return parameters


def predict(X, y, parameters):
    """
    This function is used to predict the results of a  n-layer neural network.

    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model

    Returns:
    p -- predictions for the given dataset X
    """

    m = X.shape[1]
    p = np.zeros((1, m), dtype=np.int)

    # Forward propagation
    a3, caches = L_model_forward(X, parameters)

    # convert probas to 0/1 predictions
    for i in range(0, a3.shape[1]):
        if a3[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    # print results
    print("Accuracy: " + str(np.mean((p[0, :] == y[0, :]))))

    return p


def plot_decision_boundary(model, X, y, string):
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    x2_min, x2_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(x2_min, x2_max, h))
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.title(string + "条件下的分类情况", fontproperties=font)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y[0], s=10, cmap=plt.cm.Spectral)
    # plt.show()


def predict_dec(parameters, X):
    """
    Used for plotting decision boundary.

    Arguments:
    parameters -- python dictionary containing your parameters
    X -- input data of size (m, K)

    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """

    # Predict using forward propagation and a classification threshold of 0.5
    a3, cache = L_model_forward(X, parameters)
    predictions = (a3 > 0.5)
    return predictions


def model_reg(X, Y, learning_rate=0.3, num_iterations=30000, print_cost=True, lambd=0, keep_prob=1, is_polt=True):
    '''
        神经网络步骤：
        1. 初始化参数
        2. 前向传播
        3. 计算代价
        4. 后向传播
        5. 更新参数
        6. 预测
        '''
    grads = {}
    costs = []  # to keep track of the loss
    m = X.shape[1]  # number of examples
    layers_dims = [X.shape[0], 20, 3, 1]

    # 初始化参数
    parameters = initialize_parameters_random(layers_dims)

    for i in range(num_iterations):

        # Forward propagation: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID.
        if keep_prob == 1:
            AL, caches = L_model_forward(X, parameters)
        elif keep_prob < 1:
            a3, caches = forward_propagation_with_dropout(X, parameters, keep_prob)

        # Cost function
        if lambd == 0:
            cost = compute_cost(AL, Y)
        else:
            cost = compute_cost_with_regularization(AL, Y, parameters, lambd)

        # Backward propagation.
        assert (lambd == 0 or keep_prob == 1)  # it is possible to use both L2 regularization and dropout,
        # but this assignment will only explore one at a time
        if lambd == 0 and keep_prob == 1:
            grads = L_model_backward(AL, Y, caches)
        elif lambd != 0:
            grads = backward_propagation_with_regularization(AL, Y, caches, lambd)
        elif keep_prob < 1:
            grads = backward_propagation_with_dropout(AL, Y, caches, keep_prob)
        # # 先前向传播
        # AL, caches = L_model_forward(X, parameters)
        #
        # # 计算代价函数
        # cost = compute_cost(AL, Y)
        if i % 100 == 0 and print_cost:
            print('cost = ', cost)
        if i % 1000 == 0:
            costs.append(cost)

        parameters = update_parameters(parameters, grads, learning_rate)

    # 学习完毕，绘制成本曲线
    if is_polt:
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

    # 返回学习完毕后的参数
    return parameters


def forward_propagation_with_dropout():
    pass


def compute_cost_with_regularization(AL, Y, parameters, lambd):
    m = Y.shape[1]

    cost = -np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL)) / m
    cost2 = compute_cost(AL, Y)
    L = len(parameters) // 2
    for i in range(L):
        cost += lambd / (2 * m) * np.sum(parameters['W' + str(i + 1)] ** 2)
    return cost


def backward_propagation_with_regularization(AL, Y, caches, lambd):
    grads = {}
    L = len(caches)
    m = Y.shape[1]
    # 最后一层和其他层用的是两个激活函数，需要分别求
    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL)) / m
    for i in range(L, 0, -1):
        linear_cache, activation_cache = caches[i - 1]
        if i == L:
            dZ = backward_dZ(dAL, activation_cache, 'sigmoid')
            # dZ = 1. / m * (AL - Y)
            dA_p, dW, db = backward_dWb_reg(dZ, linear_cache, lambd)
        else:
            dZ = backward_dZ(dA_p, activation_cache, 'relu')
            dA_p, dW, db = backward_dWb_reg(dZ, linear_cache, lambd)
        grads['dW' + str(i)] = dW
        grads['db' + str(i)] = db
    return grads


def backward_propagation_with_dropout(AL, Y, cache, keep_prob):
    pass


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


if __name__ == "__main__":
    # train_X, train_Y, test_X, test_Y = init_utils.load_dataset(True)
    # plt.show()
    #
    # plt.figure(figsize=(76, 26))
    # initializations = ['zeros', 'random', 'he']
    # for i, v in enumerate(initializations):
    #     plt.subplot(2, len(initializations), i + 1)
    #     parameters = model(train_X, train_Y, 0.01, initialization=v)
    #     p = predict(train_X, train_Y, parameters)
    #     predictions_test = init_utils.predict(test_X, test_Y, parameters)
    #
    #     plt.subplot(2, len(initializations), len(initializations) + i + 1)
    #     plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y, "参数初始化为:" + v)
    #
    # plt.show()

    # train_x_orig, train_y, test_x_orig, test_y, classes = load_dataset()
    #
    # m_train = train_x_orig.shape[0]
    # num_px = train_x_orig.shape[1]
    # m_test = test_x_orig.shape[0]
    #
    # train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0],
    #                                        -1).T  # The "-1" makes reshape flatten the remaining dimensions
    # test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
    #
    # # Standardize data to have feature values between 0 and 1.
    # train_x = train_x_flatten / 255.
    # test_x = test_x_flatten / 255.
    #
    # layers_dims = [12288, 20, 7, 5, 1]
    # parameters = model(train_x, train_y,learning_rate=0.0075, num_iterations=3000, print_cost=True, initialization="he", is_polt=True)
    # pred_train = predict(train_x, train_y, parameters)
    # pred_test = predict(test_x, test_y, parameters)

    # 正则化
    train_X, train_Y, test_X, test_Y = reg_utils.load_2D_dataset()
    plt.show()

    parameters = model_reg(train_X, train_Y,lambd=0.7)
    print("On the training set:")
    predictions_train = predict(train_X, train_Y, parameters)
    print("On the test set:")
    predictions_test = predict(test_X, test_Y, parameters)

    plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y, "参数初始化为:")
    plt.show()
