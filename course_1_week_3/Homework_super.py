import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from matplotlib.font_manager import FontProperties

font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)  # 解决windows环境下画图汉字乱码问题

from course_1_week_3.planar_utils import load_extra_datasets


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


def layer_sizes(X, Y):
    n_x = X.shape[0]  # 输入x（特征向量）的大小
    n_h = 4  # 固定隐藏层的大小
    n_y = Y.shape[0]  # 输出值y的大小

    return n_x, n_h, n_y


# 初始化数据大小
def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(2)
    W1 = np.random.randn(n_h, n_x) * 0.01  # 初始化w1，乘以0.01的作用是使参数尽可能小，后续反向传播时速度更快
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    return W1, b1, W2, b2


# 正向传播过程
def forward_propagation(X, W1, b1, W2, b2):
    Z1 = np.dot(W1, X) + b1  # Z1 = (4,m) 竖向排列
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2  # Z2 = (1,m)
    A2 = sigmoid(Z2)
    return Z1, A1, Z2, A2


def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))


# 计算损失函数
def compute_cost(A2, Y, parameters):
    m = Y.shape[1]
    cost = -(np.dot(Y, np.log(A2).T) + np.dot(1 - Y, np.log(1 - A2).T)) / m
    return cost


# 反向传播算法
def backward_propagation(W1, b1, W2, b2, Z1, A1, Z2, A2, X, Y):
    m = Y.shape[1]
    # 与Z2同维度相同，下方的求解也是一样，找对应的属性的维度，便于计算或排查错误
    dZ2 = A2 - Y  # (1,m)
    dW2 = np.dot(dZ2, A1.T) / m  # (1,4)
    # db2 = (np.dot(dZ2, dZ2.T)) / m  # (1,1)   出错了，计算和而不是平方和
    db2 = np.sum(dZ2, axis=1) / m
    dZ1 = np.dot(W2.T, dZ2) * (1 - A1 ** 2)  # (4,m)
    dW1 = np.dot(dZ1, X.T) / m  # (4,2)
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m  # (4,1)
    return dZ1, dW1, db1, dZ2, dW2, db2


# 梯度下降
def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate=1.2):
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    return W1, b1, W2, b2


# 合并步骤
def nn_model(X, y, n_h, num_iterations=10000, print_cost=False):
    # 初始化参数
    n_x, n_h_1, n_y = layer_sizes(X, y)
    W1, b1, W2, b2 = initialize_parameters(n_x, n_h, n_y)

    costs = []
    for i in range(num_iterations):
        # 正向传播
        Z1, A1, Z2, A2 = forward_propagation(X, W1, b1, W2, b2)

        # 计算损失函数
        cost = compute_cost(A2, y, None)

        if i % 100 == 0:
            costs.append(cost)

        if print_cost and i % 1000 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

        # 反向传播
        dZ1, dW1, db1, dZ2, dW2, db2 = backward_propagation(W1, b1, W2, b2, Z1, A1, Z2, A2, X, y)
        # 梯度下降
        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, 0.5)

    return W1, b1, W2, b2, costs


def predict(W1, b1, W2, b2, X):
    Z1, A1, Z2, A2 = forward_propagation(X, W1, b1, W2, b2)
    A2[A2 > 0.5] = 1
    A2[A2 <= 0.5] = 0

    # 或者使用以下方法
    y_predict = np.where(A2 > 0.5, 1, 0)

    return A2


TEST = True

if __name__ == "__main__":
    # 创建数据集
    X, y = load_planar_dataset()
    print('X的维度:', X.shape)
    print('Y的维度:', y.shape)
    if TEST:
        plt.title('show point')
        plt.scatter(X[0, :], X[1, :], c=y, s=40, cmap=plt.cm.Spectral)
        plt.show()

    # 线性分类器效果
    clf = linear_model.LogisticRegression()
    clf.fit(X.T, y.T)

    plot_decision_boundary(lambda x: clf.predict(x), X, y)
    if TEST:
        plt.title('Logistic Regression')
        Z = clf.predict(X.T)
        print("正确率为：", float(np.dot(y, Z) + np.dot(1 - y, 1 - Z)) / float(y.size) * 100, "%")
        plt.show()

    '''
    浅层神经网络的实现步骤：
    1. 初始化参数w,b,隐藏层数
    2. 正向传播获取hx
    3. 反向传播+ 梯度下降算法修正w,b
    4. 验证
    '''

    hidden_layer_sizes = [1, 2, 3, 4, 5, 10]
    plt.figure(figsize=(56, 16))
    W1 = None
    b1 = None
    W2 = None
    b2 = None
    for i, n_h in enumerate(hidden_layer_sizes):
        W1, b1, W2, b2, costs = nn_model(X, y, n_h, 10000, True)
        plt.subplot(2, len(hidden_layer_sizes), i + 1)
        plt.plot(np.squeeze(costs))
        plt.xlabel('迭代次数', fontproperties=font)
        plt.ylabel('代价函数', fontproperties=font)
        plt.title(('代价函数随迭代次数变化图(hidden_layer_size = %d)' % i), fontproperties=font)
        # 预测
        y_predict = predict(W1, b1, W2, b2, X)
        print('准确率为：', float(np.dot(y_predict, y.T) + np.dot(1 - y_predict, (1 - y).T)) / float(y.size) * 100, '%')

        plt.subplot(2, len(hidden_layer_sizes), len(hidden_layer_sizes) + i + 1)
        plot_decision_boundary(lambda x: predict(W1, b1, W2, b2, x.T), X, y)
        plt.title('浅层神经网络学习效果图(hidden_layer_size = %d)' % i, fontproperties=font)

    plt.show()

    # 额外测试
    noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()

    datasets = {"noisy_circles": noisy_circles,
                "noisy_moons": noisy_moons,
                "blobs": blobs,
                "gaussian_quantiles": gaussian_quantiles}

    dataset = "gaussian_quantiles"

    X, y = datasets[dataset]
    X, y = X.T, y.reshape(1, y.shape[0])

    # make blobs binary
    if dataset == "blobs":
        y = y % 2

    # Visualize the data
    plt.scatter(X[0, :], X[1, :], c=y.ravel(), s=40, cmap=plt.cm.Spectral)
    plt.show()

    plt.figure(figsize=(56, 16))

    for i, n_h in enumerate(hidden_layer_sizes):
        W1, b1, W2, b2, costs = nn_model(X, y, n_h, 10000, True)
        plt.subplot(2, len(hidden_layer_sizes), i + 1)
        plt.plot(np.squeeze(costs))
        plt.xlabel('迭代次数', fontproperties=font)
        plt.ylabel('代价函数', fontproperties=font)
        plt.title(('代价函数随迭代次数变化图(hidden_layer_size = %d)' % i), fontproperties=font)
        # 预测
        y_predict = predict(W1, b1, W2, b2, X)
        print('准确率为：', float(np.dot(y_predict, y.T) + np.dot(1 - y_predict, (1 - y).T)) / float(y.size) * 100, '%')

        plt.subplot(2, len(hidden_layer_sizes), len(hidden_layer_sizes) + i + 1)
        plot_decision_boundary(lambda x: predict(W1, b1, W2, b2, x.T), X, y)
        plt.title('浅层神经网络学习效果图(hidden_layer_size = %d)' % i, fontproperties=font)

    plt.show()


