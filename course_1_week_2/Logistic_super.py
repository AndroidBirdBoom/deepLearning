import numpy as np
import matplotlib.pyplot as plt
import h5py
from matplotlib.font_manager import FontProperties

font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)  # 解决windows环境下画图汉字乱码问题

'''
判断是否是猫的算法（逻辑回归）
1. 加载数据，展平数据
2. 初始化w,b
3. 通过前向传播算法获取h,反向传播算法更新w,b
4. 结果测试
'''


# 加载数据
def load_dataset():
    train_data = h5py.File('datasets/train_catvnoncat.h5', 'r')
    train_x = np.array(train_data['train_set_x'])
    train_y = np.array(train_data['train_set_y'])
    test_data = h5py.File('datasets/test_catvnoncat.h5', 'r')
    test_x = np.array(test_data['test_set_x'])
    test_y = np.array(test_data['test_set_y'])
    classes = np.array(test_data['list_classes'])
    return train_x, train_y, test_x, test_y, classes


def init_data(train_set_x_orig, test_set_x_orig):
    # 展平数据
    train_set_x_flatten = train_set_x_orig.reshape((train_set_x_orig.shape[0], -1)).T
    test_set_x_flatten = test_set_x_orig.reshape((test_set_x_orig.shape[0], -1)).T

    # 标准化数据
    train_set_x = train_set_x_flatten / 255
    test_set_x = test_set_x_flatten / 255
    return train_set_x, test_set_x


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = .0
    return w, b


# 前向、后向传播
def propagate(w, b, X, Y):
    m = Y.shape[0]
    # 正向传播
    z = np.dot(w.T, X) + b
    a = sigmoid(z)
    J = np.mean(-Y * np.log(a) - (1 - Y) * np.log(1 - a))

    # 反向传播
    dz = a - Y
    dw = np.dot(X, dz.T) / m
    db = np.sum(dz) / m
    return dw, db, J


# 开始梯度下降算法
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    costs = []
    for i in range(num_iterations):
        dw, db, J = propagate(w, b, X, Y)
        w -= learning_rate * dw
        b -= learning_rate * db

        if not np.isnan(J) and i % 10 == 0:
            costs.append(J)

    if print_cost and i % 100 == 0:
        print('迭代次数：%i，误差值：%f' % (i, J))


    return w, b, costs


# 预测值
def predict(w, b, X):
    Y_prediction = np.dot(w.T, X) + b
    true_values = np.where(Y_prediction > 0.5)
    false_values = np.where(Y_prediction <= 0.5)
    Y_prediction[true_values] = 1
    Y_prediction[false_values] = 0
    return Y_prediction


# 预测总方法
def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    w, b = initialize_with_zeros(train_set_x.shape[0])
    wb, db, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    Y_prediction_test = predict(w, b, X_test)

    print('训练集准确率:', (1 - (np.mean(np.abs(Y_prediction_test - Y_test)))) * 100, '%', "学习率 = ", learning_rate)
    return wb, db, costs


if __name__ == "__main__":
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

    # # 展示图片
    # plt.imshow(train_set_x_orig[25])
    # plt.show()

    train_set_x, test_set_x = init_data(train_set_x_orig, test_set_x_orig)

    learning_rates = [0.01, 0.005, 0.001, 0.0005, 0.0001]
    for i in learning_rates:
        w, b, costs = model(train_set_x, train_set_y, test_set_x, test_set_y, 2000, i, True)
        plt.plot(np.squeeze(costs), label=str(i))

    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.title('不同学习率下的代价函数随迭代次数的变化', fontproperties=font)

    legend = plt.legend(loc='upper center', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')

    plt.show()
