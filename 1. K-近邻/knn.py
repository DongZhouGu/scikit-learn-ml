from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import numpy as np


def knn_classify():
    # 生成已标记的数据集
    centers = [[-2, 2], [2, 2], [0, 4]]
    X, y = make_blobs(n_samples=60, centers=centers, random_state=0, cluster_std=0.6)

    # 训练算法
    k = 5
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X, y)

    # 对样本进行预测
    X_sample = [0, 2]
    X_sample = np.array(X_sample).reshape(1, -1)  # [[0 2]]
    y_sample = clf.predict(X_sample)
    neighbors = clf.kneighbors(X_sample, return_distance=False)  # [[16 20 48  6 23]]

    # 标记最近的5个点
    c = np.array(centers)
    # cmap就是指matplotlib.colors.Colormap,一个包含三列矩阵的色彩映射表
    # 使用c和cmap来映射颜色，s为形状的大小
    plt.scatter(X[:, 0], X[:, 1], c=y, s=100, cmap='cool')
    plt.scatter(c[:, 0], c[:, 1], s=100, marker='*', c='black')
    plt.scatter(X_sample[0][0], X_sample[0][1], marker="x", s=100, cmap='cool')  # 待预测的点
    for i in neighbors[0]:
        plt.plot([X[i][0], X_sample[0][0]], [X[i][1], X_sample[0][1]], 'k--', linewidth=0.6)
    plt.show()


def knn_regression():
    # 生成已标记的数据集
    n_dots = 40
    # 生成40行1列的服从“0~5”均匀分布的随机样本
    X = 5 * np.random.rand(n_dots, 1)
    y = np.cos(X).flatten()
    # 生成40行1列的服从“-0.1~0.1”均匀分布的随机误差
    y += 0.2 * np.random.rand(n_dots) - 0.1

    # 训练算法
    k = 5
    knn = KNeighborsRegressor(k)
    knn.fit(X, y)
    # print(knn.score(X, y))  #0.9859284704870181

    # 回归拟合
    # 随机生成0~5内500个数，并预测y值，拟合出来确实和cos曲线相似
    T = np.linspace(0, 5, 500)[:, np.newaxis]
    y_pred = knn.predict(T)
    plt.figure(dpi=144)
    plt.scatter(X, y, c='b', label='data', s=100)
    plt.plot(T, y_pred, c='g', label='prediction', lw=4)
    plt.axis('tight')
    plt.title('KNeighborsRegressor (k = %i)' % k)
    plt.show()


if __name__ == '__main__':
    knn_regression()
