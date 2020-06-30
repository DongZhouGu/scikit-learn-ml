from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, RadiusNeighborsClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score, ShuffleSplit, learning_curve
from sklearn.feature_selection import SelectKBest
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import plot_learning_curve

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


def knn_diabetes():
    # 加载数据集
    data = pd.read_csv('./pima-indians-diabetes/diabetes.csv')
    # print('dataset shape {}'.format(data.shape))
    # print(data.head())
    # print(data.groupby('Outcome').size())

    # 处理数据集
    X = data.iloc[:, :8]
    Y = data.iloc[:, 8]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    # print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
    '''
    # 模型比较
    models = []
    models.append(("KNN", KNeighborsClassifier(n_neighbors=2)))
    models.append(("KNN with weights", KNeighborsClassifier(
        n_neighbors=2, weights="distance")))
    models.append(("Radius Neighbors", RadiusNeighborsClassifier(
        n_neighbors=2, radius=500.0)))
    results = []
    for name, model in models:
        kfold = KFold(n_splits=10)
        cv_result = cross_val_score(model, X, Y, cv=kfold)
        results.append((name, cv_result))
    for i in range(len(results)):
        print("name: {}; cross val score: {}".format(
            results[i][0], results[i][1].mean()))
    '''
    # 模型训练及分析
    # knn = KNeighborsClassifier(n_neighbors=2)
    # knn.fit(X_train, Y_train)
    # train_score = knn.score(X_train, Y_train)
    # test_score = knn.score(X_test, Y_test)
    # print("train score: {}\ntest score: {}".format(train_score, test_score))
    knn = KNeighborsClassifier(n_neighbors=2)
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    plt.figure(figsize=(10, 6))
    plot_learning_curve(plt, knn, "Learn Curve for KNN Diabetes",
                        X, Y, ylim=(0.0, 1.01), cv=cv)
    plt.show()

    # 特征选择及数据可视化
    selector = SelectKBest(k=2)
    X_new = selector.fit_transform(X, Y)
    print('X_new.shape {}'.format(X_new.shape))
    plt.figure(figsize=(10, 6))
    plt.ylabel("BMI")
    plt.xlabel("Glucose")
    plt.scatter(X_new[Y==0][:, 0], X_new[Y==0][:, 1], c='r', s=20, marker='o');   #画出样本
    plt.scatter(X_new[Y==1][:, 0], X_new[Y==1][:, 1], c='g', s=20, marker='^');   #画出样本
    plt.show()



if __name__ == '__main__':
    # knn_classify()
    # knn_regression()
    knn_diabetes()
