from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from utils import plot_learning_curve
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import SubplotParams
import time

def polynomial_model(degree=1):
    polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)
    linear_regression = LinearRegression(normalize=True)
    pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("linear_regression", linear_regression)])
    return pipeline


def Regression_sin():
    # 生成训练数据
    n_dots = 200
    X = np.linspace(-2 * np.pi, 2 * np.pi, n_dots)
    Y = np.sin(X) + 0.2 * np.random.rand(n_dots) - 0.1
    # 把一个n维向量转换成一个n*1维的矩阵
    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)

    # 分别用2/3/5/10阶多项式来拟合数据集
    degrees = [2, 3, 5, 10]
    results = []
    for d in degrees:
        model = polynomial_model(degree=d)
        model.fit(X, Y)
        train_score = model.score(X, Y)
        mse = mean_squared_error(Y, model.predict(X))
        results.append({"model": model, "degree": d, "score": train_score, "mse": mse})
    for r in results:
        print("degree: {}; train score: {}; mean squared error: {}"
              .format(r["degree"], r["score"], r["mse"]))

    # 绘制不同阶数的多项式的拟合效果
    plt.figure(figsize=(12, 6), dpi=200, subplotpars=SubplotParams(hspace=0.3))
    for i, r in enumerate(results):
        fig = plt.subplot(2, 2, i + 1)
        plt.xlim(-8, 8)
        plt.title("LinearRegression degree={}".format(r["degree"]))
        plt.scatter(X, Y, s=5, c='b', alpha=0.5)
        plt.plot(X, r["model"].predict(X), 'r-')
    plt.show()

    # 绘制10阶模型在[-20,20]的区域内的曲线
    plt.figure(figsize=(12, 6), dpi=200)
    X = np.linspace(-20, 20, 2000).reshape(-1, 1)
    Y = np.sin(X).reshape(-1, 1)
    model_10 = results[3]["model"]
    plt.xlim(-20, 20)
    plt.ylim(-2, 2)
    plt.plot(X, Y, 'b-')
    plt.plot(X, model_10.predict(X), 'r-')
    dot1 = [-2 * np.pi, 0]
    dot2 = [2 * np.pi, 0]
    plt.scatter(dot1[0], dot1[1], s=50, c='r')
    plt.scatter(dot2[0], dot2[1], s=50, c='r')
    plt.show()


def Regression_boston():
    boston = load_boston()
    X = boston.data
    y = boston.target
    print(X.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

    # model = LinearRegression()
    model = polynomial_model(degree=3)
    start = time.process_time()
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print("elaspe:{0:.6f};train_score:{1:0.6f};test_score:{2:.6f}"
          .format(time.process_time() - start, train_score, test_score))

    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    plt.figure(figsize=(18, 4), dpi=200)
    title = 'Learning Curves (degree={0})'
    degrees = [1, 2, 3]
    start = time.process_time()
    for i in range(len(degrees)):
        plt.subplot(1, 3, i + 1)
        plot_learning_curve(plt, polynomial_model(degrees[i]), title.format(degrees[i]),
                            X, y, ylim=(0.01, 1.01), cv=cv)
        print('elaspe:{0:.6f}'.format(time.process_time() - start))
    plt.show()


if __name__ == '__main__':
    # Regression_sin()
    Regression_boston()
