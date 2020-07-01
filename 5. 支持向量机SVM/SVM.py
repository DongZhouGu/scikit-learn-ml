import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from utils import plot_param_curve, plot_learning_curve
import time


def plot_hyperplane(clf, X, y, h=0.02, draw_sv=True, title='hyperplan'):
    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    plt.title(title)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap='hot', alpha=0.5)

    markers = ['o', 's', '^']
    colors = ['b', 'r', 'c']
    labels = np.unique(y)
    for label in labels:
        plt.scatter(X[y == label][:, 0],
                    X[y == label][:, 1],
                    c=colors[label],
                    marker=markers[label])
    if draw_sv:
        sv = clf.support_vectors_
        plt.scatter(sv[:, 0], sv[:, 1], c='y', marker='x')


def SVM_1():
    X, y = make_blobs(n_samples=100, centers=2,
                      random_state=0, cluster_std=0.3)
    clf = svm.SVC(C=1.0, kernel='linear')
    clf.fit(X, y)

    plt.figure(figsize=(12, 4), dpi=144)
    plot_hyperplane(clf, X, y, h=0.01,
                    title='Maximum Margin Hyperplan')
    plt.show()


def SVM_2():
    X, y = make_blobs(n_samples=100, centers=3,
                      random_state=0, cluster_std=0.8)
    clf_linear = svm.SVC(C=1.0, kernel='linear')
    clf_poly = svm.SVC(C=1.0, kernel='poly', degree=3)
    clf_rbf = svm.SVC(C=1.0, kernel='rbf', gamma=0.5)
    clf_rbf2 = svm.SVC(C=1.0, kernel='rbf', gamma=0.1)

    plt.figure(figsize=(10, 10), dpi=144)

    clfs = [clf_linear, clf_poly, clf_rbf, clf_rbf2]
    titles = ['Linear Kernel',
              'Polynomial Kernel with Degree=3',
              'Gaussian Kernel with $\gamma=0.5$',
              'Gaussian Kernel with $\gamma=0.1$']
    for clf, i in zip(clfs, range(len(clfs))):
        clf.fit(X, y)
        plt.subplot(2, 2, i + 1)
        plot_hyperplane(clf, X, y, title=titles[i])
    plt.show()


def SVM_cancer_rbf(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # clf = SVC(C=1.0, kernel='rbf', gamma=0.1)
    # clf.fit(X_train, y_train)
    # train_score = clf.score(X_train, y_train)
    # test_score = clf.score(X_test, y_test)
    # print('train score: {0}; test score: {1}'.format(train_score, test_score))

    # 自动选择最优参数
    gammas = np.linspace(0, 0.0003, 30)
    param_grid = {'gamma': gammas}
    clf = GridSearchCV(SVC(), param_grid, cv=5, return_train_score=True)
    clf.fit(X, y)
    print("best param: {0}\nbest score: {1}".format(clf.best_params_, clf.best_score_))
    plt.figure(figsize=(10, 4), dpi=144)
    plot_param_curve(plt, gammas, clf.cv_results_, xlabel='gamma')
    plt.show()

    # 在gamma为0.01时，画出学习曲线
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    title = 'Learning Curves for Gaussian Kernel'
    start = time.process_time()
    plt.figure(figsize=(10, 4), dpi=144)
    plot_learning_curve(plt, SVC(C=1.0, kernel='rbf', gamma=0.01), title, X, y, ylim=(0.5, 1.01), cv=cv)
    print('elaspe: {0:.6f}'.format(time.process_time() - start))
    plt.show()


def SVM_cancer_poly(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = SVC(C=1.0, kernel='poly', degree=2)
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    print('train score: {0}; test score: {1}'.format(train_score, test_score))

    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    title = 'Learning Curves with degree={0}'
    degrees = [1, 2]
    start = time.process_time()
    plt.figure(figsize=(12, 4), dpi=144)
    for i in range(len(degrees)):
        plt.subplot(1, len(degrees), i + 1)
        plot_learning_curve(plt, SVC(C=1.0, kernel='poly', degree=degrees[i]),
                            title.format(degrees[i]), X, y, ylim=(0.8, 1.01), cv=cv, n_jobs=4)
    plt.show()
    print('elaspe: {0:.6f}'.format(time.process_time() - start))


if __name__ == '__main__':
    # SVM_1()
    # SVM_2()
    cancer = load_breast_cancer()
    X = cancer.data
    y = cancer.target
    print('data shape: {0}; no. positive: {1}; no. negative: {2}'.format(
        X.shape, y[y == 1].shape[0], y[y == 0].shape[0]))
    # SVM_cancer_rbf(X,y)
    SVM_cancer_poly(X, y)
