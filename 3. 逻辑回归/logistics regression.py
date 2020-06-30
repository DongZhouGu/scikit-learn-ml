from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from utils import plot_learning_curve
from sklearn.model_selection import ShuffleSplit
from matplotlib import pyplot as plt
import numpy as np
import time

def polynomial_model(degree=1, **kwarg):
    polynomial_features = PolynomialFeatures(degree=degree,
                                             include_bias=False)
    logistic_regression = LogisticRegression(**kwarg)
    pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("logistic_regression", logistic_regression)])
    return pipeline
def cancer_detection():
    cancer = load_breast_cancer()
    X = cancer.data
    y = cancer.target
    print('data shape: {0}; no. positive: {1}; no. negative: {2}'
          .format(X.shape, y[y == 1].shape[0], y[y == 0].shape[0]))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    '''
    model = LogisticRegression()
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print('train score: {train_score:.6f}; test_score:{test_score:.6f}'
          .format(train_score=train_score,
                  test_score=test_score))
    y_pred = model.predict(X_test)
    print('matchs: {0}/{1}'.format(np.equal(y_pred, y_test).shape[0], y_test.shape[0]))
    # 预测概率：找出预测概率低于 90% 的样本
    y_pred_proba = model.predict_proba(X_test)  # 计算每个测试样本的预测概率
    # 找出第一列，即预测为阴性的概率大于 0.1 的样本，保存在 result 里
    y_pred_proba_0 = y_pred_proba[:, 0] > 0.1
    result = y_pred_proba[y_pred_proba_0]
    # 在 result 结果集里，找出第二列，即预测为阳性的概率大于 0.1 的样本
    y_pred_proba_1 = result[:, 1] > 0.1
    print(result[y_pred_proba_1])
    '''
    model = polynomial_model(degree=2, penalty='l1', solver='liblinear')

    start = time.process_time()
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print('elaspe: {0:.6f}; train_score: {1:0.6f}; cv_score: {2:.6f}'.format(
        time.process_time() - start, train_score, test_score))
    # logistic_regression = model.named_steps['logistic_regression']
    # print('model parameters shape: {0}; count of non-zero element: {1}'.format(
    #     logistic_regression.coef_.shape,
    #     np.count_nonzero(logistic_regression.coef_)))

    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    title = 'Learning Curves (degree={0}, penalty={1})'
    degrees = [1, 2]
    penalty = 'l2'

    start = time.process_time()
    plt.figure(figsize=(12, 4), dpi=144)
    for i in range(len(degrees)):
        plt.subplot(1, len(degrees), i + 1)
        plot_learning_curve(plt, polynomial_model(degree=degrees[i], penalty=penalty, solver='liblinear', max_iter=300),
                            title.format(degrees[i], penalty), X, y, ylim=(0.8, 1.01), cv=cv)
    plt.show()
    print('l2_elaspe: {0:.6f}'.format(time.process_time() - start))


if __name__ == '__main__':
    cancer_detection()
