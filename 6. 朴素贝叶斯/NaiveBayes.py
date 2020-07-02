from time import time
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def bayes():
    print("loading train dataset ...")
    t = time()
    news_train = load_files("E:/notebook/scikit-learn机器学习：常用算法原理及编程实战/379/train")
    print("summary: {0} documents in {1} categories."
          .format(len(news_train.data), len(news_train.target_names)))
    print("done in {0} seconds".format(time() - t))

    print("vectorizing train dataset ...")
    t = time()
    vectorizer = TfidfVectorizer(encoding='latin-1')
    X_train = vectorizer.fit_transform((d for d in news_train.data))
    print("n_samples: %d, n_features: %d" % X_train.shape)
    print("number of non-zero features in samples [{0}]:{1}"
          .format(news_train.filenames[0], X_train[0].getnnz()))
    print("done in {0} seconds".format(time() - t))

    print("training models ...")
    t = time()
    y_train = news_train.target
    clf = MultinomialNB(alpha=0.0001)
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    print("train score: {0}".format(train_score))
    print("done in {0} seconds".format(time() - t))

    print("loading test dataset ...")
    t = time()
    news_test = load_files("E:/notebook/scikit-learn机器学习：常用算法原理及编程实战/379/test")
    print("summary: {0} documents in {1} categories."
          .format(len(news_test.data), len(news_test.target_names)))
    print("done in {0} seconds".format(time() - t))

    print("vectorizing test dataset ...")
    t = time()
    X_test = vectorizer.transform((d for d in news_test.data))
    y_test = news_test.target
    print("n_samples: %d, n_features: %d" % X_test.shape)
    print("number of non-zero features in sample [{0}]: {1}"
          .format(news_test.filenames[0], X_test[0].getnnz()))
    print("done in {0} seconds".format(time() - t))

    print("predicting test dataset ...")
    t = time()
    pred_test = clf.predict(X_test)
    print("done in %fs" % (time() - t))
    print("classification report on test set for classifier:")
    print(clf)
    print(classification_report(y_test, pred_test, target_names=news_test.target_names))

    # 生成混淆矩阵，观察每种类别被错误分类的情况
    cm = confusion_matrix(y_test, pred_test)
    print("confusion matrix:\n")
    print(cm)

    # 把混淆矩阵进行数据可视化
    plt.figure(figsize=(8, 8), dpi=144)
    plt.title('Confusion matrix of the classifier')
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.matshow(cm, fignum=1, cmap='gray')
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    bayes()
