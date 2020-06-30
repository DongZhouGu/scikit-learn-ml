import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import export_graphviz
import numpy as np
import matplotlib.pyplot as plt


def load_dataset(filename):
    # 指定第一列作为行索引
    data = pd.read_csv(filename, index_col=0)
    # 丢弃无用的数据
    data.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
    # 处理性别数据
    data['Sex'] = (data['Sex'] == 'male').astype('int')
    # 处理登船港口数据
    labels = data['Embarked'].unique().tolist()
    data['Embarked'] = data['Embarked'].apply(lambda n: labels.index(n))
    # 处理缺失数据
    data = data.fillna(0)
    return data


# 参数选择 max_depth
def cv_score1(d, X_train, y_train, X_test, y_test):
    clf = DecisionTreeClassifier(max_depth=d)
    clf.fit(X_train, y_train)
    tr_score = clf.score(X_train, y_train)
    cv_score = clf.score(X_test, y_test)
    return (tr_score, cv_score)


# 训练模型，并计算评分
def cv_score2(val, X_train, y_train, X_test, y_test):
    # clf = DecisionTreeClassifier(criterion='gini', min_impurity_decrease=val)
    clf = DecisionTreeClassifier(criterion='entropy', min_impurity_decrease=val)
    clf.fit(X_train, y_train)
    tr_score = clf.score(X_train, y_train)
    cv_score = clf.score(X_test, y_test)
    return (tr_score, cv_score)


def choose_depths(X_train, y_train, X_test, y_test):
    # 构造参数范围，在这个范围内分别计算模型评分
    depths = range(2, 15)
    scores = [cv_score1(d, X_train, y_train, X_test, y_test) for d in depths]
    tr_scores = [s[0] for s in scores]
    cv_scores = [s[1] for s in scores]
    best_score_index = np.argmax(cv_scores)
    best_score = cv_scores[best_score_index]
    best_param = depths[best_score_index]
    print('best param: {0}； best score： {1}'.format(best_param, best_score))

    # 把模型参数和对应的模型评分画出来
    plt.figure(figsize=(6, 4), dpi=144)
    plt.grid()
    plt.xlabel('max depth of decision tree')
    plt.ylabel('score')
    plt.plot(depths, cv_scores, '.g-', label='cross-validation score')
    plt.plot(depths, tr_scores, '.r--', label='training score')
    plt.legend()
    plt.show()


def choose_min_impurity_split(X_train, y_train, X_test, y_test):
    # 指定参数范围，分别训练模型，并计算评分
    values = np.linspace(0, 0.005, 50)
    scores = [cv_score2(v, X_train, y_train, X_test, y_test) for v in values]
    tr_scores = [s[0] for s in scores]
    cv_scores = [s[1] for s in scores]

    # 找出评分最高的模型参数
    best_score_index = np.argmax(cv_scores)
    best_score = cv_scores[best_score_index]
    best_param = values[best_score_index]
    print('best param: {0}; best score: {1}'.format(best_param, best_score))

    # 画出模型参数与模型评分的关系
    plt.figure(figsize=(10, 6), dpi=144)
    plt.grid()
    plt.xlabel('threshold of entropy')
    plt.ylabel('score')
    plt.plot(values, cv_scores, '.g-', label='cross-validation score')
    plt.plot(values, tr_scores, '.r--', label='training score')
    plt.legend()
    plt.show()


def plot_curve(train_sizes, cv_results, xlabel):
    train_scores_mean = cv_results['mean_train_score']
    train_scores_std = cv_results['std_train_score']
    test_scores_mean = cv_results['mean_test_score']
    test_scores_std = cv_results['std_test_score']
    plt.figure(figsize=(6, 4), dpi=144)
    plt.title('parameters turning')
    plt.grid()
    plt.xlabel(xlabel)
    plt.ylabel('score')
    plt.fill_between(train_sizes,
                     train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std,
                     alpha=0.1, color='r')
    plt.fill_between(train_sizes,
                     test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std,
                     alpha=0.1, color='g')
    plt.plot(train_sizes, train_scores_mean, '.--', color='r', label='Training score')
    plt.plot(train_sizes, test_scores_mean, '.-', color='g', label='Cross-validation score')
    plt.legend(loc='best')
    plt.show()


def DecisionTree():
    # 读取并划分数据集
    train = load_dataset('./titanic/train.csv')
    y = train['Survived'].values
    X = train.drop(['Survived'], axis=1).values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # print('train dataset: {0}; test dataset: {1}'.format(X_train.shape,X_test.shape))

    # 模型训练
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    print('train score: {0}; test score: {1}'.format(train_score, test_score))

    # choose_depths(X_train,y_train,X_test,y_test)
    # choose_min_impurity_split(X_train,y_train,X_test,y_test)

    # 模型参数选择工具包
    entropy_thresholds = np.linspace(0, 1, 50)
    gini_thresholds = np.linspace(0, 0.5, 50)
    param_grid = [{'criterion': ['entropy'], 'min_impurity_split': entropy_thresholds},
                  {'criterion': ['gini'], 'min_impurity_split': gini_thresholds},
                  {'max_depth': range(2, 10)},
                  {'min_samples_split': range(2, 30, 2)}]
    clf = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
    clf.fit(X, y)
    print('best param: {0}\nbest score: {1}'.format(clf.best_params_, clf.best_score_))

    # 生成决策树示意图
    clf = DecisionTreeClassifier(criterion='entropy', min_impurity_split=0.5306122448979591)
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    # 生成决策树示意图
    columns = train.columns[1:]
    # 导出 titanic.dot 文件
    with open("./titanic.dot", 'w') as f:
        f = export_graphviz(clf, out_file=f,feature_names=columns)


if __name__ == '__main__':
    DecisionTree()
