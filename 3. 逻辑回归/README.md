# Logistic 回归算法



## 1. Logistic 回归概述

Logistic 回归 或者叫逻辑回归，虽然名字有回归，但是它是用来做分类的。其主要思想是: 根据现有数据对分类边界线(Decision Boundary)建立回归公式，以此进行分类。



## 2. 算法原理

假设有一场足球赛，我们有两支球队的所有出场球员信息、历史交锋成绩、比赛时间、主客场、裁判和天气等信息，根据这些信息预测球队的输赢。假设比赛结果记为y，赢球标记为1，输球标记为0，这就是典型的二元分类问题，可以用逻辑回归算法来解决。

与线性回归算法的最大区别是，逻辑回归算法的输出是个离散值。

### 2.1 预测函数

需要找出一个预测函数模型，使其值输出在[0,1]之间。然后选择一个基准值，如0.5，如果算出来的预测值大于0.5，就认为其预测值为1，反之，则其预测值为0。

选择Sigmoid函数（也称为Logistic函数，逻辑回归的名字由此而来）
$$
g(z)=\frac{1}{1+e^{-z}}
$$
来作为预测函数，其中e是自然对数的底数。以z为横坐标，以g(z)为纵坐标，画出的图形如下所示：

![Sigmoid 函数在不同坐标下的图片](https://cdn.jsdelivr.net/gh/dongzhougu/imageuse1/LR_3.png)

从图中可以看出，当z=0时，g(z)=0.5；当z>0时，g(z)>0.5，当z越来越大时，g(z)无限接近于1；当z<0时，g(z)<0.5，当z越来越小时，g(z)无限接近于0。这正是我们想要的针对二元分类算法的预测函数。

### 2.2 判定边界

逻辑回归算法的预测函数由下面两个公式给出：
$$
h_{\theta}(x)=g\left(\theta^{T} x\right)
$$

$$
g(z)=\frac{1}{1+e^{-z}}
$$


下面给出两个判定边界的例子。假设有两个变量x1，x2，其逻辑回归预测函数是$h_{\theta}(x)=g\left(\theta_{0}+\theta_{1} x_{1}+\theta_{2} x_{2}\right)$

假设给定参数：
$$
\theta=\left[\begin{array}{c}
-3 \\
1 \\
1
\end{array}\right]
$$
那么，可以得到判定边$-3+x_{1}+x_{2}=0$ ，如果以 $x_{1}$ 为横坐标， $x_{2}$  为纵坐标，则这个函数画出来就是一条通过(0,3)和(3,0)两点的直线。这条线就是判定边界，其中，直线左下方为y=0，直线右上方为y=1，如图所示：

![img](https://cdn.jsdelivr.net/gh/dongzhougu/imageuse1/17634123-0bf913a36c2847a8.png)

如果预测函数是多项式 $h_{\theta}(x)=g\left(\theta_{0}+\theta_{1} x_{1}+\theta_{2} x_{2}+\theta_{3} x_{1}^{2}+\theta_{4} x_{2}^{2}\right)$，且给定


$$
\theta=\left[\begin{array}{c}
-1 \\
0 \\
0 \\
1 \\
1
\end{array}\right]
$$
则可以得到判定边界函数$x_{1}^{2}+x_{2}^{2}=1$ 则这是一个半径为1的圆。圆内部是y=0，圆外部是y=1，如上图所示。

### 2.3 损失函数

我们不能使用线性回归模型的损失函数来推导逻辑回归的损失函数，因为那样的损失函数太复杂，最终很可能会导致无法通过迭代找到损失函数值最小的点。

为了容易地求出损失函数的最小值，我们分成 y=1 和 y=0 两种情况来分别考虑其预测值和真实值的误差。我们先考虑最简单的情况，即计算某个样本 x，y=1 和 y=0 两种情况下的预测值与真实值的误差，我们选择的损失公式如下：
$$
\operatorname{cost}\left(h_{\theta}(x), y\right)=\left\{\begin{array}{ccc}
-\log \left(h_{\theta}(x)\right), & \text { if } & y=1 \\
-\log \left(1-h_{\theta}(x)\right), & \text { if } & y=0
\end{array}\right.
$$
其中， $h_{\theta}(x)$ 表示预测为1的概率，log(x)为自然对数。如图所示

![img](https://cdn.jsdelivr.net/gh/dongzhougu/imageuse1/image-20200630141514106.png)

根据损失函数的定义，损失是预测值与真实值的差异。当差异越大时，损失越大，模型受到的“惩罚”也越严重。在左图中，当 y=1 时，随着（预测为1的概率）越来越大，预测值越来越接近真实值，其损失越来越小；在右图中，当 y=0 时，随着（预测为1的概率）越来越大，预测值越来越偏离真实值，其损失越来越大。

### 2.4 梯度下降算法

和线性回归类似，这里使用梯度下降算法来求解逻辑回归模型参数。具体可见上一节 [线性回归回归算法](../2. 线性回归/README.md)



## 3. 多元分类

逻辑回归模型可以解决二元分类问题，即 y={0,1}，能不能解决多元分类问题呢？答案是肯定的。针对多元分类问题，y={0,1,2,3,...,n}，总共有n+1个类别。其解决思路是：首先把问题转换为二元分类问题，即y=0是一个类别，y={1,2,3,...,n}作为另外一个类别，然后计算这两个类别的概率；接着，把y=1作为一个类别，把y={0,2,3,...,n}作为另外一个类别，再计算这两个类别的概率。



## 4. 正则化

我们知道，过拟合是指模型很好地拟合了训练样本，但对新数据预测的准确性很差，这是因为模型太复杂了。解决办法是减少输入特征的个数，或者获取更多的训练样本。这里介绍的正则化也可以用来解决过拟合问题：

- 保留所有的特征，减少特征的权重 $\theta_{j} $ 的值。确保所有的特征对预测值都有少量的贡献。

- 当每个特征 $x_{j} $ 对预测值y都有少量的贡献时，这样的模型可以良好的工作，这正是正则化的目的，可以用它来解决特征过多时的过拟合问题。


### 4.1 线性回归模型正则化

我们先来看线性回归模型的损失函数是如何正则化的：
$$
J(\theta)=\frac{1}{2 m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right)^{2}+\lambda \sum_{j=1}^{n} \theta_{j}^{2}
$$
公式中前半部分就是原来的线性回归模型的损失函数，也称为预测值与实际值的误差。后半部分为加入的正则项。其中 $\lambda $ 的值有两个目的，即要维持对训练样本的拟合，又要避免对训练样本的过拟合。如果  $\lambda $  的值太大，则能确保不出现过拟合，但可能会导致对现有训练样本出现欠拟合。

### 4.2 线性回归模型正则化

同样，可以对逻辑回归模型的损失函数进行正则化，其方法也是在原来的损失函数的基础上加上正则项：
$$
J(\theta)=-\frac{1}{m}\left[\sum_{i=1}^{m} y^{(i)} \log \left(h_{\theta}\left(x^{(i)}\right)\right)+\left(1-y^{(i)}\right) \log \left(1-h_{\theta}\left(x^{(i)}\right)\right)\right]+\frac{\lambda}{2 m} \sum_{j=1}^{n} \theta_{j}^{2}
$$


## 5. 算法参数

在 `scikit-learn `里，逻辑回归模型由类 `sklearn.linear_model.LogisticRegression `实现。

### 5.1 正则项权重

上面介绍的正则项权重  $\lambda $  ，在`LogisticRegression`类里有个参数 C 与之对应，但成反比。即 C 值越大，  $\lambda $ 越小，模型容易出现过拟合；C 值越小，  $\lambda $  越大，模型容易出现欠拟合。

### 5.2 L1/L2范数

创建逻辑回归模型时，有个参数penalty（惩罚），其取值有“l1”或“l2”

- L1范数作为正则项，会让模型参数 $\theta$ 稀疏化，即让模型参数向量里的0元素尽可能多，只保留模型参数向量中重要特征的贡献。
- L2范数作为正则项，则让模型参数尽量小，但不会为0，即尽量让每个特征对应预测值都有一些小的贡献。

假设模型只有两个参数，它们构成一个二维向量 $\theta=\left[\theta_{1}, \theta_{2}\right]$,则L1范数为：
$$
\|\theta\|_{1}=\left|\theta_{1}\right|+\left|\theta_{2}\right|
$$
即L1范数是向量里元素的绝对值之和。L2范数为向量里所有元素的平方和的算术平方根：
$$
\|\theta\|_{2}=\sqrt{\theta_{1}^{2}+\theta_{2}^{2}}
$$
我们知道，梯度下降算法在参数迭代的过程中，实际上是在损失函数的等高线上跳跃，并最终收敛在误差最小的点上。那么正则项的本质是什么？正则项的本质是惩罚。在参数迭代的过程中，如果没有遵循正则项所表达的规则，那么其损失会变大，即受到了惩罚，从而往正则项所表达的规则处收敛。正则化后的模型参数应该收敛在误差等高线与正则项等高线相切的点上。

作为推论，L1范数作为正则项，有以下几个用途：

- 选择重要特征：L1范数会让模型参数向量里的元素为0的点尽量多，这样可以排除掉那些对预测值没有什么影响的特征，从而简化问题。所以L1范数解决过拟合，实际上是减少特征数量。
- 模型可解释性好：模型参数向量稀疏化后，只会留下那些对预测值有重要影响的特征。这样我们就容易解释模型的因果关系。比如，针对某种癌症的筛查，如果有100个特征，那么我们无从解释到底哪些特征对阳性呈关键作用。稀疏化后，只留下几个关键的特征，就容易看到因果关系。

由此可见，L1范数作为正则项，更多的是一个分析工具，而适合用来对模型求解。因为它会把不重要的特征直接去除。大部分的情况下解决过拟合问题，还是选择L2范数作为正则项，这也是 `scikit-learn` 里的默认值。



## 6 示例：乳腺癌检测

本节来看一个实例，使用逻辑回归算法解决乳腺癌检测问题。我们需要先采集肿瘤病灶造影图片，然后对图片进行分析，从图片中提取特征，再根据特征来训练模型。最终使用模型来检测新采集到的肿瘤病灶造影，以便判断肿瘤是良性的还是恶性的。这是个典型的二元分类问题。

### 6.1 数据采集及特征提取

为了简单起见，直接加载 `scikit-learn` 自带的一个乳腺癌数据集。这个数据集是已经采集后的数据：

```python
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target
print('data shape: {0}; no. positive: {1}; no. negative: {2}'
      .format(X.shape,y[y==1].shape[0],y[y==0].shape[0]))
print(cancer.data[0])
```

输出如下：

```python
data shape: (569, 30); no. positive: 357; no. negative: 212
[1.799e+01 1.038e+01 1.228e+02 1.001e+03 1.184e-01 2.776e-01 3.001e-01
 1.471e-01 2.419e-01 7.871e-02 1.095e+00 9.053e-01 8.589e+00 1.534e+02
 6.399e-03 4.904e-02 5.373e-02 1.587e-02 3.003e-02 6.193e-03 2.538e+01
 1.733e+01 1.846e+02 2.019e+03 1.622e-01 6.656e-01 7.119e-01 2.654e-01
 4.601e-01 1.189e-01]
```

数据集中总共有569个样本，每个样本有30个特征，其中357个阳性（y=1）样本，212个阴性（y=0）样本。同时，还打印出一个样本数据，以便直观地进行观察。

这30个特征是怎么来的呢？这个数据集总共从病灶造影图片中提取了以下10个关键属性：

- radius：半径，即病灶中心点离边界的平均距离。
- texture：纹理，灰度值的标准偏差。
- perimeter：周长，即病灶的大小。
- area：面积，也是反映病灶大小的一个指标。
- smoothness：平滑度，即半径的变化幅度。
- compactness：密实度，周长的平方除以面积，再减去1
- concavity：凹度，凹陷部分轮廓的严重程度。
- concave points：凹点，凹陷轮廓的数量。
- symmetry：对称性。
- fractal demension：分形维度。

实际上它只关注10个特征，然后又构造出了每个特征的标准差及最大值，这样每个特征就衍生出了两个特征，所以总共就有了30个特征。可以通过 `cancer.feature_names` 变量来查看这些特征的名称。

### 6.2 模型训练

首先，把数据集分成训练数据集和测试数据集：

```python
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
```

然后使用 `LogisticRegression` 模型来训练，并计算训练数据集的评分数据和测试数据集的评分数据：

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,y_train)
train_score = model.score(X_train,y_train)
test_score = model.score(X_test,y_test)
print('train score: {train_score:.6f}; test_score:{test_score:.6f}'
      .format(train_score=train_score,
             test_score=test_score))
```

输出如下：

```css
train score: 0.940659; test_score:0.964912
```

观察模型在测试样本集的表现：

```python
import numpy as np
y_pred = model.predict(X_test)
print('matchs: {0}/{1}'.format(np.equal(y_pred,y_test).shape[0],y_test.shape[0]))
```

输出如下：

```python
matchs: 114/114
```

总共114个测试样本，全部预测正确。为什么 `testscore` 却只有0.973684，而不是1呢？答案是，`scikit-learn`不是使用这个数据来计算分数，因为这个数据不能完全反映误差情况，而是使用预测概率数据计算模型评分。

针对二元分类问题，`LogisticRegression `模型会对每个样本输出两个概率，即为 0 的概率和为 1 的概率，哪个概率高就预测为哪个类别。

找出测试数据集中预测“自信度”低于90%的样本。这里先计算出测试数据集里的每个样本的预测概率数据，针对每个样本，它会有两个数据，一是预测其为阳性的概率，另外一个是预测其为阴性的概率。接着找出预测为阴性的概率大于0.1且小于0.9的样本（同时也是预测为阳性的概率大于0.1小于0.9），这些样本就是“自信度”不足90%的样本。

```python
# 预测概率：找出预测概率低于 90% 的样本
y_pred_proba = model.predict_proba(X_test)  # 计算每个测试样本的预测概率
# 找出第一列，即预测为阴性的概率大于 0.1 的样本，保存在 result 里
y_pred_proba_0 = y_pred_proba[:, 0] > 0.1
result = y_pred_proba[y_pred_proba_0]
# 在 result 结果集里，找出第二列，即预测为阳性的概率大于 0.1 的样本
y_pred_proba_1 = result[:, 1] > 0.1
print(result[y_pred_proba_1])
```

输出如下：

```python
[[0.29623162 0.70376838]
 [0.54660262 0.45339738]
 [0.17874247 0.82125753]
 [0.20917573 0.79082427]
 [0.10943452 0.89056548]
 [0.35503614 0.64496386]
 [0.23849987 0.76150013]
 [0.13634228 0.86365772]
 [0.80171734 0.19828266]
 [0.21744759 0.78255241]
 [0.81346356 0.18653644]
 [0.2225791  0.7774209 ]
 [0.10788007 0.89211993]
 [0.88068005 0.11931995]
 [0.18189724 0.81810276]]
```

由此可见，计算预测概率使用model.predict_proba()函数，而计算预测分类用model.predict()函数。

### 6.3 模型优化

首先，使用Pipeline来增加多项式特征：

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# 增加多项式预处理
def polynomial_model(degree=1, **kwarg):
    polynomial_features = PolynomialFeatures(degree=degree,
                                             include_bias=False)
    logistic_regression = LogisticRegression(**kwarg)
    pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("logistic_regression", logistic_regression)])
    return pipeline
```

接着，增加二阶多项式特征，创建并训练模型：

```python
import time
model = polynomial_model(degree=2, penalty='l1', solver='liblinear')
start = time.process_time()
model.fit(X_train, y_train)
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print('elaspe: {0:.6f}; train_score: {1:0.6f}; cv_score: {2:.6f}'.format(
    time.process_time() - start, train_score, test_score))
```

使用L1范数作为正则项（参数penalty='l1'），输出如下：

```python
elaspe: 0.156250; train_score: 1.000000; cv_score: 0.956140
```

可以看到，训练数据集评分和测试数据集评分都增加了。为什么使用L1范数作为正则项呢？L1范数作为正则项可以实现参数的稀疏化，即自动选择出那些对模型有关联的重要特征。

```python
logistic_regression = model.named_steps['logistic_regression']
print('model parameters shape: {0}; count of non-zero element: {1}'.format(
    logistic_regression.coef_.shape, 
    np.count_nonzero(logistic_regression.coef_)))
```

输出如下：

```python
model parameters shape: (1, 495); count of non-zero element: 110
```

逻辑回归模型的coef_属性里保存的就是模型参数。从输出结果可以看到，增加二阶多项式特征后，输入特征由原来的30个增加到了495个，最终大多数特征都被丢弃，只保留了110个有效特征。

### 6.4 学习曲线

首先画出使用L1范数作为正则项所对应的一阶和二阶多项式的学习曲线：

```python
from utils import plot_learning_curve
from sklearn.model_selection import ShuffleSplit
from matplotlib import pyplot as plt

cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
title = 'Learning Curves (degree={0}, penalty={1})'
degrees = [1, 2]
penalty = 'l1'

start = time.process_time()
plt.figure(figsize=(12, 4), dpi=144)
for i in range(len(degrees)):
    plt.subplot(1, len(degrees), i + 1)
    plot_learning_curve(plt, polynomial_model(degree=degrees[i], penalty=penalty, solver='liblinear', max_iter=300), 
                        title.format(degrees[i], penalty), X, y, ylim=(0.8, 1.01), cv=cv)

print('elaspe: {0:.6f}'.format(time.process_time()-start))
```

输出的结果如下：

```python
l1_elaspe: 10.781250
```

L1范数学习曲线如下图所示：

![image-20200630141514106](https://cdn.jsdelivr.net/gh/dongzhougu/imageuse1/17634123-08d8944e88c2ae64.png)

接着画出使用L2范数作为正则项所对应的一阶和二阶多项式的学习曲线：

```python
import warnings
warnings.filterwarnings("ignore")

penalty = 'l2'

start = time.clock()
plt.figure(figsize=(12, 4), dpi=144)
for i in range(len(degrees)):
    plt.subplot(1, len(degrees), i + 1)
    plot_learning_curve(plt, polynomial_model(degree=degrees[i], penalty=penalty, solver='lbfgs'), 
                        title.format(degrees[i], penalty), X, y, ylim=(0.8, 1.01), cv=cv)

print('elaspe: {0:.6f}'.format(time.clock()-start))
```

输出的结果如下：

```python
l2_elaspe: 2.718750
```

L2范数学习曲线如下图所示：

![image-20200630141743010](https://cdn.jsdelivr.net/gh/dongzhougu/imageuse1/image-20200630141743010.png)

可以明显地看出，使用二阶多项式并使用L1范数作为正则项的模型最优，因为它的训练样本评分最高，交叉验证样本评分也最高。从图中还可以看出，训练样本评分和交叉验证样本评分之间的间隙还比较大，我们可以采集更多的数据来训练模型，以便进一步优化模型。

另外从输出的时间可以看出，L1 范数对应的学习曲线，需要花费较长的时间，原因是，`scikit-learn` 的`learning_curve()` 函数在画学习曲线的过程中，要对模型进行多次训练，并计算交叉验证样本评分。同时，为了使曲线更平滑，针对每个点还会进行多次计算求平均值。这个就是 `ShuffleSplit` 类的作用。在我们这个实例里，只有569个训练样本，这是个很小的数据集。如果数据集增加100倍，甚至1000倍，拿出来画学习曲线将是场灾难。

那么，针对大数据集，怎样高效地画学习曲线？答案很简单，可以从大数据集里选择一小部分数据来画学习曲线，待选择好最优的模型之后，再使用全部的数据集来训练模型。但是要尽量保持选择出来的这部分数据的标签分布与大数据集的标签分布相同，如针对二元分类，阳性和阴性比例要一致。更直观的说就是，抽取出来的样本集为原来数据集的一个缩影，尽可能相似。



## 7.拓展阅读

实际上，我们的预测函数就是写成向量形式的：
$$
h_{\theta}(x)=g(z)=g\left(\theta^{T} x\right)=\frac{1}{1+e^{-\theta^{T} x}}
$$
这个预测函数一次只计算一个训练样本的预测值，怎样一次性计算出所有样本的预测值呢？答案是把预测函数的参数写成向量的形式：
$$
h=g(X \theta)
$$
其中g(x)为Sigmoid函数。X为m×n的矩阵，即数据集的矩阵表达。损失函数也有对应的矩阵形式：
$$
J(\theta)=\frac{1}{m}\left(-y^{T} \log (h)-(1-y)^{T} \log (1-h)\right)
$$
其中，y为目标值向量，h为一次性计算出来的所有样本的预测值。



