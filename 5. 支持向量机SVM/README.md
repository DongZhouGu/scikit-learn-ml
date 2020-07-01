# SVM支持向量机

## 1. SVM概述

支持向量机(Support Vector Machines, SVM): 是一种监督学习算法。在工业界和学术界都有广泛的应用。特别是针对数据集较小的情况下，往往其分类效果比神经网络好。

* 支持向量(Support Vector) 就是离分隔超平面最近的那些点。
* 机(Machine) 就是表示一种算法，而不是表示机器。

## 2. SVM原理

SVM的原理就是使用分隔超平面来划分数据集，并使得支持向量（数据集中离分隔超平面最近的点）到该分隔超平面的距离最大。其最大特点是能构造出最大间距的决策边界，从而提高分类算法的鲁棒性。

要给左右两边的点进行分类，明显发现: 选择D会比B、C分隔的效果要好很多。

![线性可分](https://cdn.jsdelivr.net/gh/dongzhougu/imageuse1/SVM_3_linearly-separable.jpg)

### 2.1 寻找最大间隔

#### 点到超平面的距离

* 分隔超平面`函数间距`:  $y(x)=w^Tx+b$

* 分类的结果:  $f(x)=sign(w^Tx+b)$   (sign表示>0为1，<0为-1，=0为0) 

* 点到超平面的`几何间距`: $d(x)=(w^Tx+b)/||w||$  （$||w||$表示w矩阵的二范数=> $\sqrt{w^T*w}$, 点到超平面的距离也是类似的）

$$
d=\left|\frac{A x_{0}+B y_{0}+C}{\sqrt{A^{2}+B^{2}}}\right|
$$



#### 拉格朗日乘子法

* 类别标签用-1、1，是为了后期方便 $ label*(w^Tx+b)$ 的标识和距离计算；如果 $label*(w^Tx+b)>0$ 表示预测正确，否则预测错误。
* 现在目标很明确，就是要找到`w`和`b`，因此我们必须要找到最小间隔的数据点，也就是前面所说的`支持向量`。
  * 也就说，让最小的距离取最大.(最小的距离: 就是最小间隔的数据点；最大: 就是最大间距，为了找出最优超平面--最终就是支持向量)
  * 目标函数: $arg: max_{关于w, b} \left( min[label*(w^Tx+b)]*\frac{1}{||w||} \right) $
    1. 如果 $label*(w^Tx+b)>0$ 表示预测正确，也称`函数间隔`，$||w||$ 可以理解为归一化，也称`几何间隔`。
    2. 令 $label*(w^Tx+b)>=1$， 因为0～1之间，得到的点是存在误判的可能性，所以要保障 $min[label*(w^Tx+b)]=1$，才能更好降低噪音数据影响。
    3. 所以本质上是求 $arg: max_{关于w, b}  \frac{1}{||w||} $；也就说，我们约束(前提)条件是: $label*(w^Tx+b)=1$
* 新的目标函数求解:  $arg: max_{关于w, b}  \frac{1}{||w||} $
  * => 就是求: $arg: min_{关于w, b} ||w|| $ (求矩阵会比较麻烦，如果x只是 $\frac{1}{2}*x^2$ 的偏导数，那么。。同样是求最小值)
  * => 就是求: $arg: min_{关于w, b} (\frac{1}{2}*||w||^2)$ (二次函数求导，求极值，平方也方便计算)
  * 本质上就是求线性不等式的二次优化问题(求分隔超平面，等价于求解相应的凸二次规划问题)
* 通过拉格朗日乘子法，求二次优化问题
  * 假设需要求极值的目标函数 (objective function) 为 f(x,y)，限制条件为 φ(x,y)=M  # M=1
  * 设g(x,y)=M-φ(x,y)   # 临时φ(x,y)表示下文中 $label*(w^Tx+b)$
  * 定义一个新函数: F(x,y,λ)=f(x,y)+λg(x,y)
  * a为λ（a>=0），代表要引入的拉格朗日乘子(Lagrange multiplier)
  * 那么:  $L(w,b,\alpha)=\frac{1}{2} * ||w||^2 + \sum_{i=1}^{n} \alpha_i * [1 - label * (w^Tx+b)]$
  * 因为: $label*(w^Tx+b)>=1, \alpha>=0$ , 所以 $\alpha*[1-label*(w^Tx+b)]<=0$ , $\sum_{i=1}^{n} \alpha_i * [1-label*(w^Tx+b)]<=0$ 
  * 当 $label*(w^Tx+b)>1$ 则 $\alpha=0$ ，表示该点为<font color=red>非支持向量</font>
  * 相当于求解:  $max_{关于\alpha} L(w,b,\alpha) = \frac{1}{2} *||w||^2$ 
  * 如果求:  $min_{关于w, b} \frac{1}{2} *||w||^2$ , 也就是要求:  $min_{关于w, b} \left( max_{关于\alpha} L(w,b,\alpha)\right)$ 
* 现在转化到对偶问题的求解
  * $min_{关于w, b} \left(max_{关于\alpha} L(w,b,\alpha) \right) $ >= $max_{关于\alpha} \left(min_{关于w, b}\ L(w,b,\alpha) \right) $ 
  * 现在分2步
  * 先求:  $min_{关于w, b} L(w,b,\alpha)=\frac{1}{2} * ||w||^2 + \sum_{i=1}^{n} \alpha_i * [1 - label * (w^Tx+b)]$
  * 就是求`L(w,b,a)`关于[w, b]的偏导数, 得到`w和b的值`，并化简为: `L和a的方程`。
  * 参考:  如果公式推导还是不懂，也可以参考《统计学习方法》李航-P103<学习的对偶算法>
    ![计算拉格朗日函数的对偶函数](https://cdn.jsdelivr.net/gh/dongzhougu/imageuse1/SVM_松弛变量.jpg)
* 终于得到课本上的公式:  $max_{关于\alpha} \left( \sum_{i=1}^{m} \alpha_i - \frac{1}{2} \sum_{i, j=1}^{m} label_i·label_j·\alpha_i·\alpha_j·<x_i, x_j> \right) $
* 约束条件:  $a>=0$ 并且 $\sum_{i=1}^{m} a_i·label_i=0$

### 2.2 松弛变量(slack variable)



![松弛变量公式](https://cdn.jsdelivr.net/gh/dongzhougu/imageuse1/松弛变量.png)

* 我们知道几乎所有的数据都不那么干净, 通过引入松弛变量来 `允许数据点可以处于分隔面错误的一侧`。
* 约束条件:  $C>=a>=0$ 并且 $\sum_{i=1}^{m} a_i·label_i=0$
* 总的来说: 
  * ![松弛变量](https://cdn.jsdelivr.net/gh/dongzhougu/imageuse1/SVM_5_Lagrangemultiplier.png) 表示 `松弛变量`
  * 常量C是 `惩罚因子`, 表示离群点的权重（用于控制“最大化间隔”和“保证大部分点的函数间隔小于1.0” ）
    * $label*(w^Tx+b) > 1$ and alpha = 0 (在边界外，就是非支持向量)
    * $label*(w^Tx+b) = 1$ and 0< alpha < C (在分割超平面上，就支持向量)
    * $label*(w^Tx+b) < 1$ and alpha = C (在分割超平面内，是误差点 -> C表示它该受到的惩罚因子程度)
    * 参考地址: https://www.zhihu.com/question/48351234/answer/110486455
  * C值越大，表示离群点影响越大，就越容易过度拟合；反之有可能欠拟合。
  * 我们看到，目标函数控制了离群点的数目和程度，使大部分样本点仍然遵守限制条件。
  * 例如: 正类有10000个样本，而负类只给了100个（C越大表示100个负样本的影响越大，就会出现过度拟合，所以C决定了负样本对模型拟合程度的影响！，C就是一个非常关键的优化点！）
* 这一结论十分直接，SVM中的主要工作就是要求解 alpha.

### 2.3 核函数

* 对于线性可分的情况，效果明显
* 对于非线性的情况也一样，此时需要用到一种叫`核函数(kernel)`的工具将数据转化为分类器易于理解的形式。

> 利用核函数将数据映射到高维空间

* 使用核函数: 可以将数据从某个特征空间到另一个特征空间的映射。（通常情况下: 这种映射会将低维特征空间映射到高维空间。）
* 如果觉得特征空间很装逼、很难理解。
* 可以把核函数想象成一个包装器(wrapper)或者是接口(interface)，它能将数据从某个很难处理的形式转换成为另一个较容易处理的形式。
* 经过空间转换后: 低维需要解决的非线性问题，就变成了高维需要解决的线性问题。
* SVM 优化特别好的地方，在于所有的运算都可以写成内积(inner product: 是指2个向量相乘，得到单个标量 或者 数值)；内积替换成核函数的方式被称为`核技巧(kernel trick)`或者`核"变电"(kernel substation)`
* 核函数并不仅仅应用于支持向量机，很多其他的机器学习算法也都用到核函数。最流行的核函数: 径向基函数(radial basis function)hecn/AiLearning/blob/master/src/py2.x/ml/6.SVM/svm-complete.py



## 3. scikit-learn里的SVM

在scikit-learn里对SVM的算法实现都在包sklearn.svm下面，其中SVC类是用来进行分类任务的，SVR类是用来进行数值回归任务的。我们可能会有疑问，SVM不是用来进行分类的算法吗，为什么可以用来进行数值回归？实际上这只是数学上的一些扩展而已，在计算机里，可以用离散的数值计算来代替连续的数值回归。我们在K-近邻算法中已经看到过这种扩展实现。

我们以 SVC 为例。首先需要选择 SVM 的核函数，由参数 kernel 来指定，其中值 linea r表示线性核函数，它只能产生直线形状的分隔超平面；值 poly 表示多项式核函数，用它可以构建出复杂形状的分隔超平面；值 rbf 表示径向基核函数，即高斯核函数。

不同的核函数需要指定不同的参数。针对线性核函数，只需要指定参数 C，它表示对不符合最大间距规则的样本的惩罚力度，即前面介绍的松弛系数。针对多项式核函数，除了参数 C 之外，还需要指定 degree，它表示多项式的阶数。针对高斯核函数，除了参数C之外，还需要指定 gamma 值，这个值对应的是高斯核函数公式中的$\frac{1}{2 \sigma^{2}}$

下面先来看一个最简单的例子。我们生成一个有两个特征、包含两种类别的数据，然后用线性核函数的SVM算法进行分类：

```python
import matplotlib.pyplot as plt
import numpy as np
def plot_hyperplane(clf, X, y, 
                    h=0.02, 
                    draw_sv=True, 
                    title='hyperplan'):
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
        plt.scatter(X[y==label][:, 0], 
                    X[y==label][:, 1], 
                    c=colors[label], 
                    marker=markers[label])
    if draw_sv:
        sv = clf.support_vectors_
        plt.scatter(sv[:, 0], sv[:, 1], c='y', marker='x')
```

```python
from sklearn import svm
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=100, centers=2, 
                  random_state=0, cluster_std=0.3)
clf = svm.SVC(C=1.0, kernel='linear')
clf.fit(X, y)

plt.figure(figsize=(12, 4), dpi=144)
plot_hyperplane(clf, X, y, h=0.01, 
                title='Maximum Margin Hyperplan')
```

输出的图形如下所示，其中带有x标记的点即为支持向量，它保存在模型的 `support_vector`

![image-20200701095026364](https://cdn.jsdelivr.net/gh/dongzhougu/imageuse1/image-20200701095026364.png)

此处需要注意的是 `plot_hyperplane()` 函数，其主要功能是画出样本点，同时画出分类区间。它的主要原理是使用 `numpy.meshgrid()` 生成一个坐标矩阵，最后用 `contourf()` 函数为坐标矩阵中不同类别的点填充不同的颜色。其中，`contourf()`函数是画等高线并填充颜色的函数。

接着来看另外一个例子。我们生成一个有两个特征、包含三种类别的数据集，然后分别构造出4个SVM算法来拟合数据集，分别是线性核函数、三阶多项式核函数、gamma=0.5的高斯核函数，以及gamma=0.1的高斯核函数。最后把这4个SVM算法拟合出来的分隔超平面画出来。

```python
from sklearn import svm
from sklearn.datasets import make_blobs

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
    plt.subplot(2, 2, i+1)
    plot_hyperplane(clf, X, y, title=titles[i])
```

输出的图形如下所示，其中带有 x 标记的点即为支持向量。

![image-20200701101406980](https://cdn.jsdelivr.net/gh/dongzhougu/imageuse1/k_2.jpg)

左上角是线性核函数，它只能拟合出直线分隔超平面。右上角是三阶多项式核函数，它能拟合出复杂曲线分隔超平面。左下角是 `gamma=0.5` 的高斯核函数，右下角是 `gamma=0.1` 的高斯核函数，通过调整参数 `gamma` 的值，可以调整分隔超平面的形状。典型地，`gamma `值太大，越容易造成过拟合，`gamma` 值太小，高斯核函数会退化成线性核函数。我们把代码中的 `gamma`值 改为 100 和 0.01 后看一下输出图形是什么样的。

思考：左下角 gamma=0.5 的高斯核函数的图片，带有 x 标记的点是支持向量。我们之前介绍过，离分隔超平面最近的点是支持向量，为什么很多离分隔超平面很远的点，也是支持向量呢？

原因是高斯核函数把输入特征向量映射到了无限维的向量空间里，在映射后的高维向量空间里，这些点其实是离分隔超平面最近的点。当回到二维向量空间中时，这些点“看起来”就不像是距离分隔超平面最近的点了，但实际上它们就是支持向量。



## 4.示例：乳腺癌检测

之前我们使用逻辑回归算法进行过乳腺癌检测模型的学习和训练。这里我们再使用支持向量机算法来解决这个问题。首先我们载入数据：

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
# 载入数据
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target
print('data shape: {0}; no. positive: {1}; no. negative: {2}'.format(
    X.shape, y[y==1].shape[0], y[y==0].shape[0]))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

输出如下：

```kotlin
data shape: (569, 30); no. positive: 357; no. negative: 212
```

可以看出，我们的数据集很小。高斯核函数太复杂，容易造成过拟合，模型效果应该不会太好。我们先用高斯核函数来试一下，看与我们的猜测是否一致。

```python
from sklearn.svm import SVC
clf = SVC(C=1.0, kernel='rbf', gamma=0.1)
clf.fit(X_train, y_train)
train_score = clf.score(X_train, y_train)
test_score = clf.score(X_test, y_test)
print('train score: {0}; test score: {1}'.format(train_score, test_score))
```

输出如下：

```python
train score: 1.0; test score: 0.6491228070175439
```

训练数据集分数为1.0，交叉验证数据集分数只有0.65，这是典型的过拟合现象。这里 gamma=0.1，这个值相对已经比较小了。我们可以把gamma改的更小如0.0001看看什么结果。

当然，我们完全可以使用前面介绍过的 `GridSearchCV` 来自动选择最优参数。我们看看如果使用高斯模型，最优的gamma参数值是多少，其对应的模型交叉验证评分是多少。

```python
from sklearn.model_selection import GridSearchCV
from utils import plot_param_curve
 gammas = np.linspace(0, 0.0003, 30)
param_grid = {'gamma': gammas}
 clf = GridSearchCV(SVC(), param_grid, cv=5,return_train_score=True)
clf.fit(X, y)
print("best param: {0}\nbest score: {1}".format(clf.best_params_,clf.best_score_))
plt.figure(figsize=(10, 4), dpi=144)
plot_param_curve(plt, gammas, clf.cv_results_, xlabel='gamma')
plt.show()
```

输出如下：

```python
best param: {'gamma': 0.00011379310344827585}
best score: 0.9367334264865704
```

![image-20200701102906477](https://cdn.jsdelivr.net/gh/dongzhougu/imageuse1/image-20200701101406980.png)

由此可见，即使在最好的 gamma 参数下，其平均最优得分也只是0.9367311072056239。我们选择在gamma为0.01时，画出学习曲线，更直观地观察模型拟合情况。

```python
import time
from sklearn.model_selection import ShuffleSplit
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
title = 'Learning Curves for Gaussian Kernel'
start = time.clock()
plt.figure(figsize=(10, 4), dpi=144)
plot_learning_curve(plt, SVC(C=1.0, kernel='rbf', gamma=0.01), title, X, y, ylim=(0.5, 1.01), cv=cv)
print('elaspe: {0:.6f}'.format(time.clock()-start))
```

输出如下：

```python
elaspe: 0.687500
```

画出来的图形如下所示：

![image-20200701103141555](https://cdn.jsdelivr.net/gh/dongzhougu/imageuse1/k_7.png)

这是明显的过拟合现象，交叉验证数据集的评分非常低，且离训练数据集评分非常远。



------

**接下来换一个模型，使用二阶多项式核函数的SVM来拟合模型，看看结果如何。**

```python
from sklearn.svm import SVC
clf = SVC(C=1.0, kernel='poly', degree=2)
clf.fit(X_train, y_train)
train_score = clf.score(X_train, y_train)
test_score = clf.score(X_test, y_test)
print('train score: {0}; test score: {1}'.format(train_score, test_score))
```

输出如下：

```python
train score: 0.9098901098901099; test score: 0.9210526315789473
```

看起来结果好多了。作为对比，我们画出一阶多项式核函数的SVM和二阶多项式核函数的SVM的学习曲线，观察模型的拟合情况。

```go
import time
from sklearn.model_selection import ShuffleSplit
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
title = 'Learning Curves with degree={0}'
degrees = [1, 2]
start = time.clock()
plt.figure(figsize=(12, 4), dpi=144)
for i in range(len(degrees)):
    plt.subplot(1, len(degrees), i + 1)
    plot_learning_curve(plt, SVC(C=1.0, kernel='poly', degree=degrees[i]),
                        title.format(degrees[i]), X, y, ylim=(0.8, 1.01), cv=cv, n_jobs=4)
print('elaspe: {0:.6f}'.format(time.clock()-start))
```

输出如下：

```css
elaspe: 0.281250
```

输出的图形如下所示：

![image-20200701103641519](https://cdn.jsdelivr.net/gh/dongzhougu/imageuse1/image-20200701102906477.png)

前面我们使用逻辑回归算法来处理乳腺癌检测问题时，使用二阶多项式增加特征，同时使用L1范数作为正则项，其拟合效果比这里的支持向量机效果好。更重要的是，逻辑回归算法的运算效率远远高于二阶多项式核函数的支持向量机算法。当然，这里的支持向量机算法的效果还是比使用L2范数作为正则项的逻辑回归算法好的。由此可见，模型选择和模型参数调优，在工程实践中有着非常重要的作用的。



## 5. 拓展SVM的理解

### 什么是 SVM ？

Support Vector Machine, 一个普通的 SVM 就是一条直线罢了，用来完美划分 linearly separable 的两类。但这又不是一条普通的直线，这是无数条可以分类的直线当中最完美的，因为它恰好在两个类的中间，距离两个类的点都一样远。而所谓的 Support vector 就是这些离分界线最近的『点』。如果去掉这些点，直线多半是要改变位置的。可以说是这些 vectors （主，点点） support （谓，定义）了 machine （宾，分类器）...

![k_2](https://cdn.jsdelivr.net/gh/dongzhougu/imageuse1/k_8.png "k_2")

所以谜底就在谜面上啊朋友们，只要找到了这些最靠近的点不就找到了 SVM 了嘛。

如果是高维的点，SVM 的分界线就是平面或者超平面。其实没有差，都是一刀切两块，我就统统叫直线了。

### 怎么求解 SVM ？

关于这条直线，我们知道 

(1)它离两边一样远，(2)最近距离就是到support vector的距离，其他距离只能更远。

于是自然而然可以得到重要表达 <b>I. direct representation</b>

![k_7](https://cdn.jsdelivr.net/gh/dongzhougu/imageuse1/image-20200701103141555.png "k_7")

（可以把 margin 看作是 boundary 的函数，并且想要找到使得是使得 margin 最大化的boundary，而 margin(*) 这个函数是: 输入一个 boundary ，计算（正确分类的）所有苹果和香蕉中，到 boundary 的最小距离。）

又有最大又有最小看起来好矛盾。实际上『最大』是对这个整体使用不同 boundary 层面的最大，『最小』是在比较『点』的层面上的最小。外层在比较 boundary 找最大的 margin ，内层在比较点点找最小的距离。

其中距离，说白了就是点到直线的距离；只要定义带正负号的距离，是 {苹果+1} 面为正 {香蕉-1} 面为负的距离，互相乘上各自的 label ![k_8](https://cdn.jsdelivr.net/gh/dongzhougu/imageuse1/image-20200701103641519.png "k_8") ，就和谐统一民主富强了。

![k_9](https://cdn.jsdelivr.net/gh/dongzhougu/imageuse1/k_9.png "k_9")

到这里为止已经说完了所有关于SVM的直观了解，如果不想看求解，可以跳过下面一大段直接到 objective function 。

直接表达虽然清楚但是求解无从下手。做一些简单地等价变换（分母倒上来）可以得到 <b>II. canonical representation </b> （敲黑板）

![k_10](https://cdn.jsdelivr.net/gh/dongzhougu/imageuse1/k_10.png "k_10")

要得到 <b>III. dual representation</b> 之前需要大概知道一下拉格朗日乘子法 (method of lagrange multiplier)，它是用在有各种约束条件(各种 "subject to" )下的目标函数，也就是直接可以求导可以引出 dual representation（怎么还没完摔）

![k_11](https://cdn.jsdelivr.net/gh/dongzhougu/imageuse1/k_11.png "k_11")

稍微借用刚刚数学表达里面的内容看个有趣的东西: 

还记得我们怎么预测一个新的水果是苹果还是香蕉吗？我们代入到分界的直线里，然后通过符号来判断。

刚刚w已经被表达出来了也就是说这个直线现在变成了:  ![k_12](https://cdn.jsdelivr.net/gh/dongzhougu/imageuse1/k_12.png "k_12")

看似仿佛用到了所有的训练水果，但是其中 ![k_13](https://cdn.jsdelivr.net/gh/dongzhougu/imageuse1/k_13.png "k_13") 的水果都没有起到作用，剩下的就是小部分靠边边的 Support vectors 呀。

<b>III. dual representation</b>

![k_14](https://cdn.jsdelivr.net/gh/dongzhougu/imageuse1/k_14.png "k_14")

<b>如果香蕉和苹果不能用直线分割呢？</b>

![k_3](https://cdn.jsdelivr.net/gh/dongzhougu/imageuse1/k_15.png "k_3")

Kernel trick. 

其实用直线分割的时候我们已经使用了 kernel ，那就是线性 kernel , ![k_15](https://cdn.jsdelivr.net/gh/dongzhougu/imageuse1/k_3.jpg "k_15")

如果要替换 kernel 那么把目标函数里面的内积全部替换成新的 kernel function 就好了，就是这么简单。

第一个武侠大师的比喻已经说得很直观了，低维非线性的分界线其实在高维是可以线性分割的，可以理解为——『你们是虫子！』分得开个p...（大雾）

<b>如果香蕉和苹果有交集呢？</b>

![k_4](https://cdn.jsdelivr.net/gh/dongzhougu/imageuse1/k_4.jpg "k_4")

![k_16](https://cdn.jsdelivr.net/gh/dongzhougu/imageuse1/k_16.png "k_16")

<b>如果还有梨呢？</b>

![k_5](https://cdn.jsdelivr.net/gh/dongzhougu/imageuse1/k_5.jpg "k_5")

可以每个类别做一次 SVM: 是苹果还是不是苹果？是香蕉还是不是香蕉？是梨子还是不是梨子？从中选出可能性最大的。这是 one-versus-the-rest approach。

也可以两两做一次 SVM: 是苹果还是香蕉？是香蕉还是梨子？是梨子还是苹果？最后三个分类器投票决定。这是 one-versus-one approace。

但这其实都多多少少有问题，比如苹果特别多，香蕉特别少，我就无脑判断为苹果也不会错太多；多个分类器要放到一个台面上，万一他们的 scale 没有在一个台面上也未可知。