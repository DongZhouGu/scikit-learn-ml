# 线性回归算法

线性回归算法是使用线性方程对数据集拟合的算法，本文从单变量线性回归算法、多变量线性回归算法，其中损失函数以及梯度下降算法的推导过程会用到部分线性代数和偏导数；接着重点介绍了梯度下降算法的求解步骤以及性能优化方面的内容；最后通过一个房价预测模型，介绍了线性回归算法性能优化的一些常用步骤和方法。

## 线性回归概述

说到回归，一般都是指 `线性回归(linear regression)`。线性回归意味着可以将输入项分别乘以一些常量，再将结果加起来得到输出。回归的目的是预测数值型的目标值，最直接的办法是依据输入写出一个目标值的计算公式。

假如你想要预测兰博基尼跑车的功率大小，可能会这样计算:

> HorsePower = 0.0015 * annualSalary - 0.99 * hoursListeningToPublicRadio
>

这就是所谓的 `回归方程(regression equation)`，其中的 0.0015 和 -0.99 称作 `回归系数（regression weights）`，求这些回归系数的过程就是回归。一旦有了这些回归系数，再给定输入，做预测就非常容易了。具体的做法是用回归系数乘以输入值，再将结果全部加在一起，就得到了预测值。我们这里所说的，回归系数是一个向量，输入也是向量，这些运算也就是求出二者的内积。

## 单变量线性回归算法

先考虑最简单的单变量线性回归算法，即只有一个输入特征。

### 预测函数

针对数据集x和y，预测函数会根据输入特征x来计算输出值h(x)。其输入和输出的函数关系如下：
$$
h_{\theta}(x)=\theta_{0}+\theta_{1} x
$$
这个方程表达的是一条直线。我们的任务是构造一个 $h_{\theta}$ 函数，来映射数据集中的输入特征x和输出值y，使得预测函数 $h_{\theta}$ 计算出来的值与真实值y的整体误差最小。构造  $h_{\theta}$ 函数的关键就是找到合适的 $\theta_{0}$和 $\theta_{1}$ 的值， 模型参数，也就是所说的模型参数。

假设有如下的数据集：

| 输入特征x | 输出y |
| :-------: | :---: |
|     1     |   4   |
|     2     |   6   |
|     3     |  10   |
|     4     |  15   |

假设模型参数 $\theta_{0}=1,  \theta_{1}=3$ ,  则预测函数为 $h_{\theta}(x)=1+3 x$ 。针对数据集中的第一个样本，输入为1，根据模型函数预测出来的值是4，与输出值y是吻合的。针对第二个样本，输入为2，根据模型函数预测出来的值是7，与实际输出值y相差1。模型的求解过程就是找出一组最合适的模型参数 $\theta_{0}$和 $\theta_{1}$，以便能最好地拟合数据集。

怎样来判断最好地拟合了数据集呢？没错，就是使用损失函数（也叫损失函数）。当拟合损失最小时，即找到了最好的拟合参数。

### 损失函数

单变量线性回归算法的损失函数是：

$$
J(\theta)=J\left(\theta_{0}, \theta_{1}\right)=\frac{1}{2 m} \sum_{i=1}^{m}\left(h\left(x^{(i)}\right)-y^{(i)}\right)^{2}
$$
其中，$h\left(x^{(i)}\right)-y^{(i)}$ 是预测值和真实值之间的误差，故损失就是预测值和真实值之间误差平方的平均值，之所以乘以1/2是为了方便计算。这个函数也称为均方差公式。有了损失函数，就可以精确地测量模型对训练样本拟合的好坏程度。

### 梯度下降算法

有了预测函数，也可以精确地测量预测函数对训练样本的拟合情况。但怎么求解模型参数 $\theta_{0}$和 $\theta_{1}$的值呢？这时梯度下降算法就排上了用场。

我们的任务是找到合适的 $\theta_{0}$和 $\theta_{1}$ ，使得损失函数 $J\left(\theta_{0}, \theta_{1}\right)$ 最小。为了便于理解，我们切换到三维空间来描述这个任务。在一个三维空间里，以  $\theta_{0}$ 作为 x 轴， 以 $\theta_{1}$ 作为 y 轴，以损失函数 $J\left(\theta_{0}, \theta_{1}\right)$ 作为 z 轴，那么我们的任务就是要找出当 z 轴上的值最小的时候所对应的 x 轴上的值和 y 轴上的值。

**梯度下降算法的原理：**先随机选择一组 $\theta_{0}$ 和 $\theta_{1}$ ，同时选择一个参数 $\alpha$ 作为移动的步长。然后，让x轴上的 $\theta_{0}$ 和 y轴上的  $\theta_{1}$ 别向特定的方向移动一小步，这个步长的大小就由参数  $\alpha$ 决定。经过多次迭代之后，x 轴和 y 轴上的值决定的点就慢慢靠近 z 轴上的最小值处，如图所示。

![img](https://cdn.jsdelivr.net/gh/dongzhougu/imageuse1/u=3721595541,2272727131&fm=26&gp=0.jpg)

那特定的方向怎么确定呢？答案是**偏导数**。

可以简单地把偏导数理解为斜率。我们要让 $\theta_{j}$ 不停地迭代，由当前  $\theta_{j}$ 的值，根据 $J(\theta)$ 的偏导数函数，算出 $J(\theta)$ 在  $\theta_{j}$ 上的斜率，然后在乘以学习率  $\alpha$ ，就可以让 $\theta_{j}$ 往 $J(\theta)$ 变小的方向迈一小步。

用数学来描述上述过程，梯度下降的公式为：
$$
\theta_{j}=\theta_{j}-\alpha \frac{\partial}{\partial \theta_{j}} J(\theta)
$$
把损失函数 $J(\theta)$ 的定义代入上面的公式中，不难推导出梯度下降算法公式：
$$
\begin{array}{c}
\theta_{0}=\theta_{0}-\frac{\alpha}{m} \sum_{i=1}^{m}\left(h\left(x^{(i)}\right)-y^{(j)}\right) \\
\\
\theta_{1}=\theta_{1}-\frac{\alpha}{m} \sum_{i=1}^{m}\left(\left(h\left(x^{(i)}\right)-y^{(i)}\right) x_{i}\right)
\end{array}
$$
公式中， $\alpha$  是学习率；m 是训练样本的个数: $h\left(x^{(i)}\right)-y^{(i)}$ 是模型预测值和真实值的误差。需要注意的是，针对

 $\theta_{0}$ 和 $\theta_{1}$ 分别求出了其迭代公式，在 $\theta_{1}$ 的迭代公式里，累加器中还需要乘以 $x_{i}$, 具体参考扩展部分。

## 多变量线性回归算法

实际应用中往往不止一个输入特征。熟悉了单变量线性回归算法后，我们来探讨一下多变量线性回归算法。

### 预测函数

上面介绍的单变量线性回归模型里只有一个输入特征，我们推广到更一般的情况，即多个输入特征。此时输出y的值由n个输入特征 $x_{1}, x_{2}, \ldots, x_{n}$ 决定。那么预测函数模型可以改写如下：

$$
h_{\theta}(x)=\theta_{0}+\theta_{1} x_{1}+\theta_{2} x_{2}+\ldots+\theta_{n} x_{n}
$$
假设 $x_{0}=1$，那么上面的公式可以重写为：
$$
h_{\theta}(x)=\sum_{j=0}^{n} \theta_{j} x_{j}
$$
其中，$\theta_{0}, \theta_{1}, \dots, \theta_{n}$ 统称为 $\theta$ , 是预测函数的参数。即一组 $\theta$ 值就决定了一个预测函数，记为 $h_{\theta}(x)$ , 为了简便起见，在不引起误解的情况下可以简写为 $h(x)$ 。理论上，预测函数有无穷多个，我们求解的目标就是找出一个最优的 $\theta$ 值。

#### 向量形式的预测函数

根据向量乘法运算法则，损失函数可重写为：

$$
h_{\theta}(x)=\left[\theta_{0}, \theta_{1}, \cdots, \theta_{n}\right]\left[\begin{array}{c}
x_{0} \\
x_{1} \\
\vdots \\
x_{n}
\end{array}\right]=\theta^{T} x
$$
此处，依然假设 $x_{0}=1$， $x_{0}$ 称为模型偏置（bias）。

写成向量形式的预测函数有两个原因。一是因为简洁，二是因为在实现算法时，要用到数值计算里的矩阵运算来提高效率，比如 `Numpy` 库里的矩阵运算。

#### 向量形式的训练样本

假设输入特征的个数是n，即 $x_{1}, x_{2}, \ldots, x_{n}$ , 我们总共有 m 个训练样本，为了书写方便，假设 $x_{0}=1$。这样训练样本可以写成矩阵的形式，即矩阵里每一行都是一个训练样本，总共有 m 行，每行有 n+1 列。

> 思考：为什么不是n列而是n+1列？答案是：把模型偏置 $x_{0}$也加入了训练样本里。最后把训练样本写成一个矩阵，如下：

$$
\boldsymbol{X}=\left[\begin{array}{ccccc}
x_{0}^{(1)} & x_{1}^{(1)} & x_{2}^{(1)} & \dots & x_{n}^{(1)} \\
x_{0}^{(2)} & x_{1}^{(2)} & x_{2}^{(2)} & \dots & x_{n}^{(2)} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
x_{0}^{(m)} & x_{1}^{(m)} & x_{2}^{(m)} & \cdots & x_{n}^{(m)}
\end{array}\right], \theta=\left[\begin{array}{c}
\theta_{0} \\
\theta_{1} \\
\theta_{2} \\
\vdots \\
\theta_{n}
\end{array}\right]
$$

理解训练样本矩阵的关键在于理解这些上标和下标的含义。其中，带括号的上标表示样本序号，从1到m；下标表示特征序号，从0到n，其中 $x_{0}$ 为常数1。

> $x_{j}^{(i)}$ 表示第 i 个训练样本的第 j 个特征的值。而 $x^{(i)}$ 只有上标，则表示第 i 个训练样本所构成的列向量。

综上，训练样本的预测值 $h_{\theta}(X)$ ，可以使用下面的矩阵运算公式：

$$
h_{\theta}(X)=X \theta
$$

### 损失函数

多变量线性回归算法的损失函数：

$$
J(\theta)=\frac{1}{2 m} \sum_{i=1}^{m}\left(h\left(x^{(i)}\right)-y^{(i)}\right)^{2}
$$
其中，模型参数 $\theta$ 为 n+1 维的向量，$h\left(x^{(i)}\right)-y^{(i)}$ 是预测值和实际值的差，这个形式和单变量线性回归算法的类似。

损失函数有其对应的矩阵形式：
$$
J(\theta)=\frac{1}{2 m}(X \theta-\vec{y})^{T}(X \theta-\vec{y})
$$
其中，X 为 $m \times(n+1)$ 维的训练样本矩阵；上标T表示转置矩阵；$\vec{y}$ 表示由所有的训练样本的输出 $y^{(i)}$ 构成的向量。这个公式的优势是：没有累加器，不需要循环，直接使用矩阵运算，就可以一次性计算出对特定的参数 $\theta$ 下模型的拟合损失。

### 梯度下降算法

根据单变量线性回归算法的介绍，梯度下降的公式为：
$$
\theta_{j}=\theta_{j}-\alpha \frac{\partial}{\partial \theta_{j}} J(\theta)
$$
公式中，下标 j 是参数的序号，其值从 0 到 n； $\alpha$ 为学习率。把损失函数代入上式，利用偏导数计算法则，不难推导出梯度下降算法的参数迭代公式：
$$
\theta_{j}=\theta_{j}-\frac{\alpha}{m} \sum_{i=1}^{m}\left(\left(h\left(x^{(i)}\right)-y^{(i)}\right) x_{j}^{(i)}\right)
$$
我们可以对比一下单变量线性回归函数的参数迭代公式。实际上和多变量线性回归函数的参数迭代公式是一模一样的。惟一的区别就是因为 $x_{0}$ 为常数1，在单变量线性回归算法的参数迭代公式中省去了。

应用这个公式编写机器学习算法，一般步骤如下：

- 确定学习率： $\alpha$ 太大可能会使损失函数无法收敛，太小则计算太多，机器学习算法效率就比较低。

- 参数初始化：比如让所有的参数都以1作为起始点，$\theta_{0}=1, \theta_{1}=1, \dots, \theta_{n}=1$，根据预测值和损失函数，就可以算出在参数起始位置的损失。需要注意的是，参数起始点可以根据实际情况灵活选择，以便让机器学习算法的性能更高，比如选择比较靠近极点的位置。

- 计算参数的下一组值：据梯度下降参数迭代公式，分别同时计算出新的 $\theta_{j}$ 值，进而得到新的预测函数 $h_{\theta}(x)$ 。再根据新的预测函数，代入损失函数就可以算出新的损失。

- 确定损失函数是否收敛：拿新的和旧的损失进行比较，看损失是不是变得越来越小。如果两次损失之间的差异小于误差范围，即说明已经非常靠近最小损失了，就可以近似地认为我们找到了最小损失。如果两次损失之间的差异在误差范围之外，重复步骤（3）继续计算下一组参数直到找到最优解。

  

## 模型优化

线性回归模型常用的优化方法，包括增加多项式特征以及数据归一化处理等。

### 多项式与线性回归

当线性回归模型太简单导致欠拟合时，我们可以增加特征多项式来让线性回归模型更好地拟合数据。比如有两个特征  $x_{1}$ 和 $x_{2}$ ，可以增加两个特征的乘积 $x_{1} \times x_{2}$ 作为新特征  $x_{3}$ 。同理，我们也可以增加 $x_{1}^{2}$ 和 $x_{2}^{2}$  分别作为新特征  $x_{4}$ 和 $x_{5}$ 。

在 `scikit-learn` 里，线性回归是由类 `sklearn.learn_model.LinearRegression` 实现的，多项式由类`sklearn.preprocessing.PolynomialFeatures` 实现。那么要怎样添加多项式特征呢？我们需要用一个管道把两个类串起来，即用 `sklearn.pipeline.Pipeline` 把这两个模型串起来。

比如下面的函数就可以创建一个多项式拟合：

```python
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
def polynomial_model(degree=1):
    polynomial_features = PolynomialFeatures(degree=degree,include_bias=False)
    linear_regression = LinearRegression(normalize=True)
    # 这是一个流水线，先增加多项式阶数，然后再用线性回归算法来拟合数据
    pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("linear_regression", linear_regression)])
    return pipeline
```

一个 Pipeline 可以包含多个处理节点，在 scikit-learn 里，除了最后一个节点外，其他的节点都必须实现 fit() 方法和 transform() 方法，最后一个节点只需要实现 fit() 方法即可。当训练样本数据送进 Pipeline 里进行处理时，它会逐个调用节点的 fit() 方法和 transform() 方法，最后调用最后一个节点的 fit() 方法来拟合数据。管道的示意图如下所示：

![image-20200630090937011](https://cdn.jsdelivr.net/gh/dongzhougu/imageuse1/image-20200630093109778.png)



### 数据归一化

当线性回归模型有多个输入特征时，特别是使用多项式添加特征时，需要对数据进行归一化处理。比如，特征

$x_{1}$ 的范围在[1,4]之间，特征 $x_{2}$ 的范文在[1,2000]之间，这种情况下，可以让 $x_{1}$除以4来作为新特征 $x_{1}$，同时让 $x_{2}$ 

除以2000来作为新特征 $x_{2}$ ，该过程称为特征缩放（feature scaling）。可以使用特征缩放来对训练样本进行归一化处理，处理后的特征范围在[0,1]之间。

- 归一化处理的目的是让算法收敛更快，提升模型拟合过程中的计算效率。
- 进行归一化处理后，当有个新的样本需要计算预测值时，也需要先进行归一化处理，再通过模型来计算预测值，计算出来的预测值要再乘以归一化处理的系数，这样得到的数据才是真正的预测数据。
- 在 `scikit-learn` 里，使用 `LinearRegression` 进行线性回归时，可以指定 `normalize=True` 来对数据进行归一化处理。



## 示例1：使用线性回归算法拟合正弦函数

首先生成200个在区间 $[2 \pi, 2 \pi]$ 内的正弦函数上的点，并给这些点加上一些随机的噪声。

```python
import numpy as np
n_dots = 200
X = np.linspace(-2 * np.pi, 2 * np.pi, n_dots)
Y = np.sin(X) + 0.2 * np.random.rand(n_dots) - 0.1
# 把一个n维向量转换成一个n*1维的矩阵
X = X.reshape(-1, 1)
Y = Y.reshape(-1, 1);
```

使用 `PolynomialFeatures` 和 `Pipeline` 创建一个多项式拟合模型

```python
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
def polynomial_model(degree=1):
    polynomial_features = PolynomialFeatures(degree=degree,include_bias=False)
    linear_regression = LinearRegression(normalize=True)
    pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("linear_regression", linear_regression)])
    return pipeline
```

分别用2/3/5/10阶多项式来拟合数据集：

```python
from sklearn.metrics import mean_squared_error
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
```

算出每个模型拟合的评分，此外，使用 `mean_squared_error` 算出均方根误差，即实际的点和模型预点之间的距离，均方根误差越小说明模型拟合效果越好——上述代码的输出结果为：

``` python
degree: 2; train score: 0.1543189069883787; mean squared error: 0.43058829267318416
degree: 3; train score: 0.2755383996826518; mean squared error: 0.3688679883773196
degree: 5; train score: 0.8982707756590037; mean squared error: 0.051796609130712795
degree: 10; train score: 0.9935830575581858; mean squared error: 0.0032672603337543927
```

从输出结果可以看出，多项式阶数越高，拟合评分越高，均方根误差越小，拟合效果越好。

把不同模型的拟合效果在二维坐标上画出来，可以清楚地看到不同阶数的多项式的拟合效果：

```python
import matplotlib.pyplot as plt
from matplotlib.figure import SubplotParams
plt.figure(figsize=(12,6),dpi=200,subplotpars=SubplotParams(hspace=0.3))
for i,r in enumerate(results):
    fig = plt.subplot(2,2,i+1)
    plt.xlim(-8,8)
    plt.title("LinearRegression degree={}".format(r["degree"]))
    plt.scatter(X,Y,s=5,c='b',alpha=0.5)
    plt.plot(X,r["model"].predict(X),'r-')
plt.show()
```

使用 `SubplotParam`s 调整了子图的竖直间距，并且使用 `subplot()` 函数把4个模型的拟合情况都画在同一个图形上。上述代码的输出结果如下图所示：

![image-20200630092742917](https://cdn.jsdelivr.net/gh/dongzhougu/imageuse1/image-20200630090937011.png)

在[-2π，2π]区间内，10阶多项式对数据拟合得非常好，我们可以试着画出这10阶模型在[-20,20]的区域内的曲线，观察一下该模型的曲线和正弦函数的差异。代码如下：

```python
plt.figure(figsize=(12,6),dpi=200)
X = np.linspace(-20,20,2000).reshape(-1, 1)
Y = np.sin(X).reshape(-1, 1)
model_10 = results[3]["model"]
plt.xlim(-20,20)
plt.ylim(-2,2)
plt.plot(X,Y,'b-')
plt.plot(X,model_10.predict(X),'r-')
dot1 = [-2*np.pi,0]
dot2 = [2*np.pi,0]
plt.scatter(dot1[0],dot1[1],s=50,c='r')
plt.scatter(dot2[0],dot2[1],s=50,c='r')
plt.show()
```

上述代码的输出结果如下图：

![image-20200630093109778](https://cdn.jsdelivr.net/gh/dongzhougu/imageuse1/image-20200630100257358.png)

从图中可以看出，10阶多项式模型只有在区间[-2π,2π]之间对正弦曲线拟合较好，在此区间以外，两者相差甚远。此案例告诉我们，每个模型都有自己的适用范围，在满足适用范围的基本前提下，要尽可能寻找拟合程度最高的模型来使用。



## 示例2：预测房价

本节使用 `scikit-learn` 自带的波士顿房价数据来训练模型，然后用模型来预测房价。

### 输入特征

房价和哪些因素有关？很多人可能对这个问题特别敏感，随时可以列出很多，如房子面子、房子地理位置、周边教育资源、周边商业资源、房子朝向、年限、小区情况等。在 `scikit-learn`的波士顿房价数据集里，它总共收集了13个特征，具体如下：

- CRIM：城镇人均犯罪率。

- ZN：城镇超过25000平方英尺的住宅区域的占地比例。

- INDUS：城镇非零售用地占地比例。

- CHAS：是否靠近河边，1为靠近，0为远离。

- NOX：一氧化氮浓度

- RM：每套房产的平均房间个数。

- AGE：在1940年之前就盖好，且业主自住的房子的比例。

- DIS：与波士顿市中心的距离。

- RAD：周边高速公路的便利性指数。

- TAX：每10000美元的财产税率。

- PTRATIO：小学老师的比例。

- B：城镇黑人的比例。

- LSTAT：地位较低的人口比例。


从这些指标里可以看到中美指标的一些差异。当然，这个数据是在1993年之前收集的，可能和现在会有差异。不要小看了这些指标，实际上一个模型的好坏和输入特征的选择关系密切。大家可以思考一下，如果要在中国预测房价，你会收集哪些特征数据？这些特征数据的可获得性如何？收集成本多高？

先导入数据：

```python
from sklearn.datasets import load_boston
boston = load_boston()
X = boston.data
y = boston.target
print(X.shape)  # (506, 13)
```

表明这个数据集有506个样本，每个样本有13个特征。整个训练样本放在一个506*13的矩阵里。可以通过X[0]来查看一个样本数据：

```python
print(X[0])
array([6.320e-03, 1.800e+01, 2.310e+00, 0.000e+00, 5.380e-01, 6.575e+00,
       6.520e+01, 4.090e+00, 1.000e+00, 2.960e+02, 1.530e+01, 3.969e+02,
       4.980e+00])
```

还可以通过 `boston.features_names` 来查看这些特征的标签：

```python
print(boston.feature_names)
```

输出如下：

```python
array(['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
       'TAX', 'PTRATIO', 'B', 'LSTAT'], dtype='<U7')
```

我们可以把特征和数值对应起来，观察一下数据。

### 模型训练

在 `scikit-learn` 里，`LinearRegression` 类实现了线性回归算法。在对模型进行训练之前，我们需要先把数据集分成两份，以便评估算法的准确性。

```python
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=3)
```

由于数据量比较小，我们只选了20%的样本来作为测试数据集。接着，训练模型并测试模型的准确性评分：

```python
import time
from sklearn.linear_model import LinearRegression
model = LinearRegression()
start = time.process_time()
model.fit(X_train,y_train)
train_score = model.score(X_train,y_train)
test_score = model.score(X_test,y_test)
print("elaspe:{0:.6f};train_score:{1:0.6f};test_score:{2:.6f}"
      .format(time.process_time()-start,train_score,test_score))
```

我们顺便统计了模型的训练时间，除此之外，统计模型对训练样本的准确性得分（即对训练样本拟合的好坏程度）`train_score`，还测试了模型对测试样本的得分test_score。运行结果如下：

```python
elaspe:0.000000;train_score:0.723941;test_score:0.795262
```

从得分情况来看，模型的拟合效果一般，还有没有办法来优化模型的拟合效果呢？

### 模型优化

首先观察一下数据，特征数据的范围相差比较大，最小的在$10^{-3}$级别，而最大的在$10^{2}$级别，看来我们需要先把数据进行归一化处理。归一化处理最简单的方式是，创建线性回归模型时增加normalize=True参数：

```python
model = LinearRegression(normalize=True)
```

当然，数据归一化处理只会加快算法收敛速度，优化算法训练的效率，无法提升算法的准确性。

怎么样优化模型的准确性呢？我们回到训练分数上来，可以观察到模型针对训练样本的评分比较低（train_score:0.723941），即模型对训练样本的拟合成本比较高，这是一个典型的欠拟合现象。回忆我们之前介绍的优化欠拟合模型的方法，一是挖掘更多的输入特征，而是增加多项式特征。在我们这个例子里，通过使用低成本的方案——即增加多项式特征来看能否优化模型的性能。增加多项式特征，其实就是增加模型的复杂度。

我们使用之前创建多项式模型的函数 `polynomial_model`，接着，我们使用二阶多项式来拟合数据：

```python
model = polynomial_model(degree=2)
start = time.process_time()
model.fit(X_train, y_train)
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print("elaspe:{0:.6f};train_score:{1:0.6f};test_score:{2:.6f}"
      .format(time.process_time() - start, train_score, test_score))
```

输出结果是：

```python
elaspe:0.078125;train_score:0.930547;test_score:0.860049
```

训练样本分数和测试分数都提高了，看来模型确实得到了优化。我们可以把多项式改为3阶看一下效果：

```python
elaspe:0.093750;train_score:1.000000;test_score:-105.548323
```

改为3阶多项式后，针对训练样本的分数达到了1，而针对测试样本的分数确实负数，说明这个模型过拟合了。

思考：我们总共有13个输入特征，从一阶多项式变为二阶多项式，输入特征个数增加了多少个？
 参考：二阶多项式共有：13个单一的特征，$C_{13}^{2}=78$ 个两两配对的特征，13个各自平方的特征，共计104个特征。比一阶多项式的13个特征增加了91个特征。

### 学习曲线

更好的方法是画出学习曲线，这样对模型的状态以及优化的方向就一目了然。

```python
import matplotlib.pyplot as plt
from utils import plot_learning_curve
from sklearn.model_selection import ShuffleSplit
cv = ShuffleSplit(n_splits=10,test_size=0.2,random_state=0)
plt.figure(figsize=(18,4),dpi=200)
title = 'Learning Curves (degree={0})'
degrees = [1,2,3]
start = time.process_time()
for i in range(len(degrees)):
    plt.subplot(1,3,i+1)
    plot_learning_curve(plt,polynomial_model(degrees[i]),title.format(degrees[i]),
                        X,y,ylim=(0.01,1.01),cv=cv)
    print('elaspe:{0:.6f}'.format(time.process_time()-start))
```

输出如下：

![image-20200630100257358](https://cdn.jsdelivr.net/gh/dongzhougu/imageuse1/image-20200630092742917.png)

从学习曲线中可以看出，一阶多项式欠拟合，因为针对训练样本的分数比较低；而三阶多项式过拟合，因为针对训练样本的分数达到1，却看不到交叉验证数据集的分数。针对二阶多项式拟合的情况，虽然比一阶多项式的效果好，但从图中可以明显地看出来，针对训练数据集的分数和针对交叉验证数据集的分数之间的间隔比较大，这说明训练样本数量不够，我们应该去采集更多的数据，以提高模型的准确性。



## 拓展阅读

本节内容涉及到较多的数学知识，特别是矩阵和偏导数运算法则。如果阅读起来有困难，可以先跳过。如果有一定数学基础，这些知识对理解算法的实现细节及算法的效率有较大的帮助。

### 公式推导的数学基础

AI的数学基础最主要的是高等数学、线性代数、概率论与数理统计这三门课程。下面是简易的入门文章供参考

- 高等数学 https://zhuanlan.zhihu.com/p/36311622
- 线性代数 https://zhuanlan.zhihu.com/p/36584206
- 概率论与数理统计 https://zhuanlan.zhihu.com/p/36584335

### 随机梯度下降算法

本章介绍的梯度下降算法迭代公式称为批量梯度下降算法（Batch Gradient Descent，简称BGD），用它对参数进行一次迭代运算，需要遍历所有的训练数据集。当训练数据集比较大时，其算法的效率会比较低。考虑另外一个算法：
$$
\theta_{j}=\theta_{j}-\alpha\left(\left(h\left(x^{(i)}\right)-y^{(i)}\right) x_{j}^{(i)}\right)
$$
 这个算法的关键点是把累加器去掉，不去遍历所有的数据集，而是改成每次随机地从训练数据集中取一个数据进行参数迭代计算，这就是随机梯度下降算法（Stochastic Gradient Descent，简称SGD）。随机梯度下降算法可以大大提高模型训练的效率。

### 正规方程

梯度下降算法通过不断地迭代，从而不停地逼近成本函数的最小值来求解模型的参数。另外一个方法是直接计算成本函数的微分，令微分算子为0，求解这个方程，即可得到线性回归的解。
 线性回归算法的损失函数：
$$
J(\theta)=\frac{1}{2 m} \sum_{i=0}^{n}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right)^{2}
$$
成本函数的“斜率”为0的点，即为模型参数的解。令$\frac{\partial}{\partial \theta} J(\theta)=0$，求解这个方程最终可以得到模型参数：
$$
\theta=\left(X^{T} X\right)^{-1} X^{T} y
$$
这就是我们的正规方程。它通过矩阵运算，直接从训练样本里求出参数θ的值。其中X为训练样本的矩阵形式，它是m×n的矩阵，y是训练样本的结果数据，它是个m维列向量。方程求解过程可参阅[百度百科](https://baike.baidu.com/item/%E6%AD%A3%E8%A7%84%E6%96%B9%E7%A8%8B/10001812?fr=aladdin)
