# K-NN算法
## KNN 概述

KNN（K-Nearest Neighbor，K-近邻算法）算法是一种**有监督**的机器学习算法，可以解决分类问题，也可以解决回归问题。

> **一句话总结: 近朱者赤近墨者黑！** 

k -近邻算法的输入为实例的特征向量，对应于特征空间的点；输出为实例的类别，可以取多类。k 近邻算法假设给定一个训练数据集，其中的实例类别已定。分类时，对新的实例，根据其 k 个最近邻的训练实例的类别，通过多数表决等方式进行预测。因此，k近邻算法不具有显式的学习过程。

## KNN算法原理

K-近邻算法的核心思想是未标记样本的类别，由距离其最近的 K 个邻居投票来决定。

假设，我们有一个已经标记的数据集，即已经知道了数据集中每个样本所属的类别。此时，有一个未标记的数据样本，我们的任务是预测出这个数据样本所属的类别。**K-近邻算法的原理是，计算待标记的数据样本和数据集中每个样本的距离，取距离最近的K个样本。**待标记的数据样本所属的类别，就由这K个距离最近的样本投票产生。

> KNN工作原理

假设X_test为待标记的数据样本，X_train为已标记的数据集，算法原理的伪代码如下：

- 遍历 X_train 中的所有样本，计算每个样本与 X_test 的距离，并把距离保存在 Distance 数组中。
- 对 Distance 数组进行排序，取距离最近的K个点，记为 X_knn 。
- 在 X_knn 中统计每个类别的个数，即 class0 在 X_knn 中有几个样本，class1 在 X_knn 中有几个样本等。
- 待标记样本的类别，就是在 X_knn 中样本数最多的那个类别。

> KNN算法优缺点

- 优点：准确度高，对异常值和噪声有较高的容忍度。
- 缺点：计算复杂度高、空间复杂度高，从算法原理可以看出，每次对一个未标记样本进行分类时，都需要全部计算一遍距离。

> KNN算法参数

其算法参数是K，参数选择需要根据数据来决定。K值越大，模型的偏差越大，对噪声数据越不敏感，当K值很大时，可能造成模型欠拟合；K值越小，模型的方差就会越大，当K值太小，就会造成模型过拟合。

> KNN算法变种

K-近邻算法有一些变种，其中之一就是可以增加邻居的权重。默认情况下，在计算距离时，都是使用相同的权重。实际上，我们可以针对不同的邻居指定不同的权重，如距离越近权重越高。这个可以通过指定算法的weights参数来实现。

另外一个变种是，使用一定半径内的点取代距离最近的K个点。在 `scikit-learn` 里，`RadiusNeighborsClassifier` 类实现了这个算法的变种。当数据采样不均匀时，该算法变种可以取得更好的性能。

## KNN 项目案例

### 案例1: 使用KNN算法进行分类

完整代码地址：

在 `scikit-learn`里，使用K-近邻算法进行分类处理的是 `sklearn.neightbors.KNeightborsClassifier` 类。

#### :rainbow:  生成数据集

我们使用 `sklearn.datasets.samples_generator` 包下的 `make_blobs()` 函数来生成数据集，这里生成60个训练样本，这些样本分布在 `centers` 参数指定的中心点的周围。`cluster_std` 是标准差，用来指明生成的点分布的松散程度。生成的训练数据集放在变量X里面，数据集的类别标记放在 y 里面。

```python
from sklearn.datasets import make_blobs
# 生成数据
centers = [[-2,2],[2,2],[0,4]]
X,y = make_blobs(n_samples=60,centers=centers,random_state=0,cluster_std=0.60)
```

> X:  [[ 1.59652373  1.7842681 ] ,[-1.08033247 2.88161526],...]共60个点的横纵坐标
>
> y: [1 0 0 1 0 1 1 0 2...1 2 0 1] 共60个点的类别，用0，1，2分别表示以哪个中心聚合

使用 `matplotlib` 库，它可以很容易地把生成的点画出来：

```python
import matplotlib.pyplot as plt
import numpy as np
plt.figure(figsize=(16,10),dpi=144)
c = np.array(centers)
# cmap就是指matplotlib.colors.Colormap,一个包含三列矩阵的色彩映射表
# 使用c和cmap来映射颜色，s为形状的大小
plt.scatter(X[:,0],X[:,1],c=y,s=100,cmap='cool')
plt.scatter(c[:,0],c[:,1],s=100,marker='*',c='black')
plt.show()
```

<img src="E:\Typora图片\image-20200628191906466.png" alt="image-20200628191906466"  />

这些点的分布情况在坐标轴上一目了然，其中五角星的点即各个类别的中心点。

#### :rainbow:  训练算法

使用 `KNeighborsClassifier` 来对算法进行训练，我们选择的参数是 `K=5`

```pytho
from sklearn.neighbors import KNeighborsClassifier
k = 5
clf = KNeighborsClassifier(n_neighbors=k)
clf.fit(X,y)
```

`KNeighborsClassifier`的参数细节为：

```python
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=5, p=2,
           weights='uniform')
```

#### :rainbow:  对样本进行预测

我们要预测的样本是[0,2]，使用 `kneighbors()` 方法，把这个样本周围距离最近的5个点取出来。取出来的点是训练样本X里的索引，从0开始计算。
 注意：`kneighbors() `接收一个二维数组作为参数，所以 `X_sample` 需要变成二维。

```bash
X_sample = [0,2]
X_sample = np.array(X_sample).reshape(1, -1)  #[[0 2]]
y_sample = clf.predict(X_sample)
neighbors = clf.kneighbors(X_sample,return_distance=False) #[[16 20 48  6 23]]
```

#### :rainbow:  标记最近的5个点

把待预测的样本以及和其最近的5个点标记出来

```bash
plt.figure(figsize=(16,10),dpi=144)
plt.scatter(X[:, 0], X[:, 1], c=y, s=100, cmap='cool')    # 样本
plt.scatter(c[:, 0], c[:, 1], s=100, marker='^', c='k')   # 中心点
plt.scatter(X_sample[0][0],X_sample[0][1],marker="x", s=100, cmap='cool')  #待预测的点
#预测点与距离最近的5个样本的连线
for i in neighbors[0]:
    plt.plot([X[i][0],X_sample[0][0]],[X[i][1],X_sample[0][1]],'k--',linewidth=0.6)
plt.show()
```

![](E:\Typora图片\image-20200628204257514.png)



### 案例2: 使用KNN算法进行回归拟合

分类问题的预测值是离散的，我们也可以使用 KNN 算法对连续区间内的数值进行预测，即进行回归拟合。在`scikit-learn`里面，使用KNN算法进行回归拟合的实现是 `sklearn.neighbors.KNeighborsRegressor` 类。

#### :rainbow:  生成数据集 

在余弦曲线的基础上加入了噪声：

```python
import numpy as np
n_dots = 40
# 生成40行1列的服从“0~5”均匀分布的随机样本
X = 5 * np.random.rand(n_dots, 1)
y = np.cos(X).flatten()
# 生成40行1列的服从“-0.1~0.1”均匀分布的随机误差
y += 0.2 * np.random.rand(n_dots) - 0.1
```

#### :rainbow:  训练算法 

使用 `KNeighborsRegressor` 来训练模型：

```python
from sklearn.neighbors import KNeighborsRegressor
k = 5
knn = KNeighborsRegressor(k)
knn.fit(X,y)
```

`KNeighborsRegressor`方法的参数细节为：

```python
KNeighborsRegressor(algorithm='auto', leaf_size=30, metric='minkowski',
          metric_params=None, n_jobs=1, n_neighbors=5, p=2,
          weights='uniform')
```

可以使用 `score()`方法 计算拟合曲线对训练样本的拟合准确性：

```python
knn.score(X,y)
0.9596828473009764
```

#### :rainbow:  回归拟合

 一个常用的方法是，在X轴上的指定区域生成足够多的点，针对这些足够密集的点，使用训练出来的模型进行预测，得到预测值y_pred，然后在坐标轴上，把所有的预测点连接起来，这样就画出了拟合曲线。
 生成足够密集的点并进行预测：

```python
T = np.linspace(0,5,500)[:,np.newaxis]
y_pred = knn.predict(T)
```

把这些预测点连起来，构成拟合曲线：

```python
plt.figure(figsize=(16,10),dpi=144)
plt.scatter(X,y,c='g',label='data',s=100)
plt.plot(T,y_pred,c='k',label='prediction',lw=4)
plt.axis('tight')
plt.title('KNeighborsRegressor (k = %i)' % k)
plt.show()
```

最终生成的拟合曲线和训练样本数据如图，拟合出来确实和cos曲线相似

<img src="E:\Typora图片\image-20200628223010274.png" style="zoom: 67%;" />



### 案例3: 使用KNN算法预测糖尿病

本节使用KNN算法及其变种，对Pima印第安人的糖尿病进行预测。数据来源[kaggle.com]()大家可以自己去下载。也可以使用 [仓库文件](./pima-indians-diabetes/diabetes.csv)。

####   :rainbow:加载数据集 


















