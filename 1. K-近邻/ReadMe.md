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

<img src="https://cdn.jsdelivr.net/gh/dongzhougu/imageuse1/image-20200628191906466.png" >

这些点的分布情况在坐标轴上一目了然，其中五角星的点即各个类别的中心点。

#### :rainbow:  训练算法

使用 `KNeighborsClassifier` 来对算法进行训练，我们选择的参数是 `K=5`

```python
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

```python
X_sample = [0,2]
X_sample = np.array(X_sample).reshape(1, -1)  #[[0 2]]
y_sample = clf.predict(X_sample)
neighbors = clf.kneighbors(X_sample,return_distance=False) #[[16 20 48  6 23]]
```

#### :rainbow:  标记最近的5个点

把待预测的样本以及和其最近的5个点标记出来

```python
plt.figure(figsize=(16,10),dpi=144)
plt.scatter(X[:, 0], X[:, 1], c=y, s=100, cmap='cool')    # 样本
plt.scatter(c[:, 0], c[:, 1], s=100, marker='^', c='k')   # 中心点
plt.scatter(X_sample[0][0],X_sample[0][1],marker="x", s=100, cmap='cool')  #待预测的点
#预测点与距离最近的5个样本的连线
for i in neighbors[0]:
    plt.plot([X[i][0],X_sample[0][0]],[X[i][1],X_sample[0][1]],'k--',linewidth=0.6)
plt.show()
```

![](https://cdn.jsdelivr.net/gh/dongzhougu/imageuse1/image-20200628204257514.png)



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

最终生成的拟合曲线和训练样本数据如图，拟合出来确实和 cos 曲线相似。

<img src="https://cdn.jsdelivr.net/gh/dongzhougu/imageuse1/image-20200628223010274.png" style="zoom: 67%;" />



### 案例3: 使用KNN算法预测糖尿病

本节使用KNN算法及其变种，对Pima印第安人的糖尿病进行预测。数据来源[kaggle.com]()大家可以自己去下载。也可以使用 [仓库文件](./pima-indians-diabetes/diabetes.csv)。

####   :rainbow:加载数据集 

使用Pandas加载数据：

```python
import pandas as pd
data = pd.read_csv('./pima-indians-diabetes/diabetes.csv')
print('dataset shape {}'.format(data.shape))
print(data.head())
```

输出如下：

```python
dataset shape (768, 9)
Out[23]:
Pregnancies Glucose BloodPressure   SkinThickness   Insulin BMI DiabetesPedigreeFunction    Age Outcome
0   6   148 72  35  0   33.6    0.627   50  1
1   1   85  66  29  0   26.6    0.351   31  0
2   8   183 64  0   0   23.3    0.672   32  1
3   1   89  66  23  94  28.1    0.167   21  0
4   0   137 40  35  168 43.1    2.288   33  1
```

从打印出的信息可以看到，这个数据集一共有 768 个样本、8 个特征、1 个标签：

`Pregnancies`：怀孕的次数

`Glucose`：血浆葡萄糖浓度，采用 2 小时口服葡萄糖耐量试验测得

`BloodPressure`：舒张压（毫米汞柱）

`SkinThickness`：肱三头肌皮肤褶皱厚度（毫米）

`Insulin`：两个小时血清胰岛素（ μU /毫升）

`BMI`：身体质量指数，体重除以身高的平方

`DiabetesPedigreeFunction`：糖尿病血统指数，糖尿病和家庭遗传相关

`Age`：年龄

`Outcome`：0表示没有糖尿病，1表示有糖尿病

 我们可以进一步观察数据集里的阳性和阴性样本的个数：

```python
data.groupby('Outcome').size()
```

输出为：

```python
Outcome
0    500
1    268
dtype: int64
```

其中，阴性样本500例，阳性样本268例。

#### :rainbow: 处理数据集 

 接着需要对数据集进行简单处理，把8个特征值分离出来，作为训练数据集，把Outcome列分离出来作为目标值。然后，把数据集划分为训练数据集和测试数据集。

```python
X = data.iloc[:,:8]
Y = data.iloc[:,8]
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
```

输出：

```python
(614, 8) (154, 8) (614,) (154,)
```

#### :rainbow: 模型比较 

分别使用普通的KNN算法、带权重的KNN算法和指定半径的KNN算法对数据集进行拟合并计算评分：

```python
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier

models = []
models.append(("KNN", KNeighborsClassifier(n_neighbors=2)))
models.append(("KNN with weights", KNeighborsClassifier(
    n_neighbors=2, weights="distance")))
models.append(("Radius Neighbors", RadiusNeighborsClassifier(
    n_neighbors=2, radius=500.0)))

results = []
for name, model in models:
    model.fit(X_train, Y_train)
    results.append((name, model.score(X_test, Y_test)))
for i in range(len(results)):
    print("name: {}; score: {}".format(results[i][0],results[i][1]))
```

三种算法的性能如下：

```python
name: KNN; score: 0.7467532467532467
name: KNN with weights; score: 0.6818181818181818
name: Radius Neighbors; score: 0.6558441558441559
```

带权重的KNN算法，我们选择了距离越近、权重越高。指定半径的KNN算法的半径选择了500。从上面的输出结果可以看出，普通的KNN算法性能最好。问题来了，这个判断准确么？答案是不准确。因为我们的训练样本和测试样本是随机分配的，不同的训练样本和测试样本组合可能导致计算出来的算法准确性是有差异的。我们可以试着多次运行上面的代码，观察输出值是否有变化。

怎么样更准确地对比算法准确性呢？一个方法是，多次随机分配训练数据集和交叉验证数据集，然后求模型准确性评分的平均值。所幸，我们不需要从头实现这个过程，`scikit-learn` 提供了 `KFold` 和 `cross_val_score()`函数来处理这种问题：

```python
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
results = []
for name, model in models:
    kfold = KFold(n_splits=10)
    cv_result = cross_val_score(model, X, Y, cv=kfold)
    results.append((name, cv_result))
for i in range(len(results)):
    print("name: {}; cross val score: {}".format(
        results[i][0],results[i][1].mean()))
```

上述代码中，我们通过KFold把数据集分成10份，其中1份会作为交叉验证数据集来计算模型准确性，剩余的9份作为训练数据集。cross_val_score()函数总共计算出10次不同训练数据集和交叉验证数据集组合得到的模型准确性评分，最后求平均值。这样的评价结果相对更准确一些。
 输出结果为：

```python
name: KNN; cross val score: 0.7147641831852358
name: KNN with weights; cross val score: 0.6770505809979495
name: Radius Neighbors; cross val score: 0.6497265892002735
```

#### :rainbow:模型训练及分析

通过上面的对比来看，普通的KNN算法性能更优一些。接下来，我们就使用普通的KNN算法模型对数据集进行训练，并查看对训练样本的拟合情况以及对测试样本的预测准确性情况：

```python
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, Y_train)
train_score = knn.score(X_train, Y_train)
test_score = knn.score(X_test, Y_test)
print("train score: {}\ntest score: {}".format(train_score, test_score))
```

输出结果为：

```python
train score: 0.8387622149837134
test score: 0.7337662337662337
```

从输出中可以看到两个问题。一是对训练样本的拟合情况不佳，评分才0.82多，这说明算法模型太简单了，无法很好地拟合训练样本。二是模型的准确性欠佳，不到74%的预测准确性。我们可以进一步画出学习曲线，证实结论。

```python
from sklearn.model_selection import ShuffleSplit
from common.utils import plot_learning_curve
knn = KNeighborsClassifier(n_neighbors=2)
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
plt.figure(figsize=(10, 6))
plot_learning_curve(plt, knn, "Learn Curve for KNN Diabetes", 
                    X, Y, ylim=(0.0, 1.01), cv=cv)
plt.show()
```

<img src="https://cdn.jsdelivr.net/gh/dongzhougu/imageuse1/image-20200629101150911.png"  style="zoom:67%;" />

从图中可以看出来，训练样本评分较低，且测试样本与训练样本距离较大，这是典型的欠拟合现象。KNN算法没有更好的措施来解决欠拟合问题，我们学完本书的其他章节后，可以试着用其他算法（如逻辑回归算法、支持向量机等）来对比不同模型的准确性情况。

#### :rainbow:特征选择及数据可视化

那有没有直观的方法，来揭示出为什么KNN算法不是针对这一问题的好模型？一个办法是把数据画出来，可是我们有8个特征，无法在这么高的维度里画出数据，并直观地观察。一个解决办法是特征选择，即只选择2个与输出值相关性最大的特征，这样就可以在二维平面上画出输入特征值与输出值的关系了。

`scikit-learn `在 `sklearn.feature_selection` 包里提供了丰富的特征选择方法。我们使用 `SelectKBest` 来选择相关性最大的两个特征：

```python
from sklearn.feature_selection import SelectKBest
selector = SelectKBest(k=2)
X_new = selector.fit_transform(X, Y)
print('X_new.shape {}'.format(X_new.shape))
```

把相关性最大的两个特征放在X_new变量里，输出结果为：

```python
X_new.shape (768, 2)
```

我们可能会好奇，相关性最大的特征到底是哪两个？对比一下本节开头的数据即可知道，它们分别是Glucose（血糖浓度）和BMI（身体质量指数）。血糖浓度和糖尿病的关系自不必说，身体质量指数是反映肥胖程度的指标，从业务角度来看，我们选择出来的2个相关性最高的特征还算合理。那么 `SelectKBest` 到底使用什么神奇的方法选择出了这两个相关性最高的特征呢？详情参考下一节。

我们来看看，如果只使用这2个相关性最高的特征的话，3种不同的KNN算法哪个准确性更高：

```python
results = []
for name, model in models:
    kfold = KFold(n_splits=10)
    cv_result = cross_val_score(model, X_new, Y, cv=kfold)
    results.append((name, cv_result))
for i in range(len(results)):
    print("name: {}; cross val score: {}".format(
        results[i][0],results[i][1].mean()))
```

这次使用X_new作为输入，输出如下

```python
name: KNN; cross val score: 0.725205058099795
name: KNN with weights; cross val score: 0.6900375939849623
name: Radius Neighbors; cross val score: 0.6510252904989747
```

从输出可以看出来，还是普通的KNN模型准确性较高，其准确性也达到了将近 73 %，与所有特征拿来一块儿训练的准确性差不多。这也侧面证明了 `SelectKBest` 特征选择的有效性。

回到目标上来，我们是想看看为什么KNN无法很好地拟合训练样本。现在我们只有 2 个特征，可以很方便地在二维坐标上画出所有的训练样本，观察这些数据的分布情况：

```python
plt.figure(figsize=(10, 6))
plt.ylabel("BMI")
plt.xlabel("Glucose")
plt.scatter(X_new[Y==0][:, 0], X_new[Y==0][:, 1], c='r', s=20, marker='o');   #画出样本
plt.scatter(X_new[Y==1][:, 0], X_new[Y==1][:, 1], c='g', s=20, marker='^');   #画出样本
```

<img src="https://cdn.jsdelivr.net/gh/dongzhougu/imageuse1/image-20200629102028938.png"  style="zoom:67%;" />

横坐标是血糖值 Glucose，纵坐标是BMI值，反映身体肥胖情况。从图中可以看出，在中间数据集密集的区域，阳性样本和阴性样本几乎重叠在一起了。假设现在有一个待预测的样本在中间密集区域，它的阳性邻居多还是阴性邻居多呢？这真的很难说。这样就可以直观地看到，KNN算法在这个糖尿病预测问题上，无法达到很高的预测准确性。

### 拓展阅读

这里再继续再介绍一下特征选择时，计算相关性大小的 `SelectKBest()` 函数背后的统计学知识。

#### 如何提高KNN算法的运算效率

根据算法原理，每次需要预测一个点时，我们都需要计算训练数据集里每个点到这个点的距离，然后选出距离最近的k个点进行投票。当数据集很大时，这个计算成本非常高。针对$N$个样本，$D$个特征的数据集，其算法复杂度为$O(DN^2)$。

为了解决这个问题，一种叫`K-D Tree` 的数据结构被发明出来。为了避免每次都重新计算一遍距离，算法会把距离信息保存在一棵树里，这样在计算之前从树里查询距离信息，尽量避免重新计算。其基本原理是，如果A和B距离很远，B和C距离很近，那么A和C的距离也很远。有了这个信息，就可以在合适的时候跳过距离远的点。这样优化后的算法复杂度可降低到$O(DNlog(N))$。感兴趣的读者可参阅论文：Bentley, J.L., Communications of the ACM (1975)。

1989年，另外一种称为`Ball Tree`的算法，在`K-D Tree`的基础上对性能进一步进行了优化。感兴趣的读者可以搜索Five balltree construction algorithms来了解详细的算法信息。

#### 相关性测试

先通过一个简单的例子来看假设检验问题，即判断假设的结论是否成立或成立的概率有多高。假设，在一个城市随机采样到程序员和性别的关系的数据：

<img src="https://cdn.jsdelivr.net/gh/dongzhougu/imageuse1/17634123-0c82f9f4005936fa.png"  style="zoom:50%;" />

假设，我们的结论是程序员和性别无关，这个假设称为原假设（null hypothesis）。问：通过我们随机采样观测到的数据，原假设是否成立，或者说原假设成立的概率有多高？

`卡方检验（chi-squared test）`是检测假设成立与否的一个常用的工具。它的计算公式是：

<img src="https://cdn.jsdelivr.net/gh/dongzhougu/imageuse1/17634123-3a52b648ce9e2196.png" style="zoom:50%;" />

其中，卡方检验的值记为 ,  $O$ 是观测值，$E$  是期望值。针对我们的例子，如果原假设成立，即程序员职业和性别无关，那么我们期望的男程序员数量应该为(14/489) * 242=6.928，女程序员数量应该为(14/489) * 247=7.072，同理可得到我们的期望值如下：

<img src="https://cdn.jsdelivr.net/gh/dongzhougu/imageuse1/17634123-9f2c88dc6a7eeb66.png" style="zoom:50%;" />


 根据卡方检验的公式，可以算出卡方值为：

<img src="https://cdn.jsdelivr.net/gh/dongzhougu/imageuse1/17634123-bf0dbd2fd6c8db0c.png" style="zoom:50%;" />

 算出卡方值后，怎么判断原假设成立的概率是多少呢？这里还涉及到自由度和卡方分布的概念。简单地讲，自由度是$(r-1)×(c-1)$，其中 r 是行数，c 是列数，针对我们的问题，其自由度为1。卡方分布是指，若n个相互独立的随机变量均服从正态分布，则这 n 个随机变量的平方和构成一新的随机变量，其分布规律称为卡方分布。卡方分布的密度函数和自由度相关，知道了自由度和目标概率，我们就能求出卡方值。

针对我们的问题，可以查表得到，自由度为1的卡方分布，在99%处的卡方值为6.63。我们计算出来的卡方值为7.670。由于7.67>6.63，故有99%的把握可以推翻原假设。换个说法，如果原假设成立，即程序员职业和性别无关，那么我们随机采样到的数据出现的概率将低于1%。我们可以搜索`“卡方表”`或`“Chi Squared Table”`找到不同自由度对应的卡方值。

卡方值的大小可以反映变量与目标值的相关性，值越大，相关性越大。利用这一特性，`SelectKBest()` 函数就可以计算不同特征的卡方值来判断特征与输出值的相关性大小，从而完成特征选择。在`scikit-learn`里，计算卡方值的函数是 `sklearn.feature_selection.chi2()`。除了卡方检验外，还有`F值检验`等算法，也可以用来评估特征与目标值的相关性。`SelectKBest` 默认使用的就是F值检验算法，在`scikit-learn`里，使用`sklearn.feature_selection.f_classif`来计算F值。关于F值相关的资料，感兴趣的读者可以在英文版维基百科上搜索`“Fisher’sexact test”`，了解更多信息。

