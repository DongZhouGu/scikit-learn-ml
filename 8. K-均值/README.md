# K-Means聚类算法

聚类，简单来说，就是将一个庞杂数据集中具有相似特征的数据自动归类到一起，称为一个簇，簇内的对象越相似，聚类的效果越好。它是一种无监督的学习(Unsupervised Learning)方法,不需要预先标注好的训练集。聚类与分类最大的区别就是分类的目标事先已知，例如猫狗识别，你在分类之前已经预先知道要将它分为猫、狗两个种类；而在你聚类之前，你对你的目标是未知的，同样以动物为例，对于一个动物集来说，你并不清楚这个数据集内部有多少种类的动物，你能做的只是利用聚类方法将它自动按照特征分为多类，然后人为给出这个聚类结果的定义（即簇识别）。例如，你将一个动物集分为了三簇（类），然后通过观察这三类动物的特征，你为每一个簇起一个名字，如大象、狗、猫等，这就是聚类的基本思想。     

至于“相似”这一概念，是利用距离这个评价标准来衡量的，我们通过计算对象与对象之间的距离远近来判断它们是否属于同一类别，即是否是同一个簇。至于距离如何计算，科学家们提出了许多种距离的计算方法，其中欧式距离是最为简单和常用的，除此之外还有曼哈顿距离和余弦相似性距离等。

欧式距离，我想大家再熟悉不过了，但为免有一些基础薄弱的同学，在此再说明一下，它的定义为:   
对于x点坐标为(x1,x2,x3,...,xn)和 y点坐标为(y1,y2,y3,...,yn)，两者的欧式距离为:
$$
d(x,y)
    ={\sqrt{
            (x_{1}-y_{1})^{2}+(x_{2}-y_{2})^{2} + \cdots +(x_{n}-y_{n})^{2}
        }}
    ={\sqrt{
            \sum_{ {i=1} }^{n}(x_{i}-y_{i})^{2}
        }}
$$

在二维平面，它就是我们初中时就学过的两点距离公式

## 1. K-Means 算法

K-Means 是发现给定数据集的 K 个簇的聚类算法, 之所以称之为 `K-均值` 是因为它可以发现 K 个不同的簇, 且每个簇的中心采用簇中所含值的均值计算而成. 
簇个数 K 是用户指定的, 每一个簇通过其质心（centroid）, 即簇中所有点的中心来描述. 
聚类与分类算法的最大区别在于, 分类的目标类别已知, 而聚类的目标类别是未知的. 

**优点**:

* 属于无监督学习，无须准备训练集
* 原理简单，实现起来较为容易
* 结果可解释性较好

**缺点**:

* **需手动设置k值**。 在算法开始预测之前，我们需要手动设置k值，即估计数据大概的类别个数，不合理的k值会使结果缺乏解释性
* 可能收敛到局部最小值, 在大规模数据集上收敛较慢
* 对于异常点、离群点敏感

使用数据类型 : 数值型数据


### 1.1 K-Means 场景

kmeans，如前所述，用于数据集内种类属性不明晰，希望能够通过数据挖掘出或自动归类出有相似特点的对象的场景。其商业界的应用场景一般为挖掘出具有相似特点的潜在客户群体以便公司能够重点研究、对症下药。  

例如，在2000年和2004年的美国总统大选中，候选人的得票数比较接近或者说非常接近。任一候选人得到的普选票数的最大百分比为50.7%而最小百分比为47.9% 如果1%的选民将手中的选票投向另外的候选人，那么选举结果就会截然不同。 实际上，如果妥善加以引导与吸引，少部分选民就会转换立场。尽管这类选举者占的比例较低，但当候选人的选票接近时，这些人的立场无疑会对选举结果产生非常大的影响。如何找出这类选民，以及如何在有限的预算下采取措施来吸引他们？ 答案就是聚类（Clustering)。

那么，具体如何实施呢？首先，收集用户的信息，可以同时收集用户满意或不满意的信息，这是因为任何对用户重要的内容都可能影响用户的投票结果。然后，将这些信息输入到某个聚类算法中。接着，对聚类结果中的每一个簇（最好选择最大簇 ）， 精心构造能够吸引该簇选民的消息。最后， 开展竞选活动并观察上述做法是否有效。

另一个例子就是产品部门的市场调研了。为了更好的了解自己的用户，产品部门可以采用聚类的方法得到不同特征的用户群体，然后针对不同的用户群体可以对症下药，为他们提供更加精准有效的服务。

### 1.2 K-Means 术语

* 簇: 所有数据的点集合，簇中的对象是相似的。
* 质心: 簇中所有点的中心（计算所有点的均值而来）.
* SSE: Sum of Sqared Error（误差平方和）, 它被用来评估模型的好坏，SSE 值越小，表示越接近它们的质心. 聚类效果越好。由于对误差取了平方，因此更加注重那些远离中心的点（一般为边界点或离群点）。详情见kmeans的评价标准。

有关 `簇` 和 `质心` 术语更形象的介绍, 请参考下图:

![K-Means 术语图](https://cdn.jsdelivr.net/gh/dongzhougu/imageuse1/apachecn-k-means-term-1.jpg)

### 1.3 K-Means 工作流程

1. 首先, 随机确定 K 个初始点作为质心（**不必是数据中的点**）。
2. 然后将数据集中的每个点分配到一个簇中, 具体来讲, 就是为每个点找到距其最近的质心, 并将其分配该质心所对应的簇. 这一步完成之后, 每个簇的质心更新为该簇所有点的平均值.
   3.重复上述过程直到数据集中的所有点都距离它所对应的质心最近时结束。

上述过程的 `伪代码` 如下:

* 创建 k 个点作为起始质心（通常是随机选择）
* 当任意一个点的簇分配结果发生改变时（不改变时算法结束）
  * 对数据集中的每个数据点
    * 对每个质心
      * 计算质心与数据点之间的距离
    * 将数据点分配到距其最近的簇
  * 对每一个簇, 计算簇中所有点的均值并将均值作为质心

## 2. sklearn 里的K-均值算法

cikit-learn里的K-均值算法由sklearn.cluster.KMeans类实现。下面通过一个简单的例子，来学习怎样在scikit-learn里使用K-均值算法。

我们生成一组包含两个特征的200个样本：

```python
from sklearn.datasets import make_blobs
X,y = make_blobs(n_samples=200,
                n_features=2,
                centers=4,
                cluster_std=1,
                center_box=(-10.0,10.0),
                shuffle=True,
                random_state=1)
```

然后把样本画在二维坐标系上，以便直观地观察：

```python
import matplotlib.pyplot as plt
plt.figure(figsize=(6,4),dpi=144)
plt.xticks(())
plt.yticks(())
plt.scatter(X[:,0],X[:,1],s=20,marker='o')
```

结果如图所示：

![image-20200716173322275](https://cdn.jsdelivr.net/gh/dongzhougu/imageuse1/image-20200716173322275.png)

接着使用 `KMeans` 模型来拟合。我们设置类别个数为3，并计算出其拟合后的成本。

```python
from sklearn.cluster import KMeans
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(X)
print("kmeans: k = {}, cost = {}".format(n_clusters,int(kmeans.score(X))))
```

输出如下：

```python
kmeans: k = 3, cost = -668
```

`KMeans.score()` 函数计算K-均值算法拟合后的成本，用负数表示，其绝对值越大，说明成本越高。前面介绍过，K-均值算法成本的物理意义为训练样本到其所属的聚类中心的距离平均值，在 `scikit-learn` 里，其计算成本的方法略有不同，它是计算训练样本到其所属的聚类中心的距离的总和。

当然我们还可以把分类后的样本及其所属的聚类中心都画出来，这样可以更直观地观察算法的拟合效果。

```python
labels = kmean.labels_
centers = kmean.cluster_centers_
markers = ['o', '^', '*']
colors = ['r', 'b', 'y']

plt.figure(figsize=(6,4), dpi=144)
plt.xticks(())
plt.yticks(())

# 画样本
for c in range(n_clusters):
    cluster = X[labels == c]
    plt.scatter(cluster[:, 0], cluster[:, 1], 
                marker=markers[c], s=20, c=colors[c])
# 画出中心点
plt.scatter(centers[:, 0], centers[:, 1],
            marker='o', c="white", alpha=0.9, s=300)
for i, c in enumerate(centers):
    plt.scatter(c[0], c[1], marker='$%d$' % i, s=50, c=colors[i])
```

输出结果如图所示：

![image-20200716173439992](https://cdn.jsdelivr.net/gh/dongzhougu/imageuse1/image-20200716173439992.png)

前面说过，K-均值算法的一个关键参数是K，即聚类个数。从技术角度来讲，K值越大，算法成本越低，这个很容易理解。但从业务角度来看，不是K值越大越好。针对本节的例子，分别选择K=[2,3,4]这三种不同的聚类个数，来观察一下K-均值算法最终拟合的结果及其成本。

我们可以把画出K-均值聚类结果的代码稍微改造一下，变成一个函数。这个函数会使用K-均值算法来进行聚类拟合，同时会画出按照这个聚类个数拟合后的分类情况：

```python
def fit_plot_kmean_model(n_clusters, X):
    plt.xticks(())
    plt.yticks(())

    # 使用 k-均值算法进行拟合
    kmean = KMeans(n_clusters=n_clusters)
    kmean.fit_predict(X)

    labels = kmean.labels_
    centers = kmean.cluster_centers_
    markers = ['o', '^', '*', 's']
    colors = ['r', 'b', 'y', 'k']

    # 计算成本
    score = kmean.score(X)
    plt.title("k={}, score={}".format(n_clusters, (int)(score)))

    # 画样本
    for c in range(n_clusters):
        cluster = X[labels == c]
        plt.scatter(cluster[:, 0], cluster[:, 1], 
                    marker=markers[c], s=20, c=colors[c])
    # 画出中心点
    plt.scatter(centers[:, 0], centers[:, 1],
                marker='o', c="white", alpha=0.9, s=300)
    for i, c in enumerate(centers):
        plt.scatter(c[0], c[1], marker='$%d$' % i, s=50, c=colors[i])
```

函数接受两个参数，一个是聚类个数，即K的值，另一个是数据样本。有了这个函数，接下来就简单了，可以很容易分别对[2,3,4]这三种不同的K值情况进行聚类分析，并把聚类结果可视化。

```python
from sklearn.cluster import KMeans

n_clusters = [2, 3, 4]
plt.figure(figsize=(10, 3), dpi=144)
for i, c in enumerate(n_clusters):
    plt.subplot(1, 3, i + 1)
    fit_plot_kmean_model(c, X)
```


![image-20200716175104911](https://cdn.jsdelivr.net/gh/dongzhougu/imageuse1/image-20200716175104911.png)

## 3. 使用K-均值对文档进行聚类分析

本节介绍如何使用K-均值算法对文档进行聚类分析。假设有一个博客平台，用户在平台上发布博客，我们如何对博客进行聚类分析，以方便展示不同类别下的热门文章呢？

### 3.1 加载数据集

我们的任务就是把数据集目录下`data/`的文档进行聚类分析。你可能有疑问：这些文档不是按照文件夹已经分好类了吗？是的，这是人工标记了的数据。有了人工标记的数据，就可以检验K-均值算法的性能。

首先需要导入数据：

```go
from time import time
from sklearn.datasets import load_files
print("loading documents ...")
t = time()
docs = load_files('datasets/clustering/data')
print("summary: {0} documents in {1} categories.".format(
    len(docs.data), len(docs.target_names)))
print("done in {0} seconds".format(time() - t))
```

输出如下：

```bash
loading documents ...
summary: 3949 documents in 4 categories.
done in 26.920000076293945 seconds
```

总共有3949篇文章，人工标记在4个类别里。接着把文档转化为TF-IDF向量：

```python
from sklearn.feature_extraction.text import TfidfVectorizer

max_features = 20000
print("vectorizing documents ...")
t = time()
vectorizer = TfidfVectorizer(max_df=0.4, 
                             min_df=2, 
                             max_features=max_features, 
                             encoding='latin-1')
X = vectorizer.fit_transform((d for d in docs.data))
print("n_samples: %d, n_features: %d" % X.shape)
print("number of non-zero features in sample [{0}]: {1}".format(
    docs.filenames[0], X[0].getnnz()))
print("done in {0} seconds".format(time() - t))
```

这里需要注意TfidfVectorizer的几个参数的选择。max_df=0.4表示如果一个单词在40%的文档里都出现过，则认为是一个高频词，对文档聚类没有帮助，在生成词典时就会剔除这个词。min_df=2表示，如果一个单词的词频太低，小于等于2个，则也把这个单词从词典里剔除。max_features可以进一步过滤词典的大小，它会根据TF-IDF权重从高到低进行排序，然后取前面权重高的单词构成词典。输出如下：

```css
vectorizing documents ...
n_samples: 3949, n_features: 20000
number of non-zero features in sample [datasets/clustering/data\sci.electronics\11902-54322]: 56
done in 1.9150002002716064 seconds
```

从输出可知，每篇文章构成的向量都是一个稀疏向量，其大部分元素都为0。这也容易理解，我们的词典大小为20000个词，而示例文章中不重复的单词却只有56个。

### 3.2 文本聚类分析

接着使用KMeans算法对文档进行聚类分析：

```dart
from sklearn.cluster import KMeans

print("clustering documents ...")
t = time()
n_clusters = 4
kmean = KMeans(n_clusters=n_clusters, 
               max_iter=100,
               tol=0.01,
               verbose=1,
               n_init=3)
kmean.fit(X);
print("kmean: k={}, cost={}".format(n_clusters, int(kmean.inertia_)))
print("done in {0} seconds".format(time() - t))
```

选择聚类个数为4个。max_iter=100表示最多进行100次K-均值迭代。tol=0.1表示中心点移动距离小于0.1时就认为算法已经收敛，停止迭代。verbose=1表示输出迭代过程的详细信息。n_init=3表示进行3遍K-均值运算后求平均值。前面介绍过，在算法刚开始迭代时，会随机选择聚类中心点，不同的中心点可能导致不同的收敛效果，因此多次运算求平均值的方法可以提高算法的稳定性。由于开启了迭代过程信息显示，输出了较多的信息：

```bash
clustering documents ...
Initialization complete
Iteration  0, inertia 7488.362
Iteration  1, inertia 3845.708
Iteration  2, inertia 3835.369
Iteration  3, inertia 3828.959
Iteration  4, inertia 3824.555
Iteration  5, inertia 3820.932
Iteration  6, inertia 3818.555
Iteration  7, inertia 3817.377
Iteration  8, inertia 3816.317
Iteration  9, inertia 3815.570
Iteration 10, inertia 3815.351
Iteration 11, inertia 3815.234
Iteration 12, inertia 3815.181
Iteration 13, inertia 3815.151
Iteration 14, inertia 3815.136
Iteration 15, inertia 3815.120
Iteration 16, inertia 3815.113
Iteration 17, inertia 3815.106
Iteration 18, inertia 3815.104
Converged at iteration 18: center shift 0.000000e+00 within tolerance 4.896692e-07
Initialization complete
Iteration  0, inertia 7494.329
Iteration  1, inertia 3843.474
Iteration  2, inertia 3835.570
Iteration  3, inertia 3828.511
Iteration  4, inertia 3823.826
Iteration  5, inertia 3819.972
Iteration  6, inertia 3817.714
Iteration  7, inertia 3816.666
Iteration  8, inertia 3816.032
Iteration  9, inertia 3815.778
Iteration 10, inertia 3815.652
Iteration 11, inertia 3815.548
Iteration 12, inertia 3815.462
Iteration 13, inertia 3815.424
Iteration 14, inertia 3815.411
Iteration 15, inertia 3815.404
Iteration 16, inertia 3815.402
Converged at iteration 16: center shift 0.000000e+00 within tolerance 4.896692e-07
Initialization complete
Iteration  0, inertia 7538.349
Iteration  1, inertia 3844.796
Iteration  2, inertia 3828.820
Iteration  3, inertia 3822.973
Iteration  4, inertia 3821.341
Iteration  5, inertia 3820.164
Iteration  6, inertia 3819.181
Iteration  7, inertia 3818.546
Iteration  8, inertia 3818.167
Iteration  9, inertia 3817.975
Iteration 10, inertia 3817.862
Iteration 11, inertia 3817.770
Iteration 12, inertia 3817.723
Iteration 13, inertia 3817.681
Iteration 14, inertia 3817.654
Iteration 15, inertia 3817.628
Iteration 16, inertia 3817.607
Iteration 17, inertia 3817.593
Iteration 18, inertia 3817.585
Iteration 19, inertia 3817.580
Converged at iteration 19: center shift 0.000000e+00 within tolerance 4.896692e-07
kmean: k=4, cost=3815
done in 39.484999895095825 seconds
```

我们好奇的是：在进行聚类分析的过程中，哪些单词的权重最高，从而较容易地决定一个文章的类别？我们可以查看每种类别文档中，其权重最高的10个单词分别是什么？

```python
from __future__ import print_function
print("Top terms per cluster:")
order_centroids = kmean.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(n_clusters):
    print("Cluster %d:" % i, end='')
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind], end='')
    print()
```

理解这段代码的关键在于argsort()函数，它的作用是把一个Numpy数组进行升序排列，返回的是排序后的索引。

回到我们的代码里，由于 kmean.cluster_centers 是二维数组，因此 kmean.cluster_centers.argsort()[:,::-1] 语句的含义就是把聚类中心点的不同分量，按照从大到小的顺序进行排序，并且把排序后的元素索引保存在二维数组order_centroids里。vectorizer.get_feature_names()将得到我们的词典单词，根据索引即可得到每个类别里权重最高的那些单词了。输出如下：

```csharp
Top terms per cluster:
Cluster 0: space henry nasa toronto moon pat zoo shuttle gov orbit
Cluster 1: my any me by know your some do so has
Cluster 2: key clipper encryption chip government will keys escrow we nsa
Cluster 3: geb pitt banks gordon shameful dsl n3jxp chastity cadre surrender
```

### 4.聚类算法性能评估

聚类性能评估比较复杂，不像分类那样直观。针对分类问题，我们可以直接计算被错误分类的样本数量，这样可以直接算出分类算法的准确率。聚类问题不能使用绝对数量的方法进行性能评估，原因是，聚类分析后的类别与原来已标记的类别之间不存在必然的一一对应关系。更典型的，针对K-均值算法，我们可以选择K的数值不等于已标记的类别个数。

前面介绍决策树的时候简单介绍过“熵”的概念，它是信息论中最重要的基础概念。熵表示一个系统的有序程度，而聚类问题的性能评估，就是对比经过聚类算法处理后的数据的有序程度，与人工标记的有序程度之间的差异。下面介绍几个常用的聚类算法性能评估指标
###### **1.Adjust Rand Index**

Adjust Rand Index是一种衡量两个序列相似性的算法。它的优点是，针对两个随机序列，它的值为负数或接近0。而针对两个结构相同的序列，它的值接近1。而且对类别标签不敏感。

###### **2.齐次性和完整性**

根据条件熵分析，可以得到另外两个衡量聚类算法性能的指标，分别是齐次性（homogeneity）和完整性（completeness）。齐次性表示一个聚类元素只由一种类别的元素组成。完整性表示给定的已标记的类别，全部分配到一个聚类里。它们的值均介于[0,1]之间。

###### **3.轮廓系数**

上面介绍的聚类性能评估方法都需要有已标记的类别数据，这个在实践中是很难做到的。如果已经标记了数据，就会直接使用有监督的学习算法，而无监督学习算法的最大优点就是不需要对数据集进行标记。轮廓系数可以在不需要已标记的数据集的前提下，对聚类算法的性能进行评估。

轮廓系数由以下两个指标构成：

- a：一个样本与其所在相同聚类的点的平均距离；
- b：一个样本与其距离最近的下一个聚类里的点的平均距离。

针对这个样本，其轮廓系数s的值为：
$$
s=\frac{b-a}{\max (a, b)}
$$
针对一个数据集，其轮廓系数s为其所有样本的轮廓系数的平均值。轮廓系数的数值介于[-1,1]之间，-1表示完全错误的聚类，1表示完美的聚类，0表示聚类重叠。

针对前面的例子，可以分别计算本节介绍的几个聚类算法性能评估指标，综合来看聚类算法的性能：

```python
from sklearn import metrics
labels = docs.target
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, kmean.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, kmean.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(labels, kmean.labels_))
print("Adjusted Rand-Index: %.3f"
      % metrics.adjusted_rand_score(labels, kmean.labels_))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, kmean.labels_, sample_size=1000))
```

输出如下：

```css
Homogeneity: 0.459
Completeness: 0.519
V-measure: 0.487
Adjusted Rand-Index: 0.328
Silhouette Coefficient: 0.004
```

可以看到模型性能很一般。可能的一个原因是数据集质量不高，当然我们也可以阅读原始的语料库，检验一下如果通过人工标记，是否能够标记出这些文章的正确分类。另外，针对my、any、me、by、know、your、some、do、so、has，这些都是没有特征的单词，即使人工标记，也无法判断这些单词应该属于哪种类别的文章。

