# 朴素贝叶斯算法

## 1. 朴素贝叶斯概述 

贝叶斯分类是一类分类算法的总称，这类算法均以贝叶斯定理为基础，故统称为贝叶斯分类。朴素贝叶斯（Naive Bayers）算法是一种基于概率统计的分类方法。它在条件独立假设的基础上，使用贝叶斯定理构建算法，在文本处理领域有广泛的应用。

## 2. 贝叶斯理论 & 条件概率

### 2.1 贝叶斯理论

我们现在有一个数据集，它由两类数据组成，数据分布如下图所示: 

![朴素贝叶斯示例数据分布](https://cdn.jsdelivr.net/gh/dongzhougu/imageuse1/朴素贝叶斯示例数据分布.png "参数已知的概率分布")

我们现在用 p1(x,y) 表示数据点 (x,y) 属于类别 1（图中用圆点表示的类别）的概率，用 p2(x,y) 表示数据点 (x,y) 属于类别 2（图中三角形表示的类别）的概率，那么对于一个新数据点 (x,y)，可以用下面的规则来判断它的类别: 

* 如果 p1(x,y) > p2(x,y) ，那么类别为1
* 如果 p2(x,y) > p1(x,y) ，那么类别为2

也就是说，我们会选择高概率对应的类别。这就是贝叶斯决策理论的核心思想，即选择具有最高概率的决策。

### 2.2 条件概率

如果你对 p(x,y|c1) 符号很熟悉，那么可以跳过本小节。

有一个装了 7 块石头的罐子，其中 3 块是白色的，4 块是黑色的。如果从罐子中随机取出一块石头，那么是白色石头的可能性是多少？由于取石头有 7 种可能，其中 3 种为白色，所以取出白色石头的概率为 3/7 。那么取到黑色石头的概率又是多少呢？很显然，是 4/7 。我们使用 P(white) 来表示取到白色石头的概率，其概率值可以通过白色石头数目除以总的石头数目来得到。

![包含 7 块石头的集合](https://cdn.jsdelivr.net/gh/dongzhougu/imageuse1/NB_2.png)

如果这 7 块石头如下图所示，放在两个桶中，那么上述概率应该如何计算？

![7块石头放入两个桶中](https://cdn.jsdelivr.net/gh/dongzhougu/imageuse1/NB_5.png)

计算 P(white) 或者 P(black) ，如果事先我们知道石头所在桶的信息是会改变结果的。这就是所谓的条件概率（conditional probablity）。假定计算的是从 B 桶取到白色石头的概率，这个概率可以记作 P(white|bucketB) ，我们称之为“在已知石头出自 B 桶的条件下，取出白色石头的概率”。很容易得到，P(white|bucketA) 值为 2/4 ，P(white|bucketB) 的值为 1/3 。

条件概率的计算公式如下: 

P(white|bucketB) = P(white and bucketB) / P(bucketB)

首先，我们用 B 桶中白色石头的个数除以两个桶中总的石头数，得到 P(white and bucketB) = 1/7 .其次，由于 B 桶中有 3 块石头，而总石头数为 7 ，于是 P(bucketB) 就等于 3/7 。于是又 P(white|bucketB) = P(white and bucketB) / P(bucketB) = (1/7) / (3/7) = 1/3 。

另外一种有效计算条件概率的方法称为贝叶斯准则。贝叶斯准则告诉我们如何交换条件概率中的条件与结果，即如果已知 P(x|c)，要求 P(c|x)，那么可以使用下面的计算方法: 

![计算p(c|x)的方法](https://cdn.jsdelivr.net/gh/dongzhougu/imageuse1/NB_3.png)

### 2.3 使用条件概率来分类

上面我们提到贝叶斯决策理论要求计算两个概率 p1(x, y) 和 p2(x, y):

* 如果 p1(x, y) > p2(x, y), 那么属于类别 1;
* 如果 p2(x, y) > p1(X, y), 那么属于类别 2.

这并不是贝叶斯决策理论的所有内容。使用 p1() 和 p2() 只是为了尽可能简化描述，而真正需要计算和比较的是 p(c1|x, y) 和 p(c2|x, y) .这些符号所代表的具体意义是: 给定某个由 x、y 表示的数据点，那么该数据点来自类别 c1 的概率是多少？数据点来自类别 c2 的概率又是多少？注意这些概率与概率 p(x, y|c1) 并不一样，不过可以使用贝叶斯准则来交换概率中条件与结果。具体地，应用贝叶斯准则得到: 

![应用贝叶斯准则](https://cdn.jsdelivr.net/gh/dongzhougu/imageuse1/NB_4.png)

使用上面这些定义，可以定义贝叶斯分类准则为:

* 如果 P(c1|x, y) > P(c2|x, y), 那么属于类别 c1;
* 如果 P(c2|x, y) > P(c1|x, y), 那么属于类别 c2.

在文档分类中，整个文档（如一封电子邮件）是实例，而电子邮件中的某些元素则构成特征。我们可以观察文档中出现的词，并把每个词作为一个特征，而每个词的出现或者不出现作为该特征的值，这样得到的特征数目就会跟词汇表中的词的数目一样多。

我们假设特征之间  **相互独立** 。所谓 <b>独立(independence)</b> 指的是统计意义上的独立，即一个特征或者单词出现的可能性与它和其他单词相邻没有关系，比如说，“我们”中的“我”和“们”出现的概率与这两个字相邻没有任何关系。这个假设正是朴素贝叶斯分类器中 朴素(naive) 一词的含义。朴素贝叶斯分类器中的另一个假设是，<b>每个特征同等重要</b>。

> <b>Note:</b> 朴素贝叶斯分类器通常有两种实现方式: 一种基于伯努利模型实现，一种基于多项式模型实现。前者中并不考虑词在文档中出现的次数，只考虑出不出现，因此在这个意义上相当于假设词是等权重的。



## 3. 一个简单的例子

我们先通过一个简单的例子，来看怎样应用朴素贝叶斯分类算法。假设有以下关于驾龄、平均车速和性别的统计数据：

![img](https://cdn.jsdelivr.net/gh/dongzhougu/imageuse1/17634123-28f863c2b8a241e7.png)

> 现在观察到一个驾龄为2年的人，平均车速为80。问：这个人的性别是什么？

假设 $C_{0}$ 表示女  $C_{1}$ 表示男，$x_{0}$ 表示驾龄，$x_{1}$ 表示平均车速。我们先来计算这个人为女性的概率相对值。根据统计数据，女性司机的概率 $P\left(C_{0}\right)=5 / 10=0.5$ ，。驾龄为2年的女性司机的概率即 $P\left(x_{0} \mid C_{0}\right)=1 / 5=0.2$ 。平均车速为80的女性司机的概率  $P\left(x_{1} \mid C_{0}\right)=1 / 5=0.2$，根据朴素贝叶斯分类算法的数学公式：


$$
P\left(C_{0}\right) \prod_{i=1}^{n} P\left(x_{i} \mid C_{0}\right)=0.5 \times 0.2 \times 0.2=0.02
$$
接着计算这个人为男性的概率相对值。根据统计数据，不难得出男性司机的概率 $P\left(C_{1}\right)=5 / 10=0.5$ 。驾龄为2年的男性司机的概率 $P\left(x_{0} \mid C_{1}\right)=2 / 5=0.4$ 。平均车速为80的男性司机的概率 $P\left(x_{1} \mid C_{1}\right)=3 / 5=0.6$
$$
P\left(C_{1}\right) \prod_{i=1}^{n} P\left(x_{i} \mid C_{1}\right)=0.5 \times 0.4 \times 0.6=0.12
$$
从相对概率来看，这个人是男性的概率是女性的概率的6倍，据此判断这个人是男性。我们也可以从相对概率中算出绝对概率，即这个人是男性的绝对概率是0.12/(0.12+0.02)=0.857。

## 4.概率分布

到目前为止，我们介绍的朴素贝叶斯分类算法是根据数据集里的数据，计算出绝对概率来进行求解。再看一遍朴素贝叶斯分类算法的数学公式：
$$
P\left(C_{k} \mid x\right) \propto P\left(C_{k}\right) \prod_{i=1}^{n} P\left(x_{i} \mid C_{k}\right)
$$
其中， $P\left(C_{k} \mid x\right) $表示在类别$C_{k}$ 里特征$x_{i}$ 出现的概率。这里有个最大的问题，如果数据集太小，那么从数据集里计算出来的概率偏差将非常严重。例如，观察一个质地均匀的骰子投掷6次的结果是[1,3,1,5,3,3]。质地均匀的骰子每个点出现的概率都是1/6，如果根据观察到的数据集去计算每个点的概率，和真实的概率相差将是非常大的。

怎么解决这个问题呢？答案是使用概率分布来计算概率，而不是从数据集里计算概率。

### 4.1 概率统计的基本概念

人的身高是一个连续的随机变量，而投掷一个骰子得到的点数则是一个离散的随机变量。我们闭着眼睛随便找一个人，问这个人的身高是170cm的可能性是多大呢？如果有一个函数，能描述人类身高的可能性，那么直接把170cm代入即可求出这个可能性。这个函数就是概率密度函数，也称为`PDF（Probability Density Function）`。典型的概率密度函数是高斯分布函数，如人类的身高就满足高斯分布的规律。

再例如，投掷一个质地均匀的骰子，得到6的概率是多少呢？大家都知道答案是1/6。假如有一个函数f(x)，能描述骰子出现x点数的概率，那么把x代入即可得到概率，这个函数称为概率质量函数，即PMF（Probability Mass Function）。那么，为什么还有使用概率质量函数呢？一是在数学上追求统一性，二是并不是所有的离散随机变量的概率分布都像掷一次骰子这个直观。例如，投掷6次质地均匀的骰子，得到4个4的概率是多少？这个时候如果有概率质量函数，就可以轻松求解了。

> 总结一下，随机变量分成两种，一种是连续随机变量，另外一种是离散随机变量。概率密度函数描述的是连续随机变量在某个特定值的可能性，概率质量函数描述的是离散随机变量在某个特定值的可能性。而概率分布则是描述随机变量取值的概率规律。

### 4.2 多项式分布

 抛一枚硬币，要么出现正面，要么出现反面（假设硬币不会立起来）。假如出现正面的概率是p，则出现反面的概率就是1-p。符合这个规律的概率分布，称为 `伯努利分布（Bernoulli Distribution）`。其概率质量函数为：
$$
f(k ; p)=p^{k}(1-p)^{1-k}
$$
p是出现1的概率。例如，一枚质地均匀的硬币被抛一次，得到正面的概率为0.5。我们代入上述公式，也可以得到相同的结果，即f(1;0.5)=0.5。

更一般的情况，即不止两种可能性时，假设每种可能性是$p_{i}$, 则满足  $\sum_{i}^{n} p_{i}=1$， 条件的概率分布，称为`类别分布（Categorical Distribution）`。例如，投掷一枚骰子，则会出现6中可能性，所有的可能性加起来的概率为1。类别分布的概率质量函数为：
$$
f(x \mid p)=\prod_{i=1}^{k} p_{i}^{x_{i}}
$$
那么，一枚质地均匀的硬币被抛10次，出现3次正面的概率是多少呢？这是个典型的二项式分布问题。二项式分布指的是把符号伯努利分布的实验做了n次，结果1出现0次、1次、2次……n次的概率分别是多少，它的概率质量函数为：
$$
f(k ; n, p)=C_{n}^{k} p^{k}(1-p)^{n-k}
$$
枚质地均匀的硬币被抛10次，出现3次正面的概率是多少？代入二项式分布的概率质量函数，得到：
$$
f(3 ; 10,0.5)=\frac{10 !}{3 ! \times(10-3) !} \times 0.5^{3} \times(1-0.5)^{10-3}=0.1171875
$$
其中，0的阶乘为1，即0!=1。结果跟我们预期的相符。当实验只做一次时，二项式分布退化为伯努利分布。

简单总结一下，二项式分布描述的是多次伯努利实验中，某个结果出现次数的概率。多项式分布描述的是多次进行满足类别分布的实验中，所有类别出现的次数组合的分布。

二项式分布和多项式分布结合朴素贝叶斯算法，经常被用来实现文章分类算法。例如，有一个论坛需要对用户的评论进行过滤，屏蔽不文明的评论。首先要有一个经过标记的数据集，我们称为语料库。假设使用人工标记的方法对评论进行标记，1表示不文明的评论，0表示正常评论。

假设我们的词库大小为 k ，则评论中出现某个词可以看成是一次满足k个类别的类别分布实验。我们知道，一篇评论是由n个词组成的，因此一篇评论可以看出是进行n次类别分布实验后的产物。由此得知，一篇评论服从多项式分布，它是词库里的所有词语出现的次数组合构成的随机向量。

一般情况下，词库比较大，评论只是由少量词组成，所以这个随机向量是很稀疏的，即大部分元素为0。通过分析预料库，我们容易统计出每个词出现在不文明评论及正常评论的概率，即 $p_{i}$的值。同时针对待预测的评论，我们可以统计词库里的所有词在这篇评论里出现的次数即  $x_{i}$ 的值，及评论的词语个数。代入多项式分布的概率质量函数：
$$
f(X, n, P)=\frac{n !}{\prod_{i=1}^{k} x_{i} !} \prod_{i=1}^{k} p_{i}^{x_{i}}
$$
我们可以求出，待预测评论构成的随机向量x，其为不文明评论的相对概率。同理也可以求出其为正常评论的相对概率。通过比较两个相对概率，就可以对这篇评论输出一个预测值。当然，实际应用中，涉及大量的自然语言处理的手段，包括中文分词技术、词的数学表示等，这里不再展开。

### 4.3 高斯分布

在前面的车速和性别预测的例子里，对于平均车速，给出的是离散值，实际上它是一个连续值。这个时候怎么使用贝叶斯算法来处理呢？答案是，可以用区间把连续值转换成离散值。例如，我们可以把平均车速[0,40]作为一个级别，[40-80]，等等。这样就可以把连续值变成离散值，从而使用贝叶斯算法进行处理。另外一个方法，是使用连续随机变量的概率密度函数，把数值转换为一个相对概率。高斯分布就是这样一种方法。

`高斯分布（Gaussian Distribution）`也称为 `正态分布（Normal Distribution）`，是最常见的一种分布。例如人的身高满足高斯分布，特别高和特别矮的人出现的相对概率都很低，大部分人身高都处在中间水平。还有人的智商也符合高斯分布，特别聪明的天才和特别笨的人出现的相对概率都很低，大部分人的智力都差不多。高斯分布的概率密度函数为：
$$
f(x)=\frac{1}{\sqrt{2 \pi \sigma^{2}}} \exp \left(-\frac{(x-\mu)^{2}}{2 \sigma^{2}}\right)
$$
其中，$x$ 为随机变量的值，$f(x)$ 为随机变量的相对概率，$\mu$为样本的平均值，其决定了高斯分布曲线的位置，![\sigma](https://math.jianshu.com/math?formula=%5Csigma)为标准差，其决定了高斯分布的幅度，$\sigma$ 值越大，分布越分散，![\sigma](https://math.jianshu.com/math?formula=%5Csigma)$\sigma$值越小，分布越集中。典型的高斯分布如下图所示：

![img](https://cdn.jsdelivr.net/gh/dongzhougu/imageuse1/17634123-e4f8952fc25052e3.png)

这里需要注意的是：高斯分布的概率密度函数和支持向量机里的高斯核函数的区别。二者的核心数学模型是相同的，但是目的不同。

### 4.4 连续值得处理

假设，有一组身体特征的统计数据如下：

![img](https://cdn.jsdelivr.net/gh/dongzhougu/imageuse1/17634123-ab2a72e2cf69b3f1.png)

假设某人身高6英尺，体重130榜，脚掌8英寸，请问此人的性别是什么？

根据朴素贝叶斯公式：
$$
P\left(C_{k} \mid x\right) \propto P\left(C_{k}\right) \prod_{i=1}^{n} P\left(x_{i} \mid C_{k}\right)
$$
针对待预测的这个人的数据$x$ ，我们只需要分别求出男性和女性的相对概率
$$
P(\text {Gender}) \times P(\text {Height} \mid \text {Gender}) \times P(\text {Weight} \mid \text {Gender}) \times P(\text {Feet} \mid \text {Gender})
$$
然后取相对概率较高的性别为预测值即可。这里的困难在于，所有的特征都是连续变量，无法根据统计数据计算概率。当然，这里我们可以使用区间法，把连续变量变为离散变量，然后再计算概率。但由于数据量较小，这显然不是一个好办法。由于人类身高、体重、脚掌尺寸满足高斯分布，因此更好的办法是使用高斯分布的概率密度函数来求相对概率。

首先，针对男性和女性，分别求出特征的平均值和方差：

![img](https://cdn.jsdelivr.net/gh/dongzhougu/imageuse1/17634123-4755f7b16509dc1d.png)

接着利用高斯分布的概率密度函数，来求解男性身高为6英尺的相对概率：
$$
P(\text {Height}=6 \mid \text {Male})=\frac{1}{\sqrt{2 \pi \times 0.035033^{2}}} \exp \left(-\frac{(6-5.855)^{2}}{2 \times 0.035033^{2}}\right) \approx 1.5789
$$
这里的关键是把连续值（身高）作为输入，通过高斯分布的概率密度函数的处理，直接转换为相对概率。注意这里是相对概率，所以其值大于1并未违反概率论规则。



## 5. 示例：文档分类

在 `scikit-learn`里，朴素贝叶斯算法在 `sklearn.naive_bayes` 包里实现，包含本文介绍的几种典型的概率分布算法。其中 `GaussianNB` 实现了高斯分布的朴素贝叶斯算法，`MultinomialNB` 实现了多项式分布的朴素贝叶斯算法，`BernoulliNB`实现了伯努利分布的朴素贝叶斯算法。朴素贝叶斯算法在自然语言处理领域有着广泛的应用，这里我们使用 `MultinomialNB` 来实现文档的自动分类。

### 5.1 获取数据集

这里使用的数据集来自 mlcomp.org上的20news-18828，可以直接访问[http://mlcomp.org/datasets/379](https://links.jianshu.com/go?to=http%3A%2F%2Fmlcomp.org%2Fdatasets%2F379)下载。其目录下包含3个子目录和一个名为metadata的介绍文件，数据集也可在百度网盘下载。，已分享。

> 链接：https://pan.baidu.com/s/1uQNkLWIN0niz8-p8BppRJg 
> 提取码：bvhe 
> 复制这段内容后打开百度网盘手机App，操作更方便哦

我们将使用 `train` 子目录下的文档进行模型训练，然后使用 `test` 子目录下的文档进行模型测试。`train` 子目录下包含 20 个子目录，每个子目录代表一种文档的类型，子目录下的所有文档都是属于目录名称所标识的文档类型。可以随意浏览数据集，以便对数据集有一个感性的认识。例如，datasets/mlcomp/379/train/rec.autos/6652-103421是一个讨论汽车主题的帖子：

```kotlin
Hahahahahaha. gasp pant Hm, I’m not sure whether the above was just a silly 
    remark or a serious remark. But in case there are some misconceptions,
I think Henry Robertson hasn’t updated his data file on Korea since…mid 1970s. 
Owning a car in Korea is no longer a luxury. Most middle class people 
in Korea can afford a car and do have at least one car. The problem in Korea,
especially in Seoul, is that there are just so many privately-owned cars,
as well as taxis and buses, the rush-hour has become a 24 hour phenomenon 
    and that there is no place to park. Last time I heard, back in January, 
the Kim Administration wanted to legislate a law requireing a potential 
car owner to provide his or her own parking area, just like they do in Japan.
Also, Henry would be glad to know that Hyundai isn’t the only car manufacturer 
in Korea. Daewoo has always manufactured cars and I believe Kia is back in 
business as well. Imported cars, such as Mercury Sable are becoming quite 
popular as well, though they are still quite expensive.
Finally, please ignore Henry’s posting about Korean politics and bureaucracy. 
    He’s quite uninformed.
```

### 5.2 文档的数学表达

怎样把一个文档表达为计算机可以理解并处理的信息，是自然语言处理中的一个重要课题，完整内容可以写成鸿篇巨著。本节简单介绍TF-IDF的原理，以便更好地理解本文介绍的实例。

`TF-IDF` 是一种统计方法，用以评估一个词语对于一份文档的重要程度。`TF（Term Frequency）`表示词频，对于一份文档而言，词频是指特定词语在这篇文档里出现的次数除以该文档总词数。例如，一篇文档一共有1000个词，其中“朴素贝叶斯”出现了5次，“的”出现了25次，“应用”出现了12次，那么它们的词频分别是0.005，0.025和0.012。

`IDF（Inverse Document Frequency）`表示一个词的逆向文档频率，由总文档数除以包含该词的文档数的商再取对数得到。例如：我们的数据集一共10000篇文档，其中“朴素贝叶斯”只出现在10篇文档中，则其`IDF=log(10000/10)=3`；“的”在所有文档中都出现过，则其 `IDF=log(10000/10000)=0`；“应用”在1000篇文档中出现过，则其 `IDF=log(10000/1000)=1`。

计算出每个词的TF和IDF之后，两者相乘，即可得到这个词在文档中的重要程度。词语的重要性与它在该文档中出现的次数成正比，与它在语料库中出现的文档数成反比。

有了TF-IDF这个工具，我们就可以把一篇文档转换为一个向量。首先，可以从数据集`（在自然语言处理领域也称corpus，即语料库）`里提取出所有出现的词，我们称为词典。假设词典里总共有10000个词语，则每个文档都可以转化为一个10000维的向量。其次，针对我们要转换的文档里出现的每个词语，都去计算其TF-IDF，并把这个值填入文档向量里这个词对应的元素上。这样就完成了把一篇文档转换为一个向量的过程。一个文档往往只会由词典里的一小部分词语构成，这就意味着这个向量里的大部分元素都是0。

所幸，上述过程不需要我们自己写代码去完成，`scikit-learn` 软件包里实现了把文档转换为向量的过程。首先，把训练用的语料库读入内存：

```python
from time import time
from sklearn.datasets import load_files
print("loading train dataset ...")
t = time()
news_train = load_files('datasets/mlcomp/379/train')
print("summary: {0} documents in {1} categories."
      .format(len(news_train.data),len(news_train.target_names)))
print("done in {0} seconds".format(time()-t))
```

输出如下：

```bash
loading train dataset ...
summary: 13180 documents in 20 categories.
done in 1.2616519927978516 seconds
```

其中，`datasets/mlcomp/379/train`目录下放的就是我们的语料库，其中包含20个子目录，每个子目录的名字表示的是文档的类别，子目录下包含这种类别的所有文档。`load_files()` 函数会从这个目录里把所有的文档都读入内存，并且自动根据所在的子目录名称打上标签。其中，`news_train.data`是一个数组，里面包含了所有文档的文本信息。`news_train.target`也是一个数组，包含了所有文档所属的类别，而`news_train.target_names`则是类别的名称，因此，如果我们想知道第一篇文档所属的类别名称，只需要通过代码`news_train.target_names[news_train.target[0]]`即可得到。

该语料库里总共有13180个文档，分成20个类别。接着需要把这些文档全部转换为由TF-IDF表达的权重信息构成的向量：

```python
from sklearn.feature_extraction.text import TfidfVectorizer
print("vectorizing train dataset ...")
t = time()
vectorizer = TfidfVectorizer(encoding='latin-1')
X_train = vectorizer.fit_transform((d for d in news_train.data))
print("n_samples: %d, n_features: %d" % X_train.shape)
print("number of non-zero features in samples [{0}]:{1}"
      .format(news_train.filenames[0],X_train[0].getnnz()))
print("done in {0} seconds".format(time()-t))
```

输出如下：

```css
vectorizing train dataset ...
n_samples: 13180, n_features: 130274
number of non-zero features in samples [datasets/mlcomp/379/train\talk.politics.misc\17860-178992]:108
done in 2.6174726486206055 seconds
```

其中，`TfidfVectorizer`类是用来把所有的文档转换为矩阵，该矩阵每行都代表一个文档，一行中的每个元素代表一个对应的词语的重要性，词语的重要性由`TF-IDF`来表示。其 `fit_transform()` 方法是 fit() 和transform()合并起来。其中，fit()  会先完成语料库分析、提取词典等操作，transform()会把对每篇文档转换为向量，最终构成一个矩阵，保存在X_train变量里。

由输出可以知道，我们的词典总共有 130274 个词语，即每篇文档都可转换为一个 130274 维的向量。第一篇文档中，只有108个非零元素，即这篇文档总共由108个不重复的单词组成，在这篇文档中出现的这108个单词的TF-IDF值会被计算出来，并保存在向量中的指定位置上。X_train是一个维度为13180*130274的稀疏矩阵。

X_train稀疏矩阵由一个三元组(row,col,score)构成：

```bash
print(X_train[0])
```

输出如下：

```css
  (0, 56813)    0.014332663773643272
  (0, 45689)    0.08373343949755
  (0, 46084)    0.08109733529789522
  (0, 125882)   0.0873157704840211
  (0, 50150)    0.020654313721609956
  (0, 87702)    0.04643235585055511
  (0, 33334)    0.1025405658189532
  (0, 111805)   0.014332663773643272
  : :
  (0, 67768)    0.08982314745972582
  (0, 41790)    0.09260592033433869
  (0, 105800)   0.08713990737243116
  (0, 37075)    0.10018566542781165
  (0, 23162)    0.08920437523600384
  (0, 124699)   0.06257976758779137
  (0, 94119)    0.1159317059788844
  (0, 56555)    0.06984885482106491
  (0, 62776)    0.10474995568339582
```

### 5.3 模型训练

使用 `MultinomialNB` 对数据集进行训练：

```python
from sklearn.naive_bayes import MultinomialNB
print("training models ...")
t = time()
y_train = news_train.target
clf = MultinomialNB(alpha=0.0001)
clf.fit(X_train,y_train)
train_score = clf.score(X_train,y_train)
print("train score: {0}".format(train_score))
print("done in {0} seconds".format(time()-t))
```

输出如下：

```css
training models ...
train score: 0.9978755690440061
done in 0.15497064590454102 seconds
```

其中，alpha表示平滑参数，其值越小，越容易造成过拟合，值太大，容易造成欠拟合。

接着，我们加载测试数据集，并用一篇文档来预测其是否准确。测试数据集在`datasets/mlcomp/379/test`目录下，我们用前面介绍的相同的方法先加载数据集：

```go
print("loading test dataset ...")
t = time()
news_test = load_files('datasets/mlcomp/379/test')
print("summary: {0} documents in {1} categories."
      .format(len(news_test.data),len(news_test.target_names)))
print("done in {0} seconds".format(time()-t))
```

输出如下：

```bash
loading test dataset ...
summary: 5648 documents in 20 categories.
done in 0.3548290729522705 seconds
```

测试数据集共有5648篇文档。接着，我们把文档向量化：

```bash
print("vectorizing test dataset ...")
t = time()
X_test = vectorizer.transform((d for d in news_test.data))
y_test = news_test.target
print("n_samples: %d, n_features: %d" % X_test.shape)
print("number of non-zero features in sample [{0}]: {1}"
      .format(news_test.filenames[0],X_test[0].getnnz()))
print("done in {0} seconds".format(time()-t))
```

输出如下：

```bash
vectorizing test dataset ...
n_samples: 5648, n_features: 130274
number of non-zero features in sample [datasets/mlcomp/379/test\rec.autos\7429-103268]: 61
done in 0.9695498943328857 seconds
```

注意，vectorizer变量是我们处理训练数据集时用到的向量化的类的实例，此处我们只需要调用transform()进行TF-IDF数值计算即可，不需要再调用fit()进行语料库分析了。

这样，我们的测试数据集也转换为了一个维度为5648*130274的稀疏矩阵。可以取测试数据集里的第一篇文档初步验证一下，看看训练出来的模型能否正确地预测这个文档所属的类别：

```bash
pred = clf.predict(X_test[0])
print("predict: {0} is in category {1}"
      .format(news_test.filenames[0],news_test.target_names[pred[0]]))
print("actually: {0} is in category {1}"
      .format(news_test.filenames[0],news_test.target_names[news_test.target[0]]))
```

输出如下：

```bash
predict: datasets/mlcomp/379/test\rec.autos\7429-103268 is in category rec.autos
actually: datasets/mlcomp/379/test\rec.autos\7429-103268 is in category rec.autos
```

看来预测的结果和实际结果是相符的。

### 5.4 模型评价

虽然通过验证，说明我们训练的模型是可用的，但是不能通过一个样本的预测来评价模型的准确性。我们需要对模型有个全方位的评价，所幸 `scikit-learn` 软件包提供了全方位的模型评价工具。

首先需要对测试数据集进行预测：

```bash
print("predicting test dataset ...")
t = time()
pred_test = clf.predict(X_test)
print("done in %fs" % (time()-t))
```

接着使用 `classification_report()` 函数来查看一下针对每个类别的预测准确性：

```python
from sklearn.metrics import classification_report
print("classification report on test set for classifier:")
print(clf)
print(classification_report(y_test,pred_test,target_names=news_test.target_names))
```

输出如下：

```bash
classification report on test set for classifier:
MultinomialNB(alpha=0.0001, class_prior=None, fit_prior=True)
                          precision    recall  f1-score   support

             alt.atheism       0.90      0.91      0.91       245
           comp.graphics       0.80      0.90      0.85       298
 comp.os.ms-windows.misc       0.82      0.79      0.80       292
comp.sys.ibm.pc.hardware       0.81      0.80      0.81       301
   comp.sys.mac.hardware       0.90      0.91      0.91       256
          comp.windows.x       0.88      0.88      0.88       297
            misc.forsale       0.87      0.81      0.84       290
               rec.autos       0.92      0.93      0.92       324
         rec.motorcycles       0.96      0.96      0.96       294
      rec.sport.baseball       0.97      0.94      0.96       315
        rec.sport.hockey       0.96      0.99      0.98       302
               sci.crypt       0.95      0.96      0.95       297
         sci.electronics       0.91      0.85      0.88       313
                 sci.med       0.96      0.96      0.96       277
               sci.space       0.94      0.97      0.96       305
  soc.religion.christian       0.93      0.96      0.94       293
      talk.politics.guns       0.91      0.96      0.93       246
   talk.politics.mideast       0.96      0.98      0.97       296
      talk.politics.misc       0.90      0.90      0.90       236
      talk.religion.misc       0.89      0.78      0.83       171

             avg / total       0.91      0.91      0.91      5648
```

从输出结果中可以看出，针对每种类别都统计了`查准率`、`召回率`和`F1-Score`。此外，还可以通过confusion_matrix()函数生成混淆矩阵，观察每种类别被错误分类的情况。例如，这些被错误分类的文档是被错误分类到哪些类别里的：

```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,pred_test)
print("confusion matrix:\n")
print(cm)
```

输出如下：

```csharp
confusion matrix:

[[224   0   0   0   0   0   0   0   0   0   0   0   0   0   2   5   0   0    1  13]
 [  1 267   5   5   2   8   1   1   0   0   0   2   3   2   1   0   0   0    0   0]
 [  1  13 230  24   4  10   5   0   0   0   0   1   2   1   0   0   0   0    1   0]
 [  0   9  21 242   7   2  10   1   0   0   1   1   7   0   0   0   0   0    0   0]
 [  0   1   5   5 233   2   2   2   1   0   0   3   1   0   1   0   0   0    0   0]
 [  0  20   6   3   1 260   0   0   0   2   0   1   0   0   2   0   2   0    0   0]
 [  0   2   5  12   3   1 235  10   2   3   1   0   7   0   2   0   2   1    4   0]
 [  0   1   0   0   1   0   8 300   4   1   0   0   1   2   3   0   2   0    1   0]
 [  0   1   0   0   0   2   2   3 283   0   0   0   1   0   0   0   0   0    1   1]
 [  0   1   1   0   1   2   1   2   0 297   8   1   0   1   0   0   0   0    0   0]
 [  0   0   0   0   0   0   0   0   2   2 298   0   0   0   0   0   0   0    0   0]
 [  0   1   2   0   0   1   1   0   0   0   0 284   2   1   0   0   2   1    2   0]
 [  0  11   3   5   4   2   4   5   1   1   0   4 266   1   4   0   1   0    1   0]
 [  1   1   0   1   0   2   1   0   0   0   0   0   1 266   2   1   0   0    1   0]
 [  0   3   0   0   1   1   0   0   0   0   0   1   0   1 296   0   1   0    1   0]
 [  3   1   0   1   0   0   0   0   0   0   1   0   0   2   1 280   0   1    1   2]
 [  1   0   2   0   0   0   0   0   1   0   0   0   0   0   0   0 236   1    4   1]
 [  1   0   0   0   0   1   0   0   0   0   0   0   0   0   0   3   0 290    1   0]
 [  2   1   0   0   1   1   0   1   0   0   0   0   0   0   0   1  10   7  212   0]
 [ 16   0   0   0   0   0   0   0   0   0   0   0   0   0   0  12   4   1    4 134]]
```

例如：从第一行可以看出，类别0（alt.atheism）的文档，有13个被错误地分类到类别19（talk.religion.misc）里。当然，我们还可以把混淆矩阵进行数据可视化：

```python
# Show confusion matrix
import matplotlib.pyplot as plt
plt.figure(figsize=(8,8),dpi=144)
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
plt.matshow(cm,fignum=1,cmap='gray')
plt.colorbar()
```

输出图形如下：

![image-20200702160731221](https://cdn.jsdelivr.net/gh/dongzhougu/imageuse1/image-20200702160731221.png)除对角线外，其他地方颜色越浅，说明此处错误越多。通过这些数据，我们可以详细分析样本数据，找出为什么某种类别会被错误地分类到另一种类别里，从而进一步优化模型。











