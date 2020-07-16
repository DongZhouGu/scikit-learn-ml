# PCA算法

## 1. PCA概述

PCA算法全称是 Principal Component Analysis，即主成分分析算法。它是一种维数约减（Dimensionality Reduction）算法，即把高维度数据在损失最小的情况下转换为低维度数据的算法。显然，PCA可以用来对数据进行压缩，可以在可控的失真范围内提高运算速度。。

## 2. PCA算法原理

我们先从最简单的情况谈起，假设需要把一个二维数据降维成一维数据，要怎么做呢？如下图所示。

![img](https://cdn.jsdelivr.net/gh/dongzhougu/imageuse1/17634123-805dbb25ada21691.png)

我们可以想办法找出一个向量 $u^{(1)}$ ，以便让二维数据的点（方形点）到这个向量所在的直线上的平均距离最短，即投射误差最小。

这样就可以在失真最小的情况下，把二维数据转换为向量 $u^{(1)}$ ，所在直线上的一维数据。再进一步，假设需要把三维数据降为二维数据时，我们需要找出两个向量  $u^{(1)}$  和 $u^{(2)}$ ，以便让三维数据的点在这两个向量决定的平面上的投射误差最小。

如果从数学角度来一般地描述PCA算法就是：当需要从n维数据降为k维数据时，需要找出k个向量

  $u^{(1)}$  ，  $u^{(2)}$  ，……  $u^{(k)}$  ，把n维的数据投射到这k个向量决定的线性空间里，最终使投射误差最小化的过程。

问题来了，**怎样找出投射误差最小的k个向量呢**？要完整的用数学公式推导这个方法，涉及较多高级线性代数的知识，这里不再详述。感兴趣的话可以参考后面扩展部分的内容。下面我们直接介绍PCA算法求解的一般步骤。

假设有一个数据集，用m x n维的矩阵A表示。矩阵中每一行表示一个样本，每一列表示一个特征，总共有m个样本，每个样本有n个特征。我们的目标是减少特征个数，保留最重要的k个特征。

### 2.1数据归一化和缩放

数据归一化和缩放是一种数学技巧，旨在提高PCA运算时的效率。数据归一化的目标是使特征的均值为0。数据归一化公式为:
$$
x_{j}^{(i)}=a_{j}^{(i)}-\mu_{j}
$$
其中，$a_{j}^{(i)}$是指第i个样本的第j个特征的值，$\mu_{j}$表示的是第j个特征的均值。当不同的特征值不在同一个数量级上的时候，还需要对数据进行缩放。数据归一化在缩放的公式为：
$$
x_{j}^{(i)}=\frac{a_{j}^{(i)}-\mu_{j}}{s_{j}}
$$
其中，$a_{j}^{(i)}$是指第i个样本的第j个特征的值，$\mu_{j}$表示的是第j个特征的均值。$s_{j}$表示第j个特征的范围，即 $s_{j} = max(a_{j}^{(i)})-min(a_{j}^{(i)})$

### 2.2 计算协方差矩阵的特征向量

针对预处理后的矩阵X，先计算其协方差矩阵（Covariance Matrix）：
$$
\Sigma=\frac{1}{m} X^{T} X
$$
其中，$\Sigma $ 表示协方差矩阵，用大写的Sigma表示，是一个n * n维的矩阵。

接着通过奇异值分解来计算协方差矩阵的特征向量：
$$
[U, S, V]=s v d(\Sigma)
$$
其中，svd 是奇异值分解（Singular Value Decomposition）运算，是高级线性代数的内容。经过奇异值分解后，有3个返回值，其中矩阵U是一个n * n的矩阵，如果我们选择U的列作为向量，那么我们将得到n个列向量 $u^{(1)}$  ，  $u^{(2)}$  ，……  $u^{(n)}$  ,这些向量就是协方差矩阵的特征向量。它表示的物理意义是，协方差矩阵  $\Sigma $ 可以由这些特征向量进行线性组合得到。

### 2.3  数据降维和恢复

得到特征矩阵后，就可以对数据进行降维处理了。假设降维前的值是  $x^{(i)}$，降维后是$z^{(i)}$，那么
$$
z^{(i)}=U_{r e d u c e}^{T} x^{(i)}
$$
其中，$U_{r e d u c e}=[u^{(1)} ,u^{(2)}，……u^{(k)}]$ ，它选取自矩阵U的前k个向量，$U_{r e d u c e}$

称为主成分特征矩阵，它是数据降维和恢复的关键中间变量。看一下数据维度，$U_{r e d u c e}$是n * k的矩阵，因此 $U_{r e d u c e}^{T}$是k * n的矩阵.

也可以用矩阵运算一次性转换多个向量，提高效率。假设X是行向量 $x^{(i)}$组成的矩阵，则
$$
Z=X U_{\text {reduce}}
$$
其中，X是m * n的矩阵，降维后的矩阵Z是一个m * k的矩阵。



数据降维后，怎么恢复呢？从前面的计算公式我们知道，降维后的数据计算公式
$ z^{(i)}=U_{r e d u c e}^{T} x^{(i)} $ 。所以如果要还原数据，可以使用下面的公式：
$$
x_{a p p r o x}^{(i)}=U_{r e d u c e} z^{(i)}
$$
其中，$U_{r e d u c e}$是n * k的矩阵，$z^{(i)}$是k维列向量。这样算出来的$x^{(i)} $就是n维列向量。

矩阵化数据恢复运算公式为：
$$
X_{approx}=Z U_{r e d u c e}^{T}
$$
其中, $X_{approx}$ 是还原回来的数据，是一个m * n的矩阵，每行表示一个训练样例。Z是一个m * k的矩阵，是降维后的数据。

## 3. PCA算法示例

假设我们的数据集总共有5个记录，每个记录有2个特征，这样构成的矩阵A为：
$$
A=\left[\begin{array}{ll}
3 & 2000 \\
2 & 3000 \\
4 & 5000 \\
5 & 8000 \\
1 & 2000
\end{array}\right]
$$
我们的目标是把二维数据降为一维数据。为了更好地理解PCA的计算过程，分别使用 Numpy和sklearn 对同一个数据进行PCA降维处理。

### 3.1 使用Numpy模拟PCA计算过程

```python
import numpy as np
A = np.array([[3,2000],
             [2,3000],
             [4,5000],
             [5,8000],
             [1,2000]],dtype='float')
# 数据归一化，axis=0表示按列归一化
mean = np.mean(A,axis=0)
norm = A - mean
# 数据缩放
score = np.max(norm,axis=0)-np.min(norm,axis=0)
norm = norm / score
print(norm)
```

由于两个特征的均值不在同一个数量级，所以对数据进行了缩放。输出如下：

```python
array([[ 0.        , -0.33333333],
       [-0.25      , -0.16666667],
       [ 0.25      ,  0.16666667],
       [ 0.5       ,  0.66666667],
       [-0.5       , -0.33333333]])
```

接着，对协方差矩阵进行奇异值分解，求解其特征向量：

```python
U,S,V = np.linalg.svd(np.dot(norm.T,norm))
print(U)
```

输出如下：

```python
array([[-0.67710949, -0.73588229],
       [-0.73588229,  0.67710949]])
```

由于需要把二维数据降为一维数据，因此只取特征矩阵的第一列（前k列）来构造主成分特征矩阵$U_{reduce}$

```python
U_reduce = U[:,0].reshape(2,1)
U_reduce
```

输出如下：

```python
array([[-0.67710949],
       [-0.73588229]])
```

有了主成分特征矩阵，就可以对数据进行降维了：

```python
R = np.dot(norm,U_reduce)
print(R)
```

输出如下：

```python
array([[ 0.2452941 ],
       [ 0.29192442],
       [-0.29192442],
       [-0.82914294],
       [ 0.58384884]])
```

这样就把二维的数据降为一维的数据了。如果需要还原数据，依照PCA数据恢复的计算公式，可得：

```python
Z = np.dot(R,U_reduce.T)
print(Z)
```

输出如下：

```python
array([[-0.16609096, -0.18050758],
       [-0.19766479, -0.21482201],
       [ 0.19766479,  0.21482201],
       [ 0.56142055,  0.6101516 ],
       [-0.39532959, -0.42964402]])
```

由于我们在数据预处理阶段对数据进行了归一化，并且做了缩放处理，所以需要进一步还原才能得到原始数据，这一步是数据预处理的逆运算。

```python
A1 = np.multiply(Z,scope)+mean
print(A1)
```

其中，np.multiply是矩阵对应元素相乘，np.dot是矩阵的行乘以矩阵的列。输出如下：

```python
array([[2.33563616e+00, 2.91695452e+03],
       [2.20934082e+00, 2.71106794e+03],
       [3.79065918e+00, 5.28893206e+03],
       [5.24568220e+00, 7.66090960e+03],
       [1.41868164e+00, 1.42213588e+03]])
```

与原始矩阵A相比，恢复后的数据A1还是存在一定程度的失真，这种失真是不可避免的。

### 3.2 使用sklearn进行PCA降维运算

在 `sklearn`工具包里，类 `sklearn.decomposition.PCA` 实现了 PCA 算法，使用方便，不需要了解具体的PCA的运算步骤。但需要注意的是，数据的预处理需要自己完成，其 PCA 算法本身不进行数据预处理（归一化和缩放）。此处，我们选择 `MinMaxScaler类`进行数据预处理。

```python
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

def std_PCA(**argv):
    scaler = MinMaxScaler()
    pca = PCA(**argv)
    pipeline = Pipeline([('scaler', scaler),
                         ('pca', pca)])
    return pipeline

pca = std_PCA(n_components=1)
R2 = pca.fit_transform(A)
print(R2)
```

输出如下：

```python
array([[-0.2452941 ],
       [-0.29192442],
       [ 0.29192442],
       [ 0.82914294],
       [-0.58384884]])
```

这个输出值就是矩阵A经过预处理以及PCA降维后的数值。我们发现，这里的输出结果和上面使用Numpy方式的输出结果符号相反，这其实不是错误，只是降维后选择的坐标方向不同而已。

接着把数据恢复回来：

```python
A2 = pca.inverse_transform(R2)
print(A2)
```

这里的pca是一个Pipeline实例，其逆运算inverse_transform()是逐级进行的，即先进行PCA还原，再执行预处理的逆运算。即先调用PCA.inverse_transform()，然后再调用MinMaxScaler.inverse_transform()。输出如下：

```python
array([[2.33563616e+00, 2.91695452e+03],
       [2.20934082e+00, 2.71106794e+03],
       [3.79065918e+00, 5.28893206e+03],
       [5.24568220e+00, 7.66090960e+03],
       [1.41868164e+00, 1.42213588e+03]])
```

可以看到，这里还原回来的数据和前面Numpy方式还原回来的数据是一致的。

### 3.3 PCA的物理含义

我们可以把前面例子中的数据在一个坐标轴上全部画出来，从而仔细观察PCA降维过程的物理含义。如下图所示。

```python
def draw(norm, Z, U, U_reduce):
    plt.figure(figsize=(8, 8), dpi=144)
    plt.title('Physcial meanings of PCA')
    ymin = xmin = -1
    ymax = xmax = 1
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    ax = plt.gca()  # gca 代表当前坐标轴，即 'get current axis'
    ax.spines['right'].set_color('none')  # 隐藏坐标轴
    ax.spines['top'].set_color('none')

    plt.scatter(norm[:, 0], norm[:, 1], marker='s', c='b')
    plt.scatter(Z[:, 0], Z[:, 1], marker='o', c='r')
    plt.arrow(0, 0, U[0][0], U[1][0], color='r', linestyle='-')
    plt.arrow(0, 0, U[0][1], U[1][1], color='r', linestyle='--')
    plt.annotate(r'$U_{reduce} = u^{(1)}$',
                 xy=(U[0][0], U[1][0]), xycoords='data',
                 xytext=(U_reduce[0][0] + 0.2, U_reduce[1][0] - 0.1), fontsize=10,
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    plt.annotate(r'$u^{(2)}$',
                 xy=(U[0][1], U[1][1]), xycoords='data',
                 xytext=(U[0][1] + 0.2, U[1][1] - 0.1), fontsize=10,
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    plt.annotate(r'raw data',
                 xy=(norm[0][0], norm[0][1]), xycoords='data',
                 xytext=(norm[0][0] + 0.2, norm[0][1] - 0.2), fontsize=10,
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    plt.annotate(r'projected data',
                 xy=(Z[0][0], Z[0][1]), xycoords='data',
                 xytext=(Z[0][0] + 0.2, Z[0][1] - 0.1), fontsize=10,
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    plt.show()
```

![image-20200716110204443](https://cdn.jsdelivr.net/gh/dongzhougu/imageuse1/image-20200716110204443.png)图中正方形的点是原始数据经过预处理后（归一化、缩放）的数据，圆形的点是从一维恢复到二维后的数据。同时，我们画出主成分特征向量 $u^{(1)}$ 和  $u^{(2)}$ ，。根据上图，来介绍几个有意思的结论：首先，圆形的点实际上就是方形的点在向量所在 $u^{(1)}$ 直线上的投影。所谓PCA数据恢复，并不是真正的恢复，只是把降维后的坐标转换为原坐标系中的坐标而已。针对我们的例子，只是把由向量 $u^{(1)}$决定的一维坐标系中的坐标转换为原始二维坐标系中的坐标。其次，主成分特征向量 $u^{(1)}$ 和  $u^{(2)}$ 是相互垂直的。再次，方形点和圆形点之间的距离，就是PCA数据降维后的误差。

## 4. 示例：人脸识别

本节使用英国剑桥AT&T实验室的研究人员自拍的一组照片（AT&TLaboratories Cambridge），来开发一个特定的人脸识别系统。人脸识别，本质上是个分类问题，需要把人脸图片当成训练数据集，对模型进行训练。训练好的模型，就可以对新的人脸照片进行类别预测。这就是人脸识别系统的原理。

### 4.1 加载数据集

查看数据集里所有400张照片的缩略图。数据集总共包含40位人员的照片，每个人10张照片，数据集在仓库`dataset`文件夹内。

下载完照片，就可以使用下面的代码来加载这些照片了：

```python
import time
import logging
from sklearn.datasets import fetch_olivetti_faces
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
data_home='datasets/'
logging.info('Start to load dataset')
faces = fetch_olivetti_faces(data_home=data_home)
logging.info('Done with load dataset')
```

输出如下：

```python
2019-06-23 21:45:13,639 Start to load dataset
2019-06-23 21:45:13,666 Done with load dataset
```

加载的图片数据集保存在faces变量里，`scikit-learn` 已经替我们把每张照片做了初步的处理，剪裁成64×64大小且人脸居中显示。这一步至关重要，否则我们的模型将被大量的噪声数据，即图片背景干扰。因为人脸识别的关键是五官纹理和特征，每张照片的背景都不同，人的发型也可能经常变化，这些特征都应该尽量排除在输入特征之外。

成功加载数据后，其data里保存的就是按照scikit-learn要求的训练数据集，target里保存的就是类别目标索引。我们通过下面的代码，将数据集的概要信息显示出来：

```python
import numpy as np
X = faces.data
y = faces.target
targets = np.unique(faces.target)
target_names = np.array(["c%d" % t for t in targets])
n_targets = target_names.shape[0]
n_samples, h, w = faces.images.shape
print('Sample count: {}\nTarget count: {}'.format(n_samples, n_targets))
print('Image size: {}x{}\nDataset shape: {}\n'.format(w, h, X.shape))
```

输出如下：

```python
Sample count: 400
Target count: 40
Image size: 64x64
Dataset shape: (400, 4096)
```

从输出可知，总共有40位人物的照片，图片总数是400张，输入特征有4096个。为了后续区分不同的人物，我们用索引号给目标人物命名，并保存在变量target_names里。为了更直观地观察数据，从每个人物的照片里随机选择一张显示出来。先定义一个函数来显示照片阵列：

```python
import matplotlib.pyplot as plt
def plot_gallery(images,titles,h,w,n_row=2,n_col=5):
    """显示图片阵列"""
    plt.figure(figsize=(2*n_col,2*n_row),dpi=144)
    plt.subplots_adjust(bottom=0,left=0.01,right=0.99,top=0.90,hspace=0.01)
    for i in range(n_row*n_col):
        plt.subplot(n_row,n_col,i+1)
        plt.imshow(images[i].reshape((h,w)), cmap=plt.cm.gray)
        plt.title(titles[i])
        plt.axis('off')
```

输入参数images是一个二维数据，每一行都是一个图片数据。在加载数据时，fetch_olivetti_faces()函数已经帮我们做了预处理，图片的每个像素的RGB值都转换成了[0,1]的浮点数。因此，我们画出来的照片将是黑白的，而不是彩色的。在图片识别领域，一般情况下用黑白照片就可以了，可以减少计算量，也会让模型更准确。

接着分成两行显示出这些人物的照片：

```python
n_row = 2
n_col = 6
sample_images = None
sample_titles = []
for i in range(n_targets):
    people_images = X[y==i]
    people_sample_index = np.random.randint(0, people_images.shape[0], 1)
    people_sample_image = people_images[people_sample_index, :]
    if sample_images is not None:
        sample_images = np.concatenate((sample_images, people_sample_image), axis=0)
    else:
        sample_images = people_sample_image
    sample_titles.append(target_names[i])

plot_gallery(sample_images, sample_titles, h, w, n_row, n_col)
```

代码中，X[y==i]可以选择出属于特定人物的所有照片，随机选择出来的照片都放在sample_images数组对象里，最后使用我们之前定义的函数plot_gallery()把照片画出来，如下图所示。

![image-20200716162800005](https://cdn.jsdelivr.net/gh/dongzhougu/imageuse1/image-20200716162800005.png)

从图片中可以看到，fetch_olivetti_faces()函数帮我们剪裁了中间部分，只留下脸部特征。

最后，把数据集划分成训练数据集和测试数据集：

```jsx
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=4)
```

### 4.2 一次失败的尝试

我们使用支持向量机来实现人脸识别：

```python
from sklearn.svm import SVC
from time import time
start = time()
print('Fitting train datasets ...')
clf = SVC(class_weight='balanced')
clf.fit(X_train,y_train)
print('Done in {0:.2f}s'.format(time()-start))
```

输出如下：

```python
Fitting train datasets ...
Done in 0.92s
```

指定SVC的class_weight参数，让SVC模型能根据训练样本的数量来均衡地调整权重，这对不均匀的数据集，即目标人物的照片数量相差较大的情况是非常有帮助的。由于总共只有400张照片，数据规模较小，模型很快就运行完了。

接着，针对测试数据集进行预测：

```python
start = time()
print('Predicting test dataset ...')
y_pred = clf.predict(X_test)
print('Done in {0:.2f}s'.format(time()-start))
```

输出如下：

```bash
Predicting test dataset ...
Done in 0.10s
```

最后，分别使用 `confusion_matrix` 和 `classification_report` 来查看模型分类的准确性。

```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred,labels=range(n_targets))
print('confusion matrix:\n')
np.set_printoptions(threshold=sys.maxsize)
print(cm)
```

`np.set_printoptions()` 是为了确保完整地输出cm数组的内容，这是因为这个数组是40×40的，默认情况下不会全部输出。输出如下：

```csharp
confusion matrix:

[[0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]
```

`confusion matrix` 理想的输出，是矩阵的对角线上有数字，其他地方都没有数字。但我们的结果显示不是这样的。可以明显看出，很多图片都被预测成索引为12的类别了。结果看起来完全不对，这是怎么回事呢？我们再看一下classification_report的结果：

```python
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
```

输出如下：

```undefined
             precision    recall  f1-score   support

         c0       0.00      0.00      0.00         1
         c1       0.00      0.00      0.00         3
         c2       0.00      0.00      0.00         2
         c3       0.00      0.00      0.00         1
         c4       0.00      0.00      0.00         1
         c5       0.00      0.00      0.00         1
         c6       0.00      0.00      0.00         4
         c7       0.00      0.00      0.00         2
         c8       0.00      0.00      0.00         4
         c9       0.00      0.00      0.00         2
        c10       0.00      0.00      0.00         1
        c11       0.00      0.00      0.00         0
        c12       0.00      0.00      0.00         4
        c13       0.00      0.00      0.00         4
        c14       0.00      0.00      0.00         1
        c15       0.00      0.00      0.00         1
        c16       0.00      0.00      0.00         3
        c17       0.00      0.00      0.00         2
        c18       0.00      0.00      0.00         2
        c19       0.00      0.00      0.00         2
        c20       0.00      0.00      0.00         1
        c21       0.00      0.00      0.00         2
        c22       0.00      0.00      0.00         3
        c23       0.00      0.00      0.00         2
        c24       0.00      0.00      0.00         3
        c25       0.00      0.00      0.00         3
        c26       0.00      0.00      0.00         2
        c27       0.00      0.00      0.00         2
        c28       0.00      0.00      0.00         0
        c29       0.00      0.00      0.00         2
        c30       0.00      0.00      0.00         2
        c31       0.00      0.00      0.00         3
        c32       0.00      0.00      0.00         2
        c33       0.00      0.00      0.00         2
        c34       0.00      0.00      0.00         0
        c35       0.00      0.00      0.00         2
        c36       0.00      0.00      0.00         3
        c37       0.00      0.00      0.00         1
        c38       0.00      0.00      0.00         2
        c39       0.00      0.00      0.00         2

avg / total       0.00      0.00      0.00        80
```

40个类别里，查准率、召回率、F1 Score全为0，不能有更差的预测结果了。为什么？哪里出了差错？

答案是，我们把每个像素都作为一个输入特征来处理，这样的数据噪声太严重了，模型根本没有办法对训练数据集进行拟合。想想看，我们总共有4096个特征，可是数据集大小才400个，比特征个数还少，而且我们还需要把数据集分出20%来作为测试数据集，这样训练数据集就更小了。这样的状况下，模型根本无法进行准确地训练和预测。

### 4.3 使用PCA来处理数据集

解决上述问题的一个办法是使用 PCA 来给数据降维，只选择前k个最重要的特征。问题来了，选择多少个特征合适呢？即怎么确定k的值？PCA 算法可以通过下面的公式来计算失真幅度：
$$
\frac{\frac{1}{m} \sum_{i=1}^{m}\left\|x^{(i)}-x_{a p p r o x}^{(i)}\right\|^{2}}{\frac{1}{m} \sum_{i=1}^{m}\left\|x^{(i)}\right\|}
$$
在scikit-learn里，可以从PCA模型的explained_variance_ratio_变量里获取经PCA处理后的数据还原率。这是一个数组，所有元素求和即可知道我们选择的k值的数据还原率，数值越大说明失真越小，随着k值的增大，数值会无限接近于1。

利用这一特性，可以让k取值10~300之间，每隔30进行一次取样。在所有的k值样本下，计算经过PCA算法处理后的数据还原率。然后根据数据还原率要求，来确定合理的k值。针对我们的情况，选择失真度小于5%，即PCA处理后能保留95%的原数据信息。其代码如下：

```python
from sklearn.decomposition import PCA
print("Exploring explained variance ratio for dataset ...")
candidate_components = range(10,300,30)
explained_ratios = []
start = time()
for c in candidate_components:
    pca = PCA(n_components=c)
    X_pca = pca.fit_transform(X)
    explained_ratios.append(np.sum(pca.explained_variance_ratio_))
print('Done in {0:.2f}s'.format(time()-start))
```

输出如下：

```python
Exploring explained variance ratio for dataset ...
Done in 0.75s
```

根据不同的k值，构建PCA模型，然后调用fit_transform()函数来处理数据集，再把模型处理后数据还原率，放入explained_ratios数组。接着把这个数组画出来：

```python
plt.figure(figsize=(10,6),dpi=144)
plt.grid()
plt.plot(candidate_components,explained_ratios)
plt.xlabel('Number of PCA Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained variance ratio for PCA')
plt.yticks(np.arange(0.5,1.05,0.05))
plt.xticks(np.arange(0,300,20))
```

![image-20200716163203114](https://cdn.jsdelivr.net/gh/dongzhougu/imageuse1/image-20200716163203114.png)

上图中横坐标表示k值，纵坐标表示数据还原率。从图中可以看出，要保留95%以上的数据还原率，k值选择140即可。根据上图，也可以非常容易地找出不同的数据还原率所对应的k值。为了更直观地观察和对比在不同数据还原率下的数据，我们选择数据还原率分别在95%、90%、80%、70%、60%的情况下，这些数据还原率对应的k值分别是140、75、37、19、8，画出经PCA处理后的图片。

### 4.4 最终结果

接下来问题就变得简单了。我们选择k=140作为PCA参数，对训练数据集和测试数据集进行特征提取。

```python
n_components = 140

print("Fitting PCA by using training data ...")
start = time()
pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X_train)
print("Done in {0:.2f}s".format(time() - start))

print("Projecting input data for PCA ...")
start = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("Done in {0:.2f}s".format(time() - start))
```

输出如下：

```python
Fitting PCA by using training data ...
Done in 0.08s
Projecting input data for PCA ...
Done in 0.01s
```

接着使用 `GridSearchCV` 来选择一个最佳的SVC模型参数，然后使用最佳参数对模型进行训练。

```python
from sklearn.model_selection import GridSearchCV
print("Searching the best parameters for SVC ...")
param_grid = {'C': [1, 5, 10, 50, 100],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01]}
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid, verbose=2, n_jobs=4)
clf = clf.fit(X_train_pca, y_train)
print("Best parameters found by grid search:")
print(clf.best_params_)
```

这一步执行时间比较长，因为GridSearchCV使用矩阵式搜索法，对每组参数组合进行一次训练，然后找出最好的参数的模型。我们通过设置n_jobs=4来启动4个线程并发执行，同时设置verbose=2来输出一些过程信息。最终选择出来的最佳模型参数如下：

```python
Best parameters found by grid search:
{'C': 5, 'gamma': 0.001}
```

接着使用这一模型对测试样本进行预测，并且使用confusion_matrix输出预测准确性信息。

```python
start = time()
print("Predict test dataset ...")
y_pred = clf.best_estimator_.predict(X_test_pca)
cm = confusion_matrix(y_test, y_pred, labels=range(n_targets))
print("Done in {0:.2f}.\n".format(time()-start))
print("confusion matrix:")
np.set_printoptions(threshold=np.nan)
print(cm)
```

输出如下：

```css
Predict test dataset ...
Done in 0.01.

confusion matrix:
[[1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [1 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 3 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 3 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 3 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2]]
```

从输出的对角线上的数据可以看出，大部分预测结果都正确。我们再使用classification_report输出分类报告，查看测准率，召回率及F1 Score。

```python
print(classification_report(y_test, y_pred))
```

输出如下：

```python
             precision    recall  f1-score   support

          0       0.50      1.00      0.67         1
          1       1.00      0.67      0.80         3
          2       1.00      0.50      0.67         2
          3       1.00      1.00      1.00         1
          4       0.50      1.00      0.67         1
          5       1.00      1.00      1.00         1
          6       1.00      0.75      0.86         4
          7       1.00      1.00      1.00         2
          8       1.00      1.00      1.00         4
          9       1.00      1.00      1.00         2
         10       1.00      1.00      1.00         1
         12       1.00      1.00      1.00         4
         13       1.00      1.00      1.00         4
         14       1.00      1.00      1.00         1
         15       1.00      1.00      1.00         1
         16       0.75      1.00      0.86         3
         17       1.00      1.00      1.00         2
         18       1.00      1.00      1.00         2
         19       1.00      1.00      1.00         2
         20       1.00      1.00      1.00         1
         21       1.00      1.00      1.00         2
         22       0.75      1.00      0.86         3
         23       1.00      1.00      1.00         2
         24       1.00      1.00      1.00         3
         25       1.00      0.67      0.80         3
         26       1.00      1.00      1.00         2
         27       1.00      1.00      1.00         2
         29       1.00      1.00      1.00         2
         30       1.00      1.00      1.00         2
         31       1.00      1.00      1.00         3
         32       1.00      1.00      1.00         2
         33       1.00      1.00      1.00         2
         35       1.00      1.00      1.00         2
         36       1.00      1.00      1.00         3
         37       1.00      1.00      1.00         1
         38       1.00      1.00      1.00         2
         39       1.00      1.00      1.00         2

avg / total       0.97      0.95      0.95        80
```

在总共只有400张图片，每位目标人物只有10张图片的情况下，测准率和召回率平均达到了0.95以上，这是一个非常了不起的性能。

## 5. 拓展阅读

PCA算法的推导涉及大量的线性代数的知识。张洋先生的一篇博客[《PCA的数学原理》](http://blog.codinglabs.org/articles/pca-tutorial.html)，基本上做到了从最基础的内容谈起，一步步地推导出PCA算法，值得一读。

此外，孟岩先生的几篇博客中也介绍了矩阵及其相关运算的物理含义，深入浅出，读后犹如醍醐灌顶，这些博文是[《理解矩阵》](https://blog.csdn.net/myan/article/details/647511)三篇文章

