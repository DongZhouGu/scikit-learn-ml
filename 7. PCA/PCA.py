import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


def pca_np():
    A = np.array([[3, 2000],
                  [2, 3000],
                  [4, 5000],
                  [5, 8000],
                  [1, 2000]], dtype='float')
    # 数据归一化，axis=0表示按列归一化
    mean = np.mean(A, axis=0)
    norm = A - mean
    # 数据缩放
    score = np.max(norm, axis=0) - np.min(norm, axis=0)
    norm = norm / score
    U, S, V = np.linalg.svd(np.dot(norm.T, norm))
    # 只取特征矩阵的第一列（前k列）来构造主成分特征矩阵U_reduce：
    U_reduce = U[:, 0].reshape(2, 1)
    # 有了主成分特征矩阵，就可以对数据进行降维了
    R = np.dot(norm, U_reduce)

    # 还原降维后的数据
    Z = np.dot(R, U_reduce.T)
    A1 = np.multiply(Z, score) + mean  # np.multiply是矩阵对应元素相乘
    return norm, Z, U, U_reduce


def std_pca(**argv):
    scaler = MinMaxScaler()
    pca = PCA(**argv)
    pipline = Pipeline([('scaler', scaler),
                        ('pca'.pca)])
    return pipline


def sklearn_pca():
    A = np.array([[3, 2000],
                  [2, 3000],
                  [4, 5000],
                  [5, 8000],
                  [1, 2000]], dtype='float')
    pca = std_pca(n_components=1)
    R = pca.fit_transform(A)
    A1 = pca.inverse_transform(R)


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


if __name__ == '__main__':
    norm, Z, U, U_reduce=pca_np()
    draw(norm, Z, U, U_reduce)
