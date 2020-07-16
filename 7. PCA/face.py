import logging
import sys
from sklearn.datasets import fetch_olivetti_faces
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from time import time
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV


def load_dataset():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    data_home = 'dataset/'
    logging.info('Start to load dataset')
    faces = fetch_olivetti_faces(data_home=data_home)
    logging.info('Done with load dataset')
    return faces


def draw(faces):
    X = faces.data
    y = faces.target
    targets = np.unique(faces.target)
    target_names = np.array(["c%d" % t for t in targets])
    n_targets = target_names.shape[0]
    n_samples, h, w = faces.images.shape
    print('Sample count: {}\nTarget count: {}'.format(n_samples, n_targets))
    print('Image size: {}x{}\nDataset shape: {}\n'.format(w, h, X.shape))

    n_row = 2
    n_col = 6

    sample_images = None
    sample_titles = []
    for i in range(n_targets):
        people_images = X[y == i]
        people_sample_index = np.random.randint(0, people_images.shape[0], 1)
        people_sample_image = people_images[people_sample_index, :]
        if sample_images is not None:
            sample_images = np.concatenate((sample_images, people_sample_image), axis=0)
        else:
            sample_images = people_sample_image
        sample_titles.append(target_names[i])

    plot_gallery(sample_images, sample_titles, h, w, n_row, n_col)
    return X, y, n_targets, target_names


def plot_gallery(images, titles, h, w, n_row=2, n_col=5):
    """显示图片阵列"""
    plt.figure(figsize=(2 * n_col, 2 * n_row), dpi=144)
    plt.subplots_adjust(bottom=0, left=0.01, right=0.99, top=0.90, hspace=0.01)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i])
        plt.axis('off')
    plt.show()


def train(X, y, n_targets, target_names):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
    start = time()
    print('Fitting train datasets ...')
    clf = SVC(class_weight='balanced')
    clf.fit(X_train, y_train)
    print('Done in {0:.2f}s'.format(time() - start))

    start = time()
    print('Predicting test dataset ...')
    y_pred = clf.predict(X_test)
    print('Done in {0:.2f}s'.format(time() - start))

    cm = confusion_matrix(y_test, y_pred, labels=range(n_targets))
    print('confusion matrix:\n')
    np.set_printoptions(threshold=sys.maxsize)
    print(cm)

    print(classification_report(y_test, y_pred))


def pca_train(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
    print("Exploring explained variance ratio for dataset ...")
    candidate_components = range(10, 300, 30)
    explained_ratios = []
    start = time()
    for c in candidate_components:
        pca = PCA(n_components=c)
        X_pca = pca.fit_transform(X)
        explained_ratios.append(np.sum(pca.explained_variance_ratio_))
    print('Done in {0:.2f}s'.format(time() - start))

    plt.figure(figsize=(10, 6), dpi=144)
    plt.grid()
    plt.plot(candidate_components, explained_ratios)
    plt.xlabel('Number of PCA Components')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained variance ratio for PCA')
    plt.yticks(np.arange(0.5, 1.05, 0.05))
    plt.xticks(np.arange(0, 300, 20))
    plt.show()

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

    print("Searching the best parameters for SVC ...")
    param_grid = {'C': [1, 5, 10, 50, 100],
                  'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01]}
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid, verbose=2, n_jobs=4)
    clf = clf.fit(X_train_pca, y_train)
    print("Best parameters found by grid search:")
    print(clf.best_params_)

    start = time()
    print("Predict test dataset ...")
    y_pred = clf.best_estimator_.predict(X_test_pca)
    cm = confusion_matrix(y_test, y_pred, labels=range(n_targets))
    print("Done in {0:.2f}.\n".format(time() - start))
    print("confusion matrix:")
    np.set_printoptions(threshold=sys.maxsize)
    print(cm)

    print(classification_report(y_test, y_pred))


if __name__ == '__main__':
    faces = load_dataset()
    X, y, n_targets, target_names = draw(faces)
    train(X, y, n_targets, target_names)
    pca_train(X,y)
