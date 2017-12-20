# import dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# read the iris dataset
def load_iris_dataset():
    # read data from csv file
    df = pd.read_csv("./iris.data", header=None, sep=',')
    # put in column names
    df.columns = ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
    # drop the empty line at end-of-file
    df.dropna(how="all", inplace=True)
    return df


# split the DataFrame into data X and class y
def split_dataset(df):
    X = df.ix[:, 0:4].values
    y = df.ix[:, 4].values
    return X, y


# plot histograms of 3 different classes
def plot_histogram(X, y):

    # feature dictionary
    feature_dict = {0: 'sepal length [cm]',
                    1: 'sepal width [cm]',
                    2: 'petal length [cm]',
                    3: 'petal width [cm]'}

    with plt.style.context('seaborn-whitegrid'):
        plt.figure(figsize=(8, 6))
        for index_feature in range(4):
            plt.subplot(2, 2, index_feature + 1)
            for label in ('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'):
                plt.hist(X[y == label, index_feature],
                         label=label,
                         bins=10,
                         alpha=0.3, )
            plt.xlabel(feature_dict[index_feature])
        plt.legend(loc='upper right', fancybox=True, fontsize=8)

        plt.tight_layout()
        plt.show()


# standardize X
def standardize(X):
    X_standardized = StandardScaler().fit_transform(X)
    return X_standardized


# calculate covariance matrix
def calculate_covariance_matrix(X_standardized):
    covariance_matrix = np.cov(X_standardized.T)
    print('NumPy covariance matrix: \n%s' % covariance_matrix)
    return covariance_matrix


# calculate correlation matrix
def calculate_correlation_matrix(X_standardized):
    correlation_matrix = np.corrcoef(X_standardized)
    print('NumPy correlation matrix \n%s' % correlation_matrix)
    return correlation_matrix


# perform eigen decomposition
def eigen_decomposition(X_standardized, method='covariance'):
    if method == 'covariance':
        matrix = calculate_covariance_matrix(X_standardized=X_standardized)
    elif method == 'correlation':
        matrix = calculate_correlation_matrix(X_standardized=X_standardized)
    else:
        matrix = None
    eigen_values, eigen_vectors = np.linalg.eig(matrix)
    print('Eigenvalues \n%s' % eigen_values)
    print('Eigenvectors \n%s' % eigen_vectors)
    return eigen_values, eigen_vectors


# perform singular value decomposition
def singular_value_decomposition(X_standardized):
    u, s, v = np.linalg.svd(X_standardized.T)
    print('SVD vector u: \n%s' % u)
    print('SVD vector s: \n%s' % s)
    print('SVD vector v: \n%s' % v)
    return u, s, v


# sort eigen pairs
def sort_eigen_pairs(eigen_values, eigen_vectors):

    # make a list of (eigen value, eigen vector) tuples
    eigen_pairs = list()
    for i in range(len(eigen_values)):
        eigen_pairs.append([np.abs(eigen_values[i]), eigen_vectors[:, i]])

    # sort the (eigen value, eigen vector) tuples in descending order
    eigen_pairs.sort(key=lambda x: x[0], reverse=True)

    # visually verify descending order
    print("Eigen values in descending order:")
    for i in range(len(eigen_values)):
        print(eigen_pairs[i][0])

    return eigen_pairs


def main():
    df = load_iris_dataset()
    X, y = split_dataset(df)
    plot_histogram(X, y)
    X_standardized = standardize(X)
    eigen_values, eigen_vectors = eigen_decomposition(X_standardized, method='covariance')
    eigen_pairs = sort_eigen_pairs(eigen_values, eigen_vectors)


main()
