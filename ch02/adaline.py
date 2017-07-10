import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plot import *
from numpy.random import seed


class AdalineGD(object):
    """
    Adaptive Linear Neuron classifier.

    Parameters
    -------------
    eta: float
        Learning rate (between 0.0 and 1.0)
    n_iter: int
        Passes over the trainging dataset.

    Attributes
    -------------
    w_: ld-array
        weight after fitting
    errors_: list
        Number of misclassification in every epoch.
    
    """
    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter
        self.w_ = None
        self.errors_ = []
        self.cost_ = []

    def fit(self, x, y):
        '''
        Fitting training data.

        Parameters
        -----------------
        x: shape = [n_samples, n_features]
        y: shape = [n_samples]

        Returns
        ------------------
        self: object
        '''
        self.w_ = np.zeros(1 + x.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(x)
            errors = y - output
            self.w_[1:] += self.eta * x.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2
            self.cost_.append(cost)

        return self

    def predict(self, x):
        '''Return class label'''
        return np.where(self.net_input(x) >= 0.0, 1, -1)

    def net_input(self, x):
        '''Calculate the net input'''
        return np.dot(x, self.w_[1:]) + self.w_[0]


class AdalineSGD(object):
    """
    Adaptive Linear Neuron classifier.

    Parameters
    -------------
    eta: float
        Learning rate (between 0.0 and 1.0)
    n_iter: int
        Passes over the trainging dataset.
    shuffle: bool (default True)
        Shuffles training dataset.
    random_state: int(default None)
        Set random state for shuffling and initializing the weights.

    Attributes
    -------------
    w_: ld-array
        weight after fitting
    costs_: list
        Number of misclassification in every epoch.
    
    """
    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_ = None
        self.cost_ = []
        self.shuffle = shuffle
        if random_state:
            seed(random_state)
        self.w_initialized = False

    def fit(self, x, y):
        '''
        Fitting training data.

        Parameters
        -----------------
        x: shape = [n_samples, n_features]
        y: shape = [n_samples]

        Returns
        ------------------
        self: object
        '''
        self._initialize_weights(x.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            if self.shuffle:
                x, y = self._shuffle(x, y)
            cost = []
            for xi, target in zip(x, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)

        return self

    def partial_fit(self, x, y):
        '''Fit the data without reinitializing the weights'''
        if not self._initialize_weights:
            self._initialize_weights(x.shape[1])
        if y.shape[0] > 1:
            for xi, target in zip(x, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(x, y)
        return self

    def predict(self, x):
        '''Return class label'''
        return np.where(self.net_input(x) >= 0.0, 1, -1)

    def net_input(self, x):
        '''Calculate the net input'''
        return np.dot(x, self.w_[1:]) + self.w_[0]

    def _shuffle(self, x, y):
        '''Shuffle training dataset'''
        r = np.random.permutation(len(y))
        return x[r], y[r]

    def _initialize_weights(self, m):
        '''Initialize weights to zeros'''
        self.w_ = np.zeros(1 + m)
        self._initialize_weights = True

    def _update_weights(self, xi, target):
        '''Update weights'''
        output = self.net_input(xi)
        error = target - output
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error ** 2
        return cost


if __name__ == '__main__':
    DF = pd.read_csv(
        'https://archive.ics.uci.edu/ml/'+
        'machine-learning-databases/iris/iris.data', header=None)
    Y = DF.iloc[0:100, 4].values
    Y = np.where(Y == 'Iris-setosa', -1, 1)
    X = DF.iloc[0:100, [0, 2]].values
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8,4))
    ada1 = AdalineGD(n_iter=10, eta=0.01).fit(X,Y)
    ax[0].plot(range(1, len(ada1.cost_)+1), np.log10(ada1.cost_), marker='o')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('log(Sum-squared-errors)')
    ax[0].set_title('Adaline - Learning rate 0.01')
    ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(X,Y)
    ax[1].plot(range(1, len(ada2.cost_)+1), ada2.cost_, marker='o')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Sum-squared-errors')
    ax[1].set_title('Adaline - Learning rate 0.0001')
    plt.show()

    X_std = X.copy()
    X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
    X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
    ada = AdalineSGD(n_iter=15, eta=0.01, shuffle=True)
    ada.fit(X_std, Y)
    plot_descision_regions(X_std, Y, classifier=ada)
    plt.title('Adaline - Gradient Descent')
    plt.xlabel('seqal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.legend(loc='upper left')
    plt.show()

    plt.plot(range(1, len(ada.cost_)+1), ada.cost_, marker='o')
    plt.show()
