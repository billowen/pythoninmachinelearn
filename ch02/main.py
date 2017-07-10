
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from perceptron import *
from plot import *

DF = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
Y = DF.iloc[0:100, 4].values
Y = np.where(Y == 'Iris-setosa', -1, 1)
X = DF.iloc[0:100, [0, 2]].values
w = np.zeros(X.shape[1])
print(np.dot(X, w))

plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc='upper left')
plt.show()

ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, Y)
plt.plot(range(1, len(ppn.errors_)+1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.show()

plot_descision_regions(X, Y, classifier=ppn);
plt.xlabel('seqal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()
