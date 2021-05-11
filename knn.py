from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import scipy as sp 
import matplotlib.pyplot as plt
import mglearn
'''# Generic Dataset, Classification
X, y = mglearn.datasets.make_forge()
# Plotting raw data 
mglearn.discrete_scatter(X[:,0], X[:,1], y)
plt.legend(["Class 0", "Class 1"], loc = 4)
plt.xlabel('First feature')
plt.ylabel('Second feature')
# plt.show()
print('X.shape:', X.shape)

# Generic Dataset, Regression
X, y = mglearn.datasets.make_wave(n_samples = 40)
plt.plot(X, y, 'o')
plt.ylim(-3,3)
plt.xlabel('Feature')
plt.ylabel('Target')
plt.show()

# Generic Dataset, Classification
X, y = mglearn.datasets.make_forge()
X_train, X_test, y_train, y_test = train_test_split(X, y)
clf = KNeighborsClassifier(n_neighbors=2)
clf.fit(X_train, y_train)
print('Test set predictions:', clf.predict(X_test))
print('Test set accuracy: {:.2f}'.format(clf.score(X_test, y_test)))

fig, axes = plt.subplots(1,3, figsize = (10,3))

for n_neighbors, ax in zip([1,3,9], axes):
	clf = KNeighborsClassifier(n_neighbors = n_neighbors).fit(X, y)
	mglearn.plots.plot_2d_separator(clf, X, fill = True, eps = 0.5, ax=ax, alpha = 0.4)
	mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
	ax.set_title('{} neighbor(s)'.format(n_neighbors))
	ax.set_xlabel('feature 0')
	ax.set_ylabel('feature 1')
axes[0].legend(loc=3)'''

# KNN training on breast cancer data, from 1-10 nearest neighbors
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, 
								stratify = cancer.target, random_state = 66)
training_accuracy = []
test_accuracy = []
neighbors_settings = range(1,11)

for n_neighbors in neighbors_settings:
	clf = KNeighborsClassifier(n_neighbors = n_neighbors)
	clf.fit(X_train, y_train)
	training_accuracy.append(clf.score(X_train, y_train))
	test_accuracy.append(clf.score(X_test, y_test))

plt.plot(neighbors_settings, training_accuracy, label = 'training accuracy')
plt.plot(neighbors_settings, test_accuracy, label = 'test accuracy')
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.legend()
plt.show()