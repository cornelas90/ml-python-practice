from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
import pandas as pd
import numpy as np
import scipy as sp 
import matplotlib.pyplot as plt
import mglearn
'''
## Generic Dataset, Linear Model
X, y = mglearn.datasets.make_wave(n_samples=60)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)
lr = LinearRegression().fit(X_train, y_train)
#plt.show()
print('Training set score: {:.2f}'.format(lr.score(X_train, y_train)))
print('Test set score: {:.2f}'.format(lr.score(X_test, y_test)))
'''
'''
## Boston Linear Model (high dimensional data == poorer linear fit)
X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
lr = LinearRegression().fit(X_train, y_train)
print('Training set score: {:.2f}'.format(lr.score(X_train, y_train)))
print('Test set score: {:.2f}'.format(lr.score(X_test, y_test)))
#plt.show()


## Boston Ridge Regression Model (L2 regularization forces some features near zero. better fit for higher dim. data)
X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
lr = LinearRegression().fit(X_train, y_train)
ridge = Ridge().fit(X_train, y_train)
ridge10 = Ridge(alpha = 10).fit(X_train, y_train)
ridge01 = Ridge(alpha = 0.1).fit(X_train, y_train)
print('\nShowing results of Ridge Regression Model on Boston Housing Data:')
print('Training set score: {:.2f}'.format(ridge.score(X_train, y_train)))
print('Test set score: {:.2f}'.format(ridge.score(X_test, y_test)))

plt.plot(ridge.coef_, 's', label = 'Ridge alpha = 1')
plt.plot(ridge10.coef_, '^', label = 'Ridge alpha = 10')
plt.plot(ridge01.coef_, 'v', label = 'Ridge alpha = .1')

plt.plot(lr.coef_, 'o', label = 'Linear Regression')
plt.xlabel('Coefficient Index')
plt.ylabel('Coefficient Nagnitude')
plt.hlines(0,0,len(lr.coef_))
plt.ylim(-25,25)
plt.legend()
#plt.show()

## Boston Lasso (L1 regularization, zeros some coefficients entirely, worst fit )
lasso = Lasso().fit(X_train, y_train)
print('\nShowing results of Lasso Model on Boston Housing Data:')
print('Training set score: {:.2f}'.format(lasso.score(X_train, y_train)))
print('Test set score: {:.2f}'.format(lasso.score(X_test, y_test)))
print('Number of features used:', np.sum(lasso.coef_ !=0))

## better fit with smaller alpha /  more features used
lasso001 = Lasso(alpha = 0.01, max_iter=100000).fit(X_train, y_train)
print('\nShowing results of Lasso Model on Boston Housing Data:')
print('Training set score: {:.2f}'.format(lasso001.score(X_train, y_train)))
print('Test set score: {:.2f}'.format(lasso001.score(X_test, y_test)))
print('Number of features used:', np.sum(lasso001.coef_ !=0))

## Even smaller alpha /  more features used
lasso0001 = Lasso(alpha = 0.001, max_iter=100000).fit(X_train, y_train)
print('\nShowing results of Lasso Model on Boston Housing Data:')
print('Training set score: {:.2f}'.format(lasso0001.score(X_train, y_train)))
print('Test set score: {:.2f}'.format(lasso0001.score(X_test, y_test)))
print('Number of features used:', np.sum(lasso0001.coef_ !=0))

plt.plot(lasso.coef_, 's', label = 'Ridge alpha = 1')
plt.plot(lasso001.coef_, '^', label = 'Ridge alpha = 0.01')
plt.plot(lasso0001.coef_, 'v', label = 'Ridge alpha = 0.0001')
plt.plot(ridge01.coef_, 'o', label = 'Ridge alpha = 0.1')
plt.legend(ncol = 2, loc = (0, 1.05))
plt.ylim(-25,25)
plt.xlabel('Coefficient Index')
plt.ylabel('Coefficient Magnitude')
#plt.show()

##  Logistic Regression (Remember, this is a classificaiton algo)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

X, y = mglearn.datasets.make_forge()
fig, axes = plt.subplots(1,2, figsize=(10, 3))
for model, ax in zip([LinearSVC(), LogisticRegression()], axes):
	clf = model.fit(X, y)
	mglearn.plots.plot_2d_separator(clf, X, fill=False, eps = 0.5, ax=ax, alpha = .7)
	mglearn.discrete_scatter(X[:,0], X[:,1], y, ax=ax)
	ax.set_title(clf.__class__.__name__)
	ax.set_xlabel('Feature 0')
	ax.set_ylabel('Feature 1')

axes[0].legend()

mglearn.plots.plot_linear_svc_regularization()

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify = cancer.target, random_state=42)
logreg = LogisticRegression().fit(X_train, y_train)
print('Training set score: {:.3f}'.format(logreg.score(X_train, y_train)))
print('Test set score: {:.3f}'.format(logreg.score(X_test, y_test)))
##  Increasing C
logreg100 = LogisticRegression(C=100).fit(X_train, y_train)
print('Training set score: {:.3f}'.format(logreg100.score(X_train, y_train)))
print('Test set score: {:.3f}'.format(logreg100.score(X_test, y_test)))
##  Decreasing C
logreg001 = LogisticRegression(C=0.001).fit(X_train, y_train)
print('Training set score: {:.3f}'.format(logreg001.score(X_train, y_train)))
print('Test set score: {:.3f}'.format(logreg001.score(X_test, y_test)))

## Compare feature weights for all three C's
plt.plot(logreg.coef_.T, 'o', label = 'C=1')
plt.plot(logreg100.coef_.T, '^', label = 'C=100')
plt.plot(logreg001.coef_.T, 'v', label = 'C=0.001')
plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation = 90)
plt.hlines(0, 0, cancer.data.shape[1])
plt.ylim(-5, 5)
plt.xlabel('Feature')
plt.ylabel('Coefficient Magnitude')
plt.legend()

##  Using L1 regularization for easier reading, fewer models
for C, marker in zip([0.001, 1, 100], ['o', '^', 'v']):
	lr_l1 = LogisticRegression(C=C, penalty = 'l1', solver='liblinear').fit(X_train, y_train)
	print('Training accuracy of l1 logreg with C={:.3f}: {:.2f}'.format(C, lr_l1.score(X_train, y_train)))
	print('Test accuracy of l1 logreg with C={:.3f}: {:.2f}'.format(C, lr_l1.score(X_test, y_test)))
	plt.plot(lr_l1.coef_.T, marker, label='C={:.3f}'.format(C))
plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation = 90)
plt.hlines(0, 0, cancer.data.shape[1])
plt.xlabel('Feature')
plt.ylabel('Coefficient Magnitude')

plt.ylim(-5, 5)
plt.legend(loc=3)
'''
## Linear Models for Multiclass Classification
from sklearn.datasets import make_blobs

X, y = make_blobs(random_state = 42)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')
plt.legend(['Class 0', 'Class 1', 'Class 2'])
mglearn.discrete_scatter(X[:, 0], X[:,1], y)
mglearn.plots.plot_2d_classification(linear_svm, X, fill = True, alpha = .7)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
line = np.linspace(-15,15)
for coef, intercept, color in zip(linear_svm.coef_, linear_svm.intercept_, mglearn.cm3.colors):
    plt.plot(line, -(line * coef[0] + intercept)/ coef[1], c=color)
plt.ylim(-10, 15)
plt.xlim(-10, 8)
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')
plt.legend(['Class 0', 'Class 1', 'Class 2', 'Line Class 0', 'Line Class 1', 'Line Class 2'], loc=(1.01, 0.3))
