from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import mglearn
#import graphviz
import pandas as pd
import numpy as np
import scipy as sp 
from sklearn.tree import export_graphviz
##  Example using Breast Cancer Dataset
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify = cancer.target, random_state = 42)
tree = DecisionTreeClassifier(random_state = 0)
tree.fit(X_train, y_train)
print('Accuracy on training set: {:.3f}'.format(tree.score(X_train, y_train)))
print('Accuracy on test set: {:.3f}'.format(tree.score(X_test, y_test)))

##  Pre-Pruning using 'max depth' flag
tree = DecisionTreeClassifier(max_depth=4, random_state = 0)
tree.fit(X_train, y_train)
print('Accuracy on training set: {:.3f}'.format(tree.score(X_train, y_train)))
print('Accuracy on test set: {:.3f}'.format(tree.score(X_test, y_test)))
export_graphviz(tree, out_file = 'tree.dot', class_names=['malignant', 'benign'], feature_names=cancer.feature_names, impurity = False, filled = True)

##  Feature Importance
print('Feature Importances: \n', tree.feature_importances_)

def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align = 'center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.ylim(-1, n_features)
    plt.show

#plot_feature_importances_cancer(tree)

## looking at RAM prices, Tree Regressors
import os
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression 
## RAM ex with log scale
ram_prices = pd.read_csv(os.path.join(mglearn.datasets.DATA_PATH, 'ram_price.csv'))
plt.semilogy(ram_prices.date, ram_prices.price)
plt.xlabel('Year')
plt.ylabel('Price in $/Mbyte')
#plt.show


data_train = ram_prices[ram_prices.date < 2000]
data_test = ram_prices[ram_prices.date >= 2000]
X_train = data_train.date[:, np.newaxis]
y_train = np.log(data_train.price)
tree = DecisionTreeRegressor(max_depth=3).fit(X_train, y_train)
linear_reg = LinearRegression().fit(X_train, y_train)
X_all = ram_prices.date[:, np.newaxis]
pred_tree = tree.predict(X_all)
pred_lr = linear_reg.predict(X_all)
price_tree = np.exp(pred_tree)
price_lr = np.exp(pred_lr)

plt.semilogy(data_train.date, data_train.price, label = 'Training Data')
plt.semilogy(data_test.date, data_test.price, label = 'Test data')
plt.semilogy(ram_prices.date, price_tree, label = 'Tree Prediction')
plt.semilogy(ram_prices.date, price_lr, label = 'Linear Prediction')
plt.legend()
plt.show()

## Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons
X, y = make_moons(n_samples = 100, noise = 0.25, random_state = 3)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, random_state = 42)

forest = RandomForestClassifier(n_estimators = 5, random_state = 2)
forest.fit(X_train, y_train)

fig, axes = plt.subplots(2,3, figsize = (20, 10))
for i, (ax, tree) in enumerate(zip(axes.ravel(), forest.estimators_)):
	ax.set_title('Tree{}'.format(i))
	mglearn.plots.plot_tree_partition(X_train, y_train, tree, ax = ax)
mglearn.plots.plot_2d_separator(forest, X_train, fill = True, ax = axes[-1, -1], alpha = 0.4)
axes[-1, -1].set_title('Random Forest')
mglearn.discrete_scatter(X_train[:,0], X_train[:, 1], y_train)


## Gradient Boosting with Pre-Pruning
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
gbrt = GradientBoostingClassifier(random_state = 0, max_depth = 1)
gbrt.fit(X_train, y_train)
print('Accuracy on training set: {:.3f}'.format(gbrt.score(X_train, y_train)))
print('Accuracy on test set: {:.3f}'.format(gbrt.score(X_test, y_test)))

##  Can also pre-prune using learning_rate flag