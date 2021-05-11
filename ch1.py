from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import scipy as sp 
import matplotlib.pyplot as plt
import mglearn
#Create Training and Test sets
iris_dataset = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)
#Plot raw data to get 'big picture', and ask yourself if ML is appropriate for the problemset. 
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker = 'o', hist_kwds={'bins':20}, s=60, alpha=.8, cmap = mglearn.cm3)
#plt.show()
#KNN modelling
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
KNeighborsClassifier(algorithm = 'auto', leaf_size = 30, metric = 'minkowski',
					metric_params = None, n_jobs = None, n_neighbors = 1, p = 2, weights = 'uniform')
#pop quiz
X_new = np.array([[5, 2.9, 1, 0.2]])
print('X_new.shape:', X_new.shape)
prediction = knn.predict(X_new)
print('Prediction:', prediction)
print('Predicted target name:', iris_dataset['target_names'][prediction])
#Testing real data
y_pred = knn.predict(X_test)
print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))