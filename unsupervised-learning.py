##  MinMaxScaler preprocessing for Breast Cancer Dataset, Introduction of methods
import matplotlib.pyplot as plt
import numpy as np
import mglearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=1)
print(X_train.shape)
print(X_test.shape)

##  Shifting and Scaling dataset
scaler = MinMaxScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
print('Transformed Shape: {}'.format(X_train_scaled.shape))
print('per-feature minimum before scaling:\n {}'.format(X_train.min(axis=0)))
print('per-feature maximum before scaling:\n {}'.format(X_train.max(axis=0)))
print('per-feature minimum after scaling:\n {}'.format(X_train_scaled.min(axis=0)))
print('per-feature maximum after scaling:\n {}'.format(X_train_scaled.max(axis=0)))

##  Applying transformation to test data
X_test_scaled = scaler.transform(X_test)
print('per-feature minimum after scaling:\n {}'.format(X_test_scaled.min(axis=0)))
print('per-feature maximum after scaling:\n {}'.format(X_test_scaled.max(axis=0)))

##  Transform applies same arithmetic from train data to test data, hence values outside of 0-1 scale
##  Comparing differences between original data, scaled data, and improperly scaled data (using test set transform on training set)

##  Original data and visualization
from sklearn.datasets import make_blobs
X, _ = make_blobs(n_samples=50, centers=5, random_state=4, cluster_std=2)
X_train, X_test = train_test_split(X, random_state=5, test_size=.1)


fig, axes = plt.subplots(1, 3, figsize=(13, 4))
axes[0].scatter(X_train[:, 0], X_train[:, 1], c=mglearn.cm2(0), label='Training set', s=60)
axes[0].scatter(X_test[:, 0], X_test[:, 1], marker='^', c=mglearn.cm2(1), label='Test set', s=60)
axes[0].legend(loc='upper left')
axes[0].set_title('Original Data')


##  Properly scaling Data and visualization
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

axes[1].scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=mglearn.cm2(0), label='Training set', s=60)
axes[1].scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], marker='^', c=mglearn.cm2(1), label='Test set', s=60)
axes[1].set_title('Scaled Data')

##  Incorrect, big dummy scaling where we set test datset with min / max to 0-1
test_scaler = MinMaxScaler()
test_scaler.fit(X_test)
X_test_scaled_badly = test_scaler.transform(X_test)

axes[2].scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=mglearn.cm2(0), label='Training set', s=60)
axes[2].scatter(X_test_scaled_badly[:, 0], X_test_scaled_badly[:, 1], marker='^', c=mglearn.cm2(1), label='Test set', s=60)
axes[2].set_title('Imposter Data')

## SVM applied to scaled breast cancer data
from sklearn.svm import SVC
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
svm = SVC(C=100)
svm.fit(X_train, y_train)
print('Test set accuracy: {:.2f}'.format(svm.score(X_test, y_test)))

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
svm.fit(X_train_scaled, y_train)
print('Scaled test set accuracy: {:.2f}'.format(svm.score(X_test_scaled, y_test)))

##  Testing StandardScaler (Increase from original data, not as much as MinMaxScale)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
svm.fit(X_train_scaled, y_train)
print('SVM test accuracy: {:.2f}'.format(svm.score(X_test_scaled, y_test)))

##  PCA visualization	

fig, axes = plt.subplots(15, 2, figsize=(10, 20))
malignant = cancer.data[cancer.target == 0]
benign = cancer.data[cancer.target == 1]
ax = axes.ravel()
for i in range(30):
	_, bins = np.histogram(cancer.data[:, i], bins=50)
	ax[i].hist(malignant[:, i], bins=bins, color=mglearn.cm3(0), alpha=.5)
	ax[i].hist(benign[:, i], bins=bins, color=mglearn.cm3(2), alpha=.5)
	ax[i].set_title(cancer.feature_names[i])
	ax[i].set_yticks(())
ax[0].set_xlabel('Feature Magnitude')
ax[0].set_ylabel('Frequency')
ax[0].legend(['malignant', 'benign'], loc='best')
fig.tight_layout()
plt.show

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

scaler = StandardScaler()
scaler.fit(cancer.data)
X_scaled = scaler.transform(cancer.data)
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
pca.fit(X_scaled)

X_pca = pca.transform(X_scaled)
print('Original Shape: {}'.format(str(X_scaled.shape)))
print('Reduced Shape: {}'.format(str(X_pca.shape)))

plt.figure(figsize=(8, 8))
mglearn.discrete_scatter(X_pca[:, 0], X_pca[:, 1], cancer.target)
plt.legend(cancer.target_names, loc='best')
plt.gca().set_aspect('equal')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')

print('PCA componet shape: {}'.format(pca.components_.shape))
print('PCA components: \n{}'.format(pca.components_))

##  Visualizing this information
plt.matshow(pca.components_, cmap='viridis')
plt.yticks([0, 1], ['First Component', 'Second Component'])
plt.colorbar()
plt.xticks(range(len(cancer.feature_names)), cancer.feature_names, rotation=60, ha='left')
plt.xlabel('Feature')
plt.ylabel('Principal Components')

##  Celeb Faces Example

from sklearn.datasets import fetch_lfw_people
people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
image_shape = people.images[0].shape

fig, axes = plt.subplots(2, 5, figsize=(15, 8), subplot_kw={'xticks':(), 'yticks':()})
##  Example of images in dataset
for target, image, ax in zip(people.target, people.images, axes.ravel()):
	ax.imshow(image)
	ax.set_title(people.target_names[target])
print('people.images.shape: {}'.format(people.images.shape))
print('Number of Classes: {}'.format(len(people.target_names)))
##  Shows how many pictures of each person are included
counts = np.bincount(people.target)
for i, (count, name) in enumerate(zip(counts, people.target_names)):
	print('{0:25} {1:3}'.format(name, count), end='	')
	if (i+1) % 3 ==0:
		print()
##  Limit number of images per person
mask = np.zeros(people.target.shape, dtype=np.bool)
for target in np.unique(people.target):
	mask[np.where(people.target == target)[0][:50]] = 1
X_people = people.data[mask]
y_people = people.target[mask]
X_people = X_people / 255

from sklearn.neighbors import KNeighborsClassifier
X_train, X_test, y_train, y_test = train_test_split(X_people, y_people, stratify=y_people, random_state=0)
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
print('Test set score of 1-nn: {}'.format(knn.score(X_test, y_test)))
##  Whiten for increase of about 7%
pca = PCA(n_components=100, whiten=True, random_state=0).fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print('X_train_pca.shape: {}'.format(X_train_pca.shape))

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train_pca, y_train)
print('Test set accuracy: {:.2f}'.format(knn.score(X_test_pca, y_test)))
print('pca.components_.shape: {}'.format(pca.components_.shape))

fig, axes = plt.subplots(3, 5, figsize=(15, 12), subplot_kw={'xticks':(), 'yticks':()})
for i, (component, ax) in enumerate(zip(pca.components_, axes.ravel())):
	ax.imshow(component.reshape(image_shape), cmap='viridis')
	ax.set_title('{}. component'.format((i+1)))

##  Non-Negative Matrix Factorization (NMF)
from sklearn.decomposition import NMF
nmf = NMF(n_components=15, random_state=0)
nmf.fit(X_train)
X_train_nmf = nmf.transform(X_train)
X_test_nmf = nmf.transform(X_test)

fig, axes = plt.subplots(3, 5, figsize=(15, 12), subplot_kw={'xticks': (), 'yticks': ()})
for i, (component, ax) in enumerate(zip(nmf.components_, axes.ravel())):
	ax.imshow(component.reshape(image_shape))
	ax.set_title('{}. component'.format(i))

compn = 3
inds = np.argsort(X_train_nmf[:, compn])[::-1]
fig, axes = plt.subplots(2, 5, figsize=(15, 8), subplot_kw={'xticks': (), 'yticks': ()})
fig.suptitle('Large component 3')
for i, (ind, ax) in enumerate(zip(inds, axes.ravel())):
	ax.imshow(X_train[ind].reshape(image_shape))

compn = 7
inds = np.argsort(X_train_nmf[:, compn])[::-1]
fig, axes = plt.subplots(2, 5, figsize=(15, 8), subplot_kw={'xticks': (), 'yticks': ()})
fig.suptitle('Large component 7')
for i, (ind, ax) in enumerate(zip(inds, axes.ravel())):
	ax.imshow(X_train[ind].reshape(image_shape))

##  Manifold Learning Algorithms - Allow for visualization with more complex mapping than PCA
##  t-SNE manifold learning algorithm

from sklearn.datasets import load_digits
digits = load_digits()
fig, axes = plt.subplots(2, 5, figsize=(10, 5), subplot_kw={'xticks': (), 'yticks': ()})
for ax, img in zip(axes.ravel(), digits.images):
	ax.imshow(img)
##  Building PCA model
pca = PCA(n_components=2)
pca.fit(digits.data)
digits_pca = pca.transform(digits.data)

colors = ['#476A2A', '#7851B8', '#BD3430', '#4A2D4E', '#875525', '#A83683', '#4E655E', '#853541', '#535D8E']
plt.figure(figsize=(10, 10))
plt.xlim(digits_pca[:, 0].min(), digits_pca[:, 0].max())
plt.ylim(digits_pca[:, 1].min(), digits_pca[:, 1].max())
for i in range(len(digits.data)):
	plt.text(digits_pca[i, 0], digits_pca[i, 1], str(digits.target[i]), color = colors[digits.target[i]], fontdict={'weight': 'bold', 'size': 9})
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')