import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc [2, 3]1].values
y dataset.iloc -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_TEST)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop

X1,

plt.

plt.
plt.

for

plt.
plt.
plt.
plt.
plt.

np.arange(start

X_set[:, 1].min() - 1, stop

X_set[:, 0].max() + 1, step
X_set[:, 1].max() + 1, step

contourf(X1, X2, classifier.predict(np.array([Xl.ravel(), X2.ravel()]).T).reshape(X1.shape),
alpha = 0.75, cmap = ListedColormap(('red', 'green')))

x1im(X1.min(), Xl.max())
ylim(X2.min(), X2.max())
i, j in enumerate(np.unique(y_set)):

plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],

c = ListedColormap(('red',
title('K-NN (Training set)')
xlabel('Age')
ylabel('Estimated Salary')
legend ()
show()

'green')) (i), label = j)

non

0.01),
0.01)

                 
                 from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test

X1,

plt.

plt.
plt.

for

plt.
plt.
plt.
plt.
plt.

X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step
np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step
contourf(X1, X2, classifier.predict(np.array([Xl.ravel(), X2.ravel()]).T).reshape(X1.shape),
alpha = 0.75, cmap = ListedColormap(('red', 'green')))
xlim(X1.min(), X1l.max())
ylim(X2.min(), X2.max())
i, j din enumerate(np.unique(y_set)):
plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
c = ListedColormap(('red', 'green'))(i), label = j)
title('K-NN (Test set)')
xlabel('Age')
ylabel('Estimated Salary')
legend ()
show()

0.01),
0.01)
