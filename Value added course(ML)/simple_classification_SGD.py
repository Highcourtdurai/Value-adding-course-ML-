import numpy as np
import sklearn
from sklearn.datasets import fetch_openml
mnist = fetch_openml(name='mnist_784')
print(mnist)
len(mnist['data'])
X, y = mnist['data'], mnist['target']
print(X)
y = y.astype("float")
print(y)

#visualization
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
def viz(n):
    plt.imshow(X[n].reshape(28,28))
    plt.show()
    return


#splitting train , test sets method 1
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

#splitting train , test sets method 2
num_split = 60000
X_train, X_test, y_train, y_test = X[:num_split], X[num_split:], y[:num_split], y[num_split:]

#shuffling dataset
shuffle_index = np.random.permutation(num_split)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

#binary classifier (convert dataset to zero and non zero)
y_train_0 = (y_train == 0)
y_test_0 = (y_test == 0)

#SGD Classifier training
from sklearn.linear_model import SGDClassifier
clf = SGDClassifier(random_state = 0)
clf.fit(X_train, y_train_0)

#Prediction
viz(1000)
print(clf.predict(X[1000].reshape(1, -1)))
viz(2000)
print(clf.predict(X[2000].reshape(1, -1)))
