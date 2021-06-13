import numpy as np 
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
import pandas as pd
from sklearn.model_selection import train_test_split

def viz(n):
    plt.imshow(x[n].reshape(28,28))
    plt.show()
    return

mnist=fetch_openml(name="mnist_784")
print(mnist)
print(len(mnist["data"]))

x=mnist['data']
y=mnist['target']

y=y.astype("float")


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,shuffle=True,random_state=42)

num_split=60000
x_train_x_test,y_train,y_test=x[:num_split],x[num_split:],y[:,num_split],y[num_split:]

#shuffling dataset
shuffle_index=np.random.permutation(num_split)
x_train,y_train=x_train[shuffle_index],y_train[shuffle_index]



x_train_0=(y_train == 0)
y_test_0=(y_test == 0)

#SGD Classifier training
from sklearn.linear_model import SGDClassifier
clf=SGDClassifier(random_state=0)
clf.fit(x_train,y_train_0)

#Prediction
viz(1000)
print(clf.predict(x[1000].reshape(1,-1)))
viz(2000)
print(clf.predict(x[2000].reshape(1,-1)))
