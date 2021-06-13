import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

dataset=[[-2.0011,0],
         [-1.4654,0],
         [0.0965,0],
         [1.3881,0],
         [3.0641,0],
         [7.6275,1],
         [5.3324,1],
         [6.9225,1],
         [8.6754,1],
         [7.6737,1]] #data in 1 dimensional

x=np.array(dataset) [:, 0:1]
y=np.array(dataset) [: ,1]

LR=LogisticRegression(C=1.0,penalty='l2',tol=0.0001,solver="lbfgs")
LR.fit(x,y)

y_pred=LR.predict(np.array([-0.0000006]).reshape(1,-1))
print(y_pred)

