import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

#Data preperation 
x = 10*np.random.rand(100)
y = 3*x+np.random.rand(100)
print(x,y)
plt.scatter(x, y)
plt.show()

#choose class of model
model=LinearRegression(fit_intercept=True)

#arrange data in matrix
X=x.reshape(-1,1)

#fit model
model.fit(X,y)
print(model.coef_,model.intercept_)

#Data preperation for prediction
x_fit=np.linspace(-1,11)
X_fit=x_fit.reshape(-1,1)

#prediction
y_fit=model.predict(X_fit)
plt.scatter(x,y)
plt.plot(x_fit,y_fit)
plt.show()
