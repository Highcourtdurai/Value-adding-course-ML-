import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

data=pd.read_csv("Salary_Data.csv")
#EDA-Explomatory data analytis
sns.lmplot(data=data,x="YearsExperience",y="Salary")
#sns.boxplot(data=data,x="YearsExperience")
sns.boxplot(data=data,x="Salary")

x=data.iloc[:,0].values.reshape(-1,1)
y=data.iloc[:,1].values.reshape(-1,1)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)

y_pred=lr.predict(x_test)

plt.plot(x_test, y_pred,c="r")
plt.scatter(x_test, y_test)
plt.xlabel("YearsExperience")
plt.ylabel("Salary")
plt.show()


import math
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
mse=mean_squared_error(y_test,y_pred)
print("MSE:",mse)
rmse=math.sqrt(mse)
print("RMSE:",rmse)
r2=r2_score(y_test,y_pred)
print("R2 score:",r2)

#Manual method
#Using Formulas

def slope(x,y):
    x_mean=x.mean()
    y_mean=y.mean()
    
    x_mean_dif=[]
    y_mean_dif=[]
    
    for i in x:
        x_mean_dif.append(i-x_mean)
    for j in y:
        y_mean_dif.append(j-y_mean)
    
    x_mean_dif=np.array(x_mean_dif)
    y_mean_dif=np.array(y_mean_dif)
    
    x_y_mul=sum(x_mean_dif*y_mean_dif)
    
    x_square=sum(map(lambda x:x**2,x_mean_dif))
    
    m=x_y_mul/x_square
    
    return m


def constant(m,x_mean,y_mean):
    b=y_mean-(m*x_mean)
    return b

def predict(m,b,test):
    pred=[]
    for i in test:
        pred.append((m*i)+b)
     
    return np.array(pred) 
    
m=slope(x_train,y_train)
b=constant(m,x_train.mean(),y_train.mean())
pred=predict(m,b,x_test)

print(m,b)

    





import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv("Real estate.csv")
print(data.columns)
sns.pairplot(data)

print(data.isna().sum())

x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sd=StandardScaler()
x_train=sd.fit_transform(x_train)
x_test=sd.transform(x_test)

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)

y_pred=lr.predict(x_test)

import math
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
mse=mean_squared_error(y_test,y_pred)
print("MSE:",mse)
rmse=math.sqrt(mse)
print("RMSE:",rmse)
r2=r2_score(y_test,y_pred)
print("R2 score:",r2)








