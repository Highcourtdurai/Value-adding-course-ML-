from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

sns.set_style("whitegrid")

df=pd.read_csv("rainfall in india 1901-2015.csv")

data_cleaned=df.dropna(how='any', inplace=True)#Drop if any null value placed


print(df)

print(df.info)

x=df.loc[:,"YEAR"].values.reshape(-1,1)
y=df.loc[:,"heavy"].values.reshape(-1,1)

# data_set= data_cleaned[data_cleaned.SUBDIVISION==x_names[0]] 
# req_data= data_set[['ANNUAL','YEAR']]
# x=req_data['YEAR'].values.reshape(-1,1)
# y=req_data['ANNUAL'].values.reshape(-1,1)

#Prediction  
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)


from sklearn.linear_model import LinearRegression
linear = LinearRegression()
linear.fit(x_train,y_train)
linear.predict(x_test)


# y_pred =model.predict(x_test)
# pred = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
# print(pred)

# pred.plot(kind='bar',figsize=(16,10))
# plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
# plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
# plt.show()

from sklearn.tree import DecisionTreeRegressor
tree=DecisionTreeRegressor()
tree.fit(x_train,y_train)
# print(tree)
y_pred=tree.predict(x_test)
# print(pred)


from sklearn.neighbors import KNeighborsRegressor
knn=KNeighborsRegressor(n_neighbors=8)
knn.fit(x_train,y_train)
knn.predict(x_test)

from sklearn.ensemble import RandomForestRegressor
forest=RandomForestRegressor(n_estimators=15)
forest.fit(x_train,y_train)
forest.predict(x_test)

from sklearn.svm import SVR
support=SVR(kernel="rbf",gamma=0.001,C=15)
support.fit(x_train,y_train)
support.predict(x_test)

from sklearn.model_selection import KFold,cross_val_score

models=[linear,tree,knn,forest,support]
kfold=KFold(n_splits=10)

for model in models:
    accuracy=cross_val_score(model,x,y,cv=kfold)
    print(model,accuracy.mean())