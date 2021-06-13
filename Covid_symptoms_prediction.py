import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("whitegrid")


df=pd.read_csv("covid_symptoms.csv")

print(df)

print(df.info)

print(df.shape) #View the no.of columns and rows

print(df.head()) #It will print first 5 datas

print(df.tail()) #It will print Last 5 datas


print(df.columns) #print the column names

print(df.describe()) #Print the statistical value like minimum value,Mean,standard deviation

print(df.isnull().sum()) #Show all the null values

x=df.iloc[:,:5].values.reshape(-1,1)
y=df["Severity_Mild","Severity_Moderate","Severity_Severe"].values

#get correlations of each feature in dataset
corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")

from sklearn.preprocessing import StandardScaler
sd=StandardScaler()
x=sd.fit_transform(x)


from sklearn.linear_model import LinearRegression
linear=LinearRegression()
linear.fit(x,y)

from sklearn.tree import DecisionTreeRegressor
tree=DecisionTreeRegressor(anacon)
tree.fit(x,y)


from sklearn.neighbors import KNeighborsRegressor
knn=KNeighborsRegressor(n_neighbors=8)
knn.fit()

from sklearn.ensemble import RandomForestRegressor
forest=RandomForestRegressor(n_estimators=15)

from sklearn.svm import SVR
support=SVR(kernel="rbf",gamma=0.001,C=15)
from sklearn.model_selection import KFold,cross_val_score

models=[linear,tree,knn,forest,support]
kfold=KFold(n_splits=10)

for model in models:
    accuracy=cross_val_score(model,x,y,cv=kfold)
    print(model,accuracy.mean())