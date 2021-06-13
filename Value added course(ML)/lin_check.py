import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import seaborn as sns
import pandas as pd



df = pd.read_csv('F:/my_ml/housing.data', delim_whitespace = True, header = None)


col_name=['CRIM','ZIN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PIRATIO','B','LSTAT','MEDV']

df.columns = col_name

col_study=['CRIM','ZIN','INDUS','CHAS','MEDV']
#sns.pairplot(df[col_study],size=1.5)
#plt.show()

X = df['CRIM'].values
X= X.reshape(-1,1)

y=df['MEDV'].values

model = LinearRegression()

model.fit(X,y)

print(model.coef_ , model.intercept_)

sns.regplot(X,y);

plt.xlabel("CRIM")

plt.ylabel("MEDV")

plt.show()






