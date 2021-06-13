#reason for going to robust regression- RANSAC
#http://digitalfirst.bfwpub.com/stats_applet/stats_applet_5_correg.html

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor

#Read raw csv file without headers and comma delimiter
df=pd.read_csv('housing.data',delim_whitespace=True,header=None)

#give names for each column
col_name=['CRIM','ZIN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PIRATIO','B','LSTAT','MEDV']
df.columns=col_name

#Select the input and output datas
X=df['RM'].values.reshape(-1,1)
y=df['MEDV'].values

#fitting the data with RANdom SAmple Consensus (RANSAC) Algorithm
ransac=RANSACRegressor()
ransac.fit(X,y)

#saperating inler and outlier points
inlier_mask=ransac.inlier_mask_
outlier_mask=np.logical_not(inlier_mask)

#predicting the values with (3,4,5,6,7,8,9)
line_X=np.arange(3,10,1)
line_y_ransac=ransac.predict(line_X.reshape(-1,1))

#plot the figure
sns.set(style='darkgrid',context='notebook')
plt.figure(figsize=(12,10));
plt.scatter(X[inlier_mask],y[inlier_mask],c='blue',marker='o',label='Inliers')
plt.scatter(X[outlier_mask],y[outlier_mask],c='brown',marker='s',label='Outliers')
plt.plot(line_X,line_y_ransac,color='red')
plt.xlabel('RM')
plt.ylabel('MEDV')
plt.legend(loc='upper left')
plt.show()

#check and compare the coef and intercept with linear reg
print(ransac.estimator_.coef_,ransac.estimator_.intercept_)
