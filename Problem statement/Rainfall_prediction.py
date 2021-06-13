from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

data=pd.read_csv("rainfall in india 1901-2015.csv")

print(data)

print(data.info)

# Annual rainfall in subdivisions of India
plt.figure(figsize=(20,18))
ax = sns.boxplot(x="SUBDIVISION", y="ANNUAL", data=data, width=1, linewidth=2)
ax.set_xlabel('Subdivision',fontsize=30)
ax.set_ylabel('Annual Rainfall (in mm)',fontsize=30)
plt.title('Annual Rainfall in Subdivisions of India',fontsize=40)
ax.tick_params(axis='x', labelsize=20, rotation=90)
ax.tick_params(axis='y', labelsize=20, rotation=0)

# Average monthly rainfall in India
ax=data[['JAN', 'FEB', 'MAR', 'APR','MAY', 'JUN', 'AUG', 'SEP', 'OCT','NOV','DEC']].mean().plot.bar(width=0.5, linewidth=2, figsize=(16,10))
plt.xlabel('Month',fontsize=30)
plt.ylabel('Monthly Rainfall (in mm)', fontsize=30)
plt.title('Monthly Rainfall in Subdivisions of India', fontsize=25)
ax.tick_params(labelsize=10)
plt.grid()

# Average monthly rainfall in India
ax = data[['Jan-Feb', 'Mar-May', 'Jun-Sep', 'Oct-Dec']].mean().plot.bar(width=0.5, linewidth=2, figsize=(16,10))
plt.xlabel('Quarter',fontsize=15)
plt.ylabel('Quarterly Rainfall (in mm)', fontsize=15)
plt.title('Quarterly Rainfall in India', fontsize=20)
ax.tick_params(labelsize=15)
plt.grid()

# Visualizing annual rainfall over the years(1901-2015) in India
ax = data.groupby("YEAR").mean()['ANNUAL'].plot(ylim=(1000,2000),color='r',marker='o',linestyle='-',linewidth=2,figsize=(12,10));
plt.xlabel('Year',fontsize=20)
plt.ylabel('Annual Rainfall (in mm)',fontsize=20)
plt.title('Annual Rainfall from Year 1901 to 2015 in India',fontsize=25)
ax.tick_params(labelsize=15)
plt.grid()

#Arunachal Pradesh

# Getting rainfall data for Arunachal Pradesh
Arunachal = data.loc[(data['SUBDIVISION'] == 'ARUNACHAL PRADESH')]
Arunachal.head()

# Average monthly rainfall in Arunachal
ax = Arunachal[['JAN', 'FEB', 'MAR', 'APR','MAY', 'JUN', 'AUG', 'SEP', 'OCT','NOV','DEC']].mean().plot.bar(width=0.5, linewidth=2, figsize=(16,10))
plt.xlabel('Month',fontsize=30)
plt.ylabel('Monthly Rainfall (in mm)', fontsize=30)
plt.title('Monthly Rainfall in Arunachal', fontsize=25)
ax.tick_params(labelsize=10)
plt.grid()

# Average monthly rainfall in Arunachal Pradesh
ax = Arunachal[['Jan-Feb', 'Mar-May', 'Jun-Sep', 'Oct-Dec']].mean().plot.bar(width=0.5, linewidth=2, figsize=(16,10))
plt.xlabel('Quarter',fontsize=15)
plt.ylabel('Quarterly Rainfall (in mm)', fontsize=15)
plt.title('Quarterly Rainfall in Arunachal', fontsize=20)
ax.tick_params(labelsize=15)
plt.grid()

# Visualizing annual rainfall over the years(1901-2015) in Arunachal Pradesh
ax = Arunachal.groupby("YEAR").mean()['ANNUAL'].plot(ylim=(1500,6500),color='r',marker='o',linestyle='-',linewidth=2,figsize=(12,10));
plt.xlabel('Year',fontsize=20)
plt.ylabel('Arunachal Annual Rainfall (in mm)',fontsize=20)
plt.title('Arunachal Annual Rainfall from Year 1901 to 2015',fontsize=25)
ax.tick_params(labelsize=15)
plt.grid()

print('Average annual rainfall received by Arunachal Pradesh = ',int(Arunachal['ANNUAL'].mean()),'mm')
a = Arunachal[Arunachal['YEAR'] == 1948]
print(a)

#Let's find out which states and years have recorded the maximum and minimum rainfall in India.
# Subdivisions receiving maximum and minimum rainfall
print(data.groupby('SUBDIVISION').mean()['ANNUAL'].sort_values(ascending=False).head(10))
print('\n')
print("--------------------------------------------")
print(data.groupby('SUBDIVISION').mean()['ANNUAL'].sort_values(ascending=False).tail(10))

#As you can see that the subdivsions receiving maximum rainfall belongs to the southern and eastern parts of India whereas those receiving minimum rainfall belongs to the northern parts of India.

# Years which recorded maximum and minimum rainfall
print(data.groupby('YEAR').mean()['ANNUAL'].sort_values(ascending=False).head(10))
print('\n')
print("--------------------------------------------")
print(data.groupby('YEAR').mean()['ANNUAL'].sort_values(ascending=False).tail(10))

