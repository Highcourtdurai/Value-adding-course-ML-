import pandas as pd
import numpy as np

df=pd.read_csv("indian_liver_patient.csv")

print(df)

print(df.shape) #View the no.of columns and rows

print(df.head()) #It will print first 5 datas

print(df.tail()) #It will print Last 5 datas

print(df.columns) #print the column names

print(df.describe()) #Print the statistical value like minimum value,Mean,standard deviation

print(df.isnull().sum()) #Show all the null values