#1.Linear Regression

#2.Decision Tree Regressor

#3.Random Forest Regressor
import pandas as pd
from sklearn import datasets

data=pd.read_csv('D:\ml projects\concrete\concrete.csv')
print(data)
data=data.astype(int)
print(data.head())
print(data.columns)
print(data.isnull().sum())
from sklearn.model_selection import train_test_split
X=data[['Cement', 'Blast Furnace Slag', 'Fly Ash', 'Water', 'Superplasticizer', 'Coarse Aggregate', 'Fine Aggregate', 'Age']]
Y=data['Strength']
X_TRAIN,X_TEST,Y_TRAIN,Y_TEST=train_test_split(X,Y,test_size=0.2,random_state=500)
from sklearn.linear_model import LinearRegression
lin=LinearRegression()
lin.fit(X_TRAIN,Y_TRAIN)
Y_PREDICT=lin.predict(X_TEST)
from sklearn.metrics import mean_squared_error
import numpy as np
mse=np.sqrt(mean_squared_error(Y_TEST, Y_PREDICT))
print(mse)
print(lin.score(X_TRAIN, Y_TRAIN))
from sklearn.tree import DecisionTreeRegressor 
reg2 = DecisionTreeRegressor() 
     

#Fitting data into the model.
reg2.fit(X_train, y_train) 