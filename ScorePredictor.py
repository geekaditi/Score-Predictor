#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 18:59:40 2020

@author: aditisaxena
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

X_train=pd.read_csv('Linear_X_Train.csv')
y_train=pd.read_csv('Linear_Y_Train.csv')
X_test=pd.read_csv('Linear_X_Test.csv')

from sklearn.linear_model import LinearRegression
Regressor=LinearRegression()
Regressor.fit(X_train,y_train)

y_pred=Regressor.predict(X_test)

plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,Regressor.predict(X_train),color='blue')
plt.title('Score vs Time(Test)')
plt.xlabel('Time')
plt.ylabel('Score')
plt.show()

plt.plot(X_test,y_pred,color='blue')
plt.title('Score vs Time (Test predictions)')
plt.xlabel('Time')
plt.ylabel('Score')
plt.show()
