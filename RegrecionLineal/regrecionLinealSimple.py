
from turtle import color
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importar el dataset
dataset = pd.read_csv(
    'C:/Users/pcc/Desktop/Escritorio/Inteliencia Artificial/MachineLearning-az-master/cursoML/RegrecionLineal/datasets/Salary_Data.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,-1].values


# divir el dataset en un conjunto de entrenamiento y un conjunto de testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

# crear modelo de regrecion lineal simple con el conjunto de entrenamiento
regression = LinearRegression()
regression.fit(X_train, y_train)

#Predecir e conjunto de test
y_pred = regression.predict(X_test)

#Visualizar los resultados de la prediccion
# 

plt.scatter(X_train, y_train, color='red') 
plt.plot(X_train, regression.predict(X_train), color='blue')
plt.title('Salary vs Experience (Taining set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
#Visualizar los resultados de la prediccion
# 
plt.scatter(X_test, y_test, color='red') 
plt.plot(X_test, regression.predict(X_test), color='blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()