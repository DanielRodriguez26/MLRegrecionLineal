
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, metrics
from sklearn.model_selection import train_test_split

# cargar el dataset de boston
boston = datasets.load_boston(return_X_y=False)

# definr la matriz  X y la respuesta del vector  Y
X = boston.data
y = boston.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=1)

model = linear_model.LinearRegression()
model.fit(X_train, y_train)

# regression coefficients
print('Coefficients: ', model.coef_)

# variance score: 1 means perfect prediction
print('Variance score: {}'.format(model.score(X_test, y_test)))


# setting plot style
plt.style.use('fivethirtyeight')

# plotting residual errors in training data
plt.scatter(model.predict(X_train), model.predict(X_train) - y_train,
            color="green", s=10, label='Train data')

# plotting residual errors in test data
plt.scatter(model.predict(X_test), model.predict(X_test) - y_test,
            color="blue", s=10, label='Test data')

# plotting line for zero residual error
plt.hlines(y=0, xmin=0, xmax=50, linewidth=2)

# plotting legend
plt.legend(loc='upper right')

# plot title
plt.title("Residual errors")

# method call for showing the plot
plt.show()
