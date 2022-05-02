import matplotlib.pyplot as plot
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error

data_X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
print(data_X) #Print the content of the dataset

model_X_train = data_X
model_X_test = data_X

model_Y_train = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
model_Y_test = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])

model = linear_model.LinearRegression()
model.fit(model_X_train, model_Y_train)
model_Y_predicted = model.predict(model_X_test)

print("Mean squared error is: ", mean_squared_error(model_Y_test, model_Y_predicted))
print("Mean absolute error is: ", mean_absolute_error(model_Y_test, model_Y_predicted))
print("Weights: ", model.coef_)
print("Intercept: ", model.intercept_)

plot.scatter(model_X_test, model_Y_test)
plot.plot(model_X_test, model_Y_predicted)
plot.show()