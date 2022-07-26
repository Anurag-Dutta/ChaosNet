import matplotlib.pyplot as plot
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error

diabetes = datasets.load_diabetes()

print(diabetes.keys()) #Print the unique keys

diabetes_X = diabetes.data[:, np.newaxis, 3] #diabetes_X = diabetes.data can be used if we want to make use of the complete dataset rather than some columns. But in that case, we won't be able to plot the scatter plot.
print(diabetes_X) #Print the content of the dataset under column 3

diabetes_X_train = diabetes_X[:-80]
diabetes_X_test = diabetes_X[-20:]

diabetes_Y_train = diabetes.target[:-80]
diabetes_Y_test = diabetes.target[-20:]

model = linear_model.LinearRegression()
model.fit(diabetes_X_train, diabetes_Y_train)
diabetes_Y_predicted = model.predict(diabetes_X_test)

print("Mean squared error is: ", mean_squared_error(diabetes_Y_test, diabetes_Y_predicted))
print("Mean absolute error is: ", mean_absolute_error(diabetes_Y_test, diabetes_Y_predicted))
print("Weights: ", model.coef_)
print("Intercept: ", model.intercept_)

plot.scatter(diabetes_X_test, diabetes_Y_test)
plot.plot(diabetes_X_test, diabetes_Y_predicted)
plot.show()