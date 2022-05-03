from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = iris.data [:, np.newaxis, 3] #Takes the 3'rd Column, that is the Petal Length into consideration
Y = (iris.target == 2).astype(np.int) #Tells whether a given attribute is corresponding to Iris Virginica or not, and stores 1 & 0 accordingly
print(X, Y)
#print(iris.DESCR)
clf = LogisticRegression()
clf.fit(X, Y)

predict = clf.predict([[1.6]])
print(predict)