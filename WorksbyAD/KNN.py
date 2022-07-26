from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
features = iris.data
labels = iris.target
print(iris.DESCR)
clf = KNeighborsClassifier()
clf.fit(features, labels)

print("Enter the Sepal Length: ", end = "")
sepal_length = float(input())
print("Enter the Sepal Width: ", end = "")
sepal_width = float(input())
print("Enter the Petal Length: ", end = "")
petal_length = float(input())
print("Enter the Petal Width: ", end = "")
petal_width = float(input())
predict = clf.predict([[sepal_length, sepal_width, petal_length, petal_width]])
print(predict)