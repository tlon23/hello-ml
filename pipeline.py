#import a dataset
from sklearn import datasets
iris = datasets.load_iris()

#think of a classifier like a function f(x) = y 
#the features are x
#the labels are y
x = iris.data
y = iris.target 

#half the data will be used to train and half will be used to test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.5)

# The results & accuracy for this classifier will be the same as the tree 
from sklearn.neighbors import KNeighborsClassifier
my_classifier = KNeighborsClassifier()

my_classifier.fit(x_train, y_train)

predictions = my_classifier.predict(x_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))

