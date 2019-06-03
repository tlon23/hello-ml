from sklearn import tree
import numpy as np

#Machine learning "hello world" program that classifies 
#whether an object is an apple or an orange

# 1 for 2-seats, 0 for >2-seats 
features = [[300,1], [450,1], [200,0], [150,0]]

# 0 sports car, 1 for minivan
labels = [0, 0, 1, 1]

#creating a classifier object that uses a decision tree
clf = tree.DecisionTreeClassifier()

#fit method helps find the patterns in the data 
clf = clf.fit(features, labels)

#predicts what the object is given a set of data about the object
a = clf.predict([[130, 0]])

if a == np.array([0]):
	print("The car is a sports car")
else:
	print("This car is a minivan")