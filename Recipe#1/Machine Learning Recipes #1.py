# This is a simple learning algorithm that predicts if an input item is an apple or an orange based on its weight and if
# it is considered bumpy or smooth

from sklearn import tree

# training set
features = [[140, 1], [130, 1], [150, 0], [170, 1]]
# the first feature is the weight of the fruit and in the second 1 represents smooth and 0 represents bumpy

labels = [0, 0, 1, 1]
# 0 represents apples and 1 represents

clf = tree.DecisionTreeClassifier()  #creates a decision tree classifier based on the training set

clf = clf.fit(features, labels) #the training algorithm

print(clf.predict([[160, 0]])) #want to predict if an unknow fruit that weighs 160 and is bumpy
# this program returns a 1, meaning that it predicts the output