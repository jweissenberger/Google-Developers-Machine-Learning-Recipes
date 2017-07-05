from sklearn import datasets

#loads the iris data set from sklearn
iris = datasets.load_iris()

X = iris.data  # features
Y = iris.target  # labels

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.5) #half the data to be used for training

# predictions with Decision tree classifier
from sklearn import tree
my_classifier = tree.DecisionTreeClassifier()

# trains the classifier with half of the data set
my_classifier.fit(X_train, Y_train)

# predicts what type of iris the second half the of the data set is based on the classifier
predictions = my_classifier.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy score from Decision Tree: ", accuracy_score(Y_test, predictions))  # the accuracy of the decision tree


