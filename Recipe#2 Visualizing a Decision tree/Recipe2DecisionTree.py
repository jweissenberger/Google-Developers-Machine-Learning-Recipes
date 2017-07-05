#learning algorithm that determines if a given plant is a certian type of iris depending on features like sepal length
# and width and petal length and width

#0 represents setosa, 1 is versicolor and 2 is virginica
import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()

test_idx = [0,50,100] # index of examples that will be removed for testing data

print(iris.feature_names)  #prints out the different kinds of features that are being used
print(iris.target_names) #prints out the names of the 3 kids of irises we are trying to classify

#training data
#this is the majority of the data because we want the classifier to see as many examples as possible
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

#testing data
# an example of each kind of flower was removed to see if the algorithm could predict each different kind
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

#decision tree classifier
clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

print(test_target) # this is what the algorithm should predict for the test data
print(clf.predict(test_data)) # if these match the above values, then the classifier has correctly predicted the type
# of flower

#visualize the decision tree in pdf
from sklearn.externals.six import StringIO
import pydot
dot_data = StringIO()
tree.export_graphviz(clf,
                     out_file=dot_data,
                     feature_names=iris.feature_names,
                     class_names=iris.target_names,
                     filled=True, rounded=True,
                     impurity=False)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris.pdf")
