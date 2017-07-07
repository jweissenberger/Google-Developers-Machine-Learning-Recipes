# This program takes the same code from recipe #4 but replaces the classifer with a self-built classifier called
# simple KNN which is a simplified version of the K-nearest-neighbor classifier

from scipy.spatial import distance

# finds the euclidean distance between two data points
def eucdis(a, b):
    return distance.euclidean(a, b)

class simpleKNN():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    # classifies the data according to that nearest value
    def predict(self, X_test):
        predictions = []
        for row in X_test:
            label = self.closest(row) #hardcoded k to 1, so only looking for the singular closest data point
            predictions.append(label)
        return predictions

    # finds the closest data point to the test value and then
    def closest(self, row):
        bestDist = eucdis(row, self.X_train[0])
        bestIndex = 0
        for i in range(1, len(self.X_train)):
            dist = eucdis(row, self.X_train[i])
            if dist < bestDist:
                bestDist = dist
                bestIndex = i
        return self.y_train[bestIndex]

from sklearn import datasets

iris = datasets.load_iris()

X = iris.data  # features
Y = iris.target  # labels

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.5) #half the data to be used for training

# calls the new classifier
my_classifier = simpleKNN()

# trains the calssifier
my_classifier.fit(X_train, Y_train)

# predicts the type of flower of the test half of the data set
predictions = my_classifier.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy score from Decision Tree: ", accuracy_score(Y_test, predictions)) # the accuracy of the new classifier
