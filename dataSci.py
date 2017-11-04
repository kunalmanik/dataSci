from  sklearn import tree, neighbors, svm, naive_bayes
from sklearn.metrics import accuracy_score

X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39], [177, 70, 40],
[159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'female', 'female', 'female', 'male', 'male', 'male', 'female', 'male', 'female', 'male']

clfTree = tree.DecisionTreeClassifier()
clfKNeighbour = neighbors.KNeighborsClassifier()
clfSVC = svm.SVC()
clfGaussian = naive_bayes.GaussianNB()

clfTree = clfTree.fit(X,Y)
clfKNeighbour = clfKNeighbour.fit(X,Y)
clfSVC = clfSVC.fit(X,Y)
clfGaussian = clfGaussian.fit(X,Y)

X1 = [[190, 70, 43]]

predictionTree = clfTree.predict(X1)
predictionKNeighbour = clfKNeighbour.predict(X1)
predictionSVC = clfSVC.predict(X1)
predictionGaussian = clfGaussian.predict(X1)

Y2 = ['male'] #['female']

accuracyTree = accuracy_score(Y2, predictionTree)
accuracyKNeighbour = accuracy_score(Y2, predictionKNeighbour)
accuracySVC = accuracy_score(Y2, predictionSVC)
accuracyGaussian = accuracy_score(Y2, predictionGaussian)

print(predictionTree, accuracyTree)
print(predictionKNeighbour, accuracyKNeighbour)
print(predictionSVC, accuracySVC)
print(predictionGaussian, accuracyGaussian)
