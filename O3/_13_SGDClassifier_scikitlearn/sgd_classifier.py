from sklearn.linear_model import SGDClassifier

perceptron = SGDClassifier(loss='perceptron')

logistic_regression = SGDClassifier(loss='log')

svm = SGDClassifier(loss='hinge')
