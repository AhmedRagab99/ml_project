import pickle
import pandas as pd


def LoadTrainedModel(pkl_filename):
    loadedModel = pickle.load(open(pkl_filename, 'rb'))
    return loadedModel


decision_tree_pkl_filename = 'decision_tree_classifier.pkl'
KNNClassifier_pkl_filename = 'KNNClassifier_pkl_filename.pkl'
PolynomialKernalSVM_pkl_filename = 'PolynomialKernalSVM.pkl'
GausianKernalSVM_pkl_filename = 'GausianKernalSVM_classifier.pkl'

DecisionTree_Model = LoadTrainedModel(decision_tree_pkl_filename)
Polynomial_Model = LoadTrainedModel(PolynomialKernalSVM_pkl_filename)
Gaussian_Model = LoadTrainedModel(GausianKernalSVM_pkl_filename)
KNN_Model = LoadTrainedModel(KNNClassifier_pkl_filename)

X_test = pd.read_csv('X_Test_Classification.csv')
Y_test = pd.read_csv('Y_Test_Classification.csv')

print("DecisionTree_Model:")
print(DecisionTree_Model.predict(X_test))
print(DecisionTree_Model.score(X_test, Y_test))
print("Polynomial_Model:")
print(Polynomial_Model.predict(X_test))
print(Polynomial_Model.score(X_test, Y_test))
print("Gaussian_Model")
print(Gaussian_Model.predict(X_test))
print(Gaussian_Model.score(X_test, Y_test))
print("KNN_Model")
print(KNN_Model.predict(X_test))
print(KNN_Model.score(X_test, Y_test))
