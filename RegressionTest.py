import pickle
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures


def LoadTrainedModel(pkl_filename):
    loadedModel = pickle.load(open(pkl_filename, 'rb'))
    return loadedModel


Linear_pkl_filename = 'Linear.pkl'
Polynomial_pkl_filename = 'Polynomial.pkl'

Linear_Model = LoadTrainedModel(Linear_pkl_filename)
Polynomial_Model = LoadTrainedModel(Polynomial_pkl_filename)


X_test = pd.read_csv('X_Test_Regression.csv')
Y_test = pd.read_csv('Y_Test_Regression.csv')

print("LinearRegression Model")
print(Linear_Model.predict(X_test))
print(Linear_Model.score(X_test, Y_test))

print("Polynomial Regression")
poly_reg = PolynomialFeatures(degree=4)
print(Polynomial_Model.predict(poly_reg.fit_transform(X_test)))
Y_predict = Polynomial_Model.predict(poly_reg.fit_transform(X_test))
print(r2_score(Y_test, Y_predict))
