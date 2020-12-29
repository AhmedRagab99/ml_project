import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.tree import DecisionTreeClassifier

from sklearn.utils import shuffle
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVR, SVC
from matplotlib import pyplot






def Feature_Selection():
    selector = SelectKBest(score_func=f_classif, k='all')
    fitted = selector.fit(X, Y)
    features_scores = pd.DataFrame(fitted.scores_)
    features_columns = pd.DataFrame(X.columns)
    best_features = pd.concat([features_columns, features_scores], axis=1)
    best_features.columns = ['Feature', 'Score']
    best_features.sort_values(by='Score', ascending=False, inplace=True)
    new_best_features = []
    feature = []
    score = []
    for i in range(5):
        new_best_features.append(best_features['Feature'][i])
        feature.append(best_features['Feature'][i])
        score.append(best_features['Score'][i])
    return new_best_features



##read data
df = pd.read_csv('spotify_training_classification.csv')

# data preproccesing  exept the artist convertion with one_ hot_ encoder
features = ['year', 'acousticness', 'energy', 'loudness', 'instrumentalness', 'duration_ms',
            'explicit', 'tempo', 'valence', 'mode', 'liveness', 'speechiness']

# replace nan values with the mean of the column
df['year'] = df['year'].replace(np.nan, np.mean(df['year']))
df['acousticness'] = df['acousticness'].replace(np.nan, np.mean(df['acousticness']))
df['energy'] = df['energy'].replace(np.nan, np.mean(df['energy']))
df['loudness'] = df['loudness'].replace(np.nan, np.mean(df['loudness']))
df['instrumentalness'] = df['instrumentalness'].replace(np.nan, np.mean(df['instrumentalness']))
df['duration_ms'] = df['duration_ms'].replace(np.nan, np.mean(df['duration_ms']))
df['explicit'] = df['explicit'].replace(np.nan, np.mean(df['explicit']))
df['mode'] = df['mode'].replace(np.nan, np.mean(df['mode']))
df['tempo'] = df['tempo'].replace(np.nan, np.mean(df['tempo']))
df['valence'] = df['valence'].replace(np.nan, np.mean(df['valence']))
df['speechiness'] = df['speechiness'].replace(np.nan, np.mean(df['speechiness']))
df['liveness'] = df['liveness'].replace(np.nan, np.mean(df['liveness']))

training = df.sample(frac=0.8, random_state=420)
##Assign features's value

popularity = {'Intermediate': 0.5, 'High': 1,"Low":0}
df.popularity_level = [popularity[item] for item in df.popularity_level]

# replace strings with numbers



X = training[features]
Y = training['popularity_level']

new_best_features = Feature_Selection()
X_train, X_test, y_train, y_test = train_test_split(training[new_best_features], Y, test_size=0.2, random_state=420)

error = []


def KNNClasifier():
    knn = KNeighborsClassifier(n_neighbors=100)
    knn.fit(X_train, y_train)
    print('KNN Accuracy:')
    print(knn.score(X_test, y_test))
    print('Error:')
    for i in range(1, 40):
        pred_i = knn.predict(X_test)
        error.append(np.mean(pred_i != y_test))



# Calculating error for K values between 1 and 40



def DecistionTreeClassifier():
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), algorithm="SAMME", n_estimators=300)
    model = bdt.fit(X_train, y_train)
    predict = bdt.predict(X_test)
    print(' DecistionTree Accuracy:')
    print(model.score(X_test, y_test))
    print('Error:')
    # print(mean_squared_error(y_test, predict))




def PolyinomialKernalSVM():
    svclassifier = SVC(kernel='poly',degree=8)
    model1 = svclassifier.fit(X_train, y_train)
    predict = model1.predict(X_test)
    print('Accuracy for Polonomial :')
    print(model1.score(X_test, y_test))
    print('Error:')
    # print(mean_squared_error(y_test, predict))


def GuasianKernalSVM():
    svclassifier = SVC(kernel="rbf")
    model1 = svclassifier.fit(X_train, y_train)
    predict = model1.predict(X_test)
    print('Accuracy for Guasioan :')
    print(model1.score(X_test, y_test))
    print('Error:')

DecistionTreeClassifier()
GuasianKernalSVM()
PolyinomialKernalSVM()
KNNClasifier()