import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


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


def Update_DataSet():
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
    df['key'] = df['key'].replace(np.nan, np.mean(df['key']))
    # replace strings with numbers
    popularity = {'Intermediate': 1, 'High': 2, 'Low': 0}
    df['popularity_level'] = [popularity[item] for item in df['popularity_level']]


def Encoding():
    le = LabelEncoder()
    for col in df.columns.values:
        if df[col].dtypes == 'object':
            df[col] = le.fit_transform(df[col].astype(str))
    integer_encoded_reshape = np.array(df['key']).reshape(len(df['key']), 1)
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded_reshape)
    df['key'] = onehot_encoded


def Save_trained_model(pkl_filename, model):
    model_pkl = open(pkl_filename, 'wb')
    pickle.dump(model, model_pkl)
    model_pkl.close()


def KNN_Classifier():
    # change  w from uniform to distance
    knn = KNeighborsClassifier(n_neighbors=100, weights="uniform")
    knn.fit(X_train, y_train.astype('int'))
    Prediction = knn.predict(X_test)
    Save_trained_model(KNNClassifier_pkl_filename, knn)
    print('KNN Accuracy:')
    print(knn.score(X_test, y_test))
    print('Error:')
    print(mean_squared_error(y_test, Prediction))


def DecisionTreeClassifieer():
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), algorithm="SAMME", n_estimators=100)
    model = bdt.fit(X_train, y_train.astype('int'))
    predict = bdt.predict(X_test)
    Save_trained_model(decision_tree_pkl_filename, model)
    print('DecisionTree Accuracy:')
    print(model.score(X_test, y_test))
    print('Error:')
    print(mean_squared_error(y_test, predict))


def Polynomial_KernalSVM():
    svclassifier = SVC(kernel='poly', degree=3)
    model1 = svclassifier.fit(X_train, y_train.astype('int'))
    predict = model1.predict(X_test)
    Save_trained_model(PolynomialKernalSVM_pkl_filename, model1)
    print('Accuracy for Polynomial :')
    print(model1.score(X_test, y_test))
    print('Error:')
    print(mean_squared_error(y_test, predict))


def Guasian_KernalSVM():
    svclassifier = SVC(kernel="rbf")
    model1 = svclassifier.fit(X_train, y_train.astype('int'))
    predict = model1.predict(X_test)
    Save_trained_model(GausianKernalSVM_pkl_filename, model1)
    print('Accuracy for Guasian :')
    print(model1.score(X_test, y_test))
    print('Error:')
    print(mean_squared_error(y_test, predict))


##read data
df = pd.read_csv('spotify_training_classification.csv')

## generate pkl files
decision_tree_pkl_filename = 'decision_tree_classifier.pkl'
KNNClassifier_pkl_filename = 'KNNClassifier_pkl_filename.pkl'
PolynomialKernalSVM_pkl_filename = 'PolynomialKernalSVM.pkl'
GausianKernalSVM_pkl_filename = 'GausianKernalSVM_classifier.pkl'

##preprocessing
Update_DataSet()
Encoding()

# data except the key conversion with one_ hot_ encoder and artists conversion with label encoding
features = ['year', 'acousticness', 'energy', 'loudness', 'instrumentalness', 'duration_ms',
            'explicit', 'tempo', 'valence', 'mode', 'liveness', 'speechiness']

##Assign features's value
# training = df.sample(frac=0.8, random_state=420)
X = df[features]
Y = df['popularity_level']
new_best_features = Feature_Selection()
X = df[new_best_features]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=420)


X_test.to_csv("X_Test_Classification.csv", index=False)
y_test.to_csv("Y_Test_Classification.csv", index=False)

##Run Classification Models
KNN_Classifier()
DecisionTreeClassifieer()
Polynomial_KernalSVM()
Guasian_KernalSVM()
