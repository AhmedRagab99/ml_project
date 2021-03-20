import pickle
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np


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


def Update_DataSet(dataframe):
    # replace nan values with the mean of the column
    dataframe['year'] = dataframe['year'].replace(np.nan, np.mean(dataframe['year']))
    dataframe['acousticness'] = dataframe['acousticness'].replace(np.nan, np.mean(dataframe['acousticness']))
    dataframe['energy'] = dataframe['energy'].replace(np.nan, np.mean(dataframe['energy']))
    dataframe['loudness'] = dataframe['loudness'].replace(np.nan, np.mean(dataframe['loudness']))
    dataframe['instrumentalness'] = dataframe['instrumentalness'].replace(np.nan,
                                                                          np.mean(dataframe['instrumentalness']))
    dataframe['duration_ms'] = dataframe['duration_ms'].replace(np.nan, np.mean(dataframe['duration_ms']))
    dataframe['explicit'] = dataframe['explicit'].replace(np.nan, np.mean(dataframe['explicit']))
    dataframe['mode'] = dataframe['mode'].replace(np.nan, np.mean(dataframe['mode']))
    dataframe['tempo'] = dataframe['tempo'].replace(np.nan, np.mean(dataframe['tempo']))
    dataframe['valence'] = dataframe['valence'].replace(np.nan, np.mean(dataframe['valence']))
    dataframe['speechiness'] = dataframe['speechiness'].replace(np.nan, np.mean(dataframe['speechiness']))
    dataframe['liveness'] = dataframe['liveness'].replace(np.nan, np.mean(dataframe['liveness']))
    dataframe['key'] = dataframe['key'].replace(np.nan, np.mean(dataframe['key']))
    # replace strings with numbers
    popularity = {'Intermediate': 1, 'High': 2, 'Low': 0}
    dataframe['popularity_level'] = [popularity[item] for item in dataframe['popularity_level']]
    return dataframe


def Encoding(df):
    le = LabelEncoder()
    for col in df.columns.values:
        if df[col].dtypes == 'object':
            df[col] = le.fit_transform(df[col].astype(str))
    integer_encoded_reshape = np.array(df['key']).reshape(len(df['key']), 1)
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded_reshape)
    df['key'] = onehot_encoded
    return df


def LoadTrainedModel(pkl_filename):
    loadedModel = pickle.load(open(pkl_filename, 'rb'))
    return loadedModel


DF = pd.read_csv('spotify_training_classification.csv')
DF = Update_DataSet(DF)
DF = Encoding(DF)
features = ['year', 'acousticness', 'energy', 'loudness', 'instrumentalness', 'duration_ms',
            'explicit', 'tempo', 'valence', 'mode', 'liveness', 'speechiness']
X = DF[features]
Y = DF['popularity_level']
new_best_features = Feature_Selection()
X = DF[new_best_features]

decision_tree_pkl_filename = 'decision_tree_classifier.pkl'
KNNClassifier_pkl_filename = 'KNNClassifier_pkl_filename.pkl'
PolynomialKernalSVM_pkl_filename = 'PolynomialKernalSVM.pkl'
GausianKernalSVM_pkl_filename = 'GausianKernalSVM_classifier.pkl'
AdaBoostClassifier_pkl_filename = 'AdaBoostClassifier_pkl_filename.pkl'

DecisionTree_Model = LoadTrainedModel(decision_tree_pkl_filename)
Polynomial_Model = LoadTrainedModel(PolynomialKernalSVM_pkl_filename)
AdaBoost_Model = LoadTrainedModel(AdaBoostClassifier_pkl_filename)
KNN_Model = LoadTrainedModel(KNNClassifier_pkl_filename)

# X_test = pd.read_csv('X_Test_Classification.csv')
# Y_test = pd.read_csv('Y_Test_Classification.csv')

print("DecisionTree_Model:")
print(DecisionTree_Model.predict(X))
print(DecisionTree_Model.score(X, Y))

print("KNN_Model")
print(KNN_Model.predict(X))
print(KNN_Model.score(X, Y))

print("AdaBoost_Model")
print(AdaBoost_Model.predict(X))
print(AdaBoost_Model.score(X, Y))
