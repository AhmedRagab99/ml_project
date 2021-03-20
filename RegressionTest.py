import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
import seaborn as sns


def Feature_Selection():
    selector = SelectKBest(score_func=f_regression, k='all')
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

    plt.figure(figsize=(10, 8))
    sns.barplot(feature, score)
    plt.ylabel('feature')
    plt.xlabel('score')
    plt.show()
    return new_best_features


def Update_DataSet(df):
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
    return df


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


DF = pd.read_csv('spotify_training.csv')
DF = Update_DataSet(DF)
DF = Encoding(DF)
features = ['year', 'acousticness', 'energy', 'loudness', 'instrumentalness', 'duration_ms',
            'explicit', 'tempo', 'valence', 'mode', 'liveness', 'speechiness']

X = DF[features]
Y = DF['popularity']

new_best_features = Feature_Selection()
X = DF[new_best_features]

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
