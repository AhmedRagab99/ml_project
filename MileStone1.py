import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
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


def Correlation():
    correlations = DF.corr()
    plt.figure(figsize=(10, 6))
    heatmap = sns.heatmap(correlations, vmin=-1, vmax=1, annot=True, cmap='viridis')
    heatmap.set_title('Correlation Heatmap', fontdict={'fontsize': 12}, pad=12)
    plt.show()


def Save_trained_model(pkl_filename, model):
    model_pkl = open(pkl_filename, 'wb')
    pickle.dump(model, model_pkl)
    model_pkl.close()


##read data
DF = pd.read_csv('spotify_training.csv')

## generate pkl files
Linear_pkl_filename = 'Linear.pkl'
Polynomial_pkl_filename = 'Polynomial.pkl'

## correlation heatmap
Correlation()
DF = Update_DataSet(DF)
DF = Encoding(DF)

features = ['year', 'acousticness', 'energy', 'loudness', 'instrumentalness', 'duration_ms',
            'explicit', 'tempo', 'valence', 'mode', 'liveness', 'speechiness']

##Assign features's value

X = DF[features]
Y = DF['popularity']

new_best_features = Feature_Selection()
X = DF[new_best_features]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=420)

X_test.to_csv("X_Test_Regression.csv", index=False)
y_test.to_csv("Y_Test_Regression.csv", index=False)


def Linear_Regression():
    regressor = LinearRegression()
    model = regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    Save_trained_model(Linear_pkl_filename, model)
    print('linear regression Accuracy:')
    print(r2_score(y_test, y_pred))
    print('linear regression Error:')
    print(mean_squared_error(y_test, y_pred))

    # fig, axes = plt.subplots(1, len(X_test.columns), sharey=True, constrained_layout=True, figsize=(30, 15))
    #
    # mn = min(len(X_test), len(y_pred))
    #
    # for i, e in enumerate(X_test.columns):
    #     newX_Train = X_test[e][: mn]
    #     regressor.fit(newX_Train[:, np.newaxis], y_pred)
    #     axes[i].set_title("linear regression")
    #     axes[i].set_xlabel(str(e))
    #     axes[i].set_ylabel('Prediction')
    #     axes[i].scatter(newX_Train[:, np.newaxis], y_pred, color='b')
    #     axes[i].plot(newX_Train[:, np.newaxis],
    #                  regressor.predict(newX_Train[:, np.newaxis]), color='k')
    #
    # plt.show()


def Polynomial_Regression():
    poly_reg = PolynomialFeatures(degree=4)
    X_poly = poly_reg.fit_transform(X_train)
    pol_reg = LinearRegression()
    pol_reg.fit(X_poly, y_train)
    y_pred = pol_reg.predict(poly_reg.fit_transform(X_test))
    Save_trained_model(Polynomial_pkl_filename, pol_reg)
    mse = np.sqrt(mean_squared_error(y_test, y_pred))
    print('Polynomial_Regression Error:')
    print(mse)
    r2 = r2_score(y_test, y_pred)
    print('Polynomial_Regression Accuracy:')
    print(r2)

    # plt.figure(figsize=(10, 6))
    # plt.scatter(X, Y, s=15)
    # plt.plot(X_test, y_pred, color='g', label='Polynomial Regression')
    # plt.xlabel('Predictor', fontsize=16)
    # plt.ylabel('Target', fontsize=16)
    # plt.legend()
    # plt.show()


Linear_Regression()
Polynomial_Regression()
