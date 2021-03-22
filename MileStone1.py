import pickle
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, PolynomialFeatures, StandardScaler
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
    best_features.sort_values(by='Score', ascending=False, ignore_index=True, inplace=True)
    new_best_features = []

    for i in range(5):
        new_best_features.append(best_features['Feature'][i])

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
DF = Update_DataSet(DF)
DF = Encoding(DF)
Correlation()

features = ['year', 'acousticness', 'energy', 'loudness', 'instrumentalness', 'duration_ms',
            'explicit', 'tempo', 'valence', 'mode', 'liveness', 'speechiness']

##Assign features's value

X = DF[features]
Y = DF['popularity']

new_best_features = Feature_Selection()
X = DF[new_best_features]
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=420)

X_test.to_csv("X_Test_Regression.csv", index=False)
y_test.to_csv("Y_Test_Regression.csv", index=False)


def Linear_Regression():
    regressor = LinearRegression()
    training_start = time.time()
    model = regressor.fit(X_train, y_train)
    training_end = time.time()
    trainig_time = training_end - training_start
    y_pred = regressor.predict(X_test)
    Save_trained_model(Linear_pkl_filename, model)
    print('linear regression Accuracy:')
    print(r2_score(y_test, y_pred))
    print('linear regression Error:')
    print(mean_squared_error(y_test, y_pred))
    print("training time")
    print(trainig_time)
    fig, axes = plt.subplots(1, len(X_train.columns.values), sharey=True, constrained_layout=True, figsize=(30, 15))

    for i, e in enumerate(X_train.columns):
        regressor.fit(X_train[e].values[:, np.newaxis], y_train.values)
        axes[i].set_title("Best fit line")
        axes[i].set_xlabel(str(e))
        axes[i].set_ylabel('SalePrice')
        axes[i].scatter(X_train[e].values[:, np.newaxis], y_train, color='b')
        axes[i].plot(X_train[e].values[:, np.newaxis],
                     regressor.predict(X_train[e].values[:, np.newaxis]), color='k')

    plt.show()


def Polynomial_Regression():
    poly_reg = PolynomialFeatures(degree=4)
    training_start = time.time()
    X_poly = poly_reg.fit_transform(X_train)
    pol_reg = LinearRegression()
    pol_reg.fit(X_poly, y_train)
    training_end = time.time()
    trainig_time = training_end - training_start
    y_pred = pol_reg.predict(poly_reg.fit_transform(X_test))
    Save_trained_model(Polynomial_pkl_filename, pol_reg)
    mse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print('Polynomial_Regression Accuracy:')
    print(r2)
    print('Polynomial_Regression Error:')
    print(mse)
    print("training time")
    print(trainig_time)


Linear_Regression()
Polynomial_Regression()
