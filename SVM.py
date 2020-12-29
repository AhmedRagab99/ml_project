import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.utils import shuffle
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVR
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

    plt.figure(figsize=(10, 8))
    sns.barplot(feature, score)
    plt.ylabel('feature')
    plt.xlabel('score')
    plt.show()
    return new_best_features


##read data
df = pd.read_csv('spotify_training.csv')

##drop nulls
df.dropna(inplace=True)

## correlation heatmap
correlations = df.corr()
plt.figure(figsize=(10, 6))
heatmap = sns.heatmap(correlations, vmin=-1, vmax=1, annot=True, cmap='viridis')
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize': 12}, pad=12)
plt.show()

##label and one-hot-encoder
le = LabelEncoder()
onehot_encoder = OneHotEncoder(sparse=False)
for col in df.columns.values:
    if df[col].dtypes == 'object':
        df[col] = le.fit_transform(df[col].astype(str))

integer_encoded_reshape = np.array(df['key']).reshape(len(df['key']), 1)
onehot_encoder = OneHotEncoder(sparse=False)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded_reshape)
df['key'] = onehot_encoded

features = ['year', 'acousticness', 'energy', 'loudness', 'instrumentalness', 'duration_ms',
            'explicit', 'tempo', 'valence', 'mode', 'liveness', 'speechiness']

##Assign features's value
training = df.sample(frac=0.6, random_state=420)
X = training[features]
Y = training['popularity']

new_best_features = Feature_Selection()

X_train, X_test, y_train, y_test = train_test_split(training[new_best_features], Y, test_size=0.4, random_state=420)


def Linear_Regression():
    regressor = LinearRegression()
    model = regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    print('Accuracy:')
    print(model.score(X_test, y_test))
    print('Error:')
    print(mean_squared_error(y_test, y_pred))

    fig, axes = plt.subplots(1, len(X_train.columns), sharey=True, constrained_layout=True, figsize=(30, 15))

    mn = min(len(X_train), len(y_pred))

    for i, e in enumerate(X_train.columns):
        newX_Train = X_train[e][: mn]
        regressor.fit(newX_Train[:, np.newaxis], y_pred)
        axes[i].set_title("linear regression")
        axes[i].set_xlabel(str(e))
        axes[i].set_ylabel('Predection')
        axes[i].scatter(newX_Train[:, np.newaxis], y_pred, color='b')
        axes[i].plot(newX_Train[:, np.newaxis],
                     regressor.predict(newX_Train[:, np.newaxis]), color='k')

    plt.show()


Linear_Regression()


def Polynomial_Regression():
    poly_reg = PolynomialFeatures(degree=4)
    X_poly = poly_reg.fit_transform(X_train)
    pol_reg = LinearRegression()
    pol_reg.fit(X_poly, y_train)
    y_pred = pol_reg.predict(X_poly)
    mse = np.sqrt(mean_squared_error(y_train, y_pred))
    r2 = r2_score(y_train, y_pred)
    print('Error:')
    print(mse)
    print('Accuracy:')
    print(r2)


Polynomial_Regression()