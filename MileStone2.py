import time
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import seaborn as sns
from sklearn.preprocessing import StandardScaler


def Feature_Selection():
    selector = SelectKBest(score_func=f_classif, k='all')
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


def KNN_Classifier():
    # change  w from uniform to distance
    knn = KNeighborsClassifier(n_neighbors=100, weights="uniform")
    training_start = time.time()
    knn.fit(X_train, y_train.astype('int'))
    training_end = time.time()
    trainig_time = training_end - training_start
    test_start = time.time()
    Prediction = knn.predict(X_test)
    test_end = time.time()
    test_time = test_end - test_start
    Save_trained_model(KNNClassifier_pkl_filename, knn)
    print('KNN Accuracy:')
    Accuracy = sklearn.metrics.accuracy_score(y_test, Prediction)
    print(Accuracy)
    print('Error:')
    print(mean_squared_error(y_test, Prediction))
    return [Accuracy, trainig_time, test_time]


def DecisionTreeClassifieer():
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), algorithm="SAMME", n_estimators=100)
    training_start = time.time()
    model = bdt.fit(X_train, y_train.astype('int'))
    training_end = time.time()
    trainig_time = training_end - training_start
    test_start = time.time()
    Prediction = bdt.predict(X_test)
    test_end = time.time()
    test_time = test_end - test_start
    Save_trained_model(decision_tree_pkl_filename, model)
    print('DecisionTree Accuracy:')
    Accuracy = sklearn.metrics.accuracy_score(y_test, Prediction)
    print(Accuracy)
    print('Error:')
    print(mean_squared_error(y_test, Prediction))

    return [Accuracy, trainig_time, test_time]


def AdaBoost_Classifier():
    abc = AdaBoostClassifier(n_estimators=50,
                             learning_rate=1)
    training_start = time.time()
    model = abc.fit(X_train, y_train.astype('int'))
    training_end = time.time()
    trainig_time = training_end - training_start
    test_start = time.time()
    Prediction = model.predict(X_test)
    test_end = time.time()
    test_time = test_end - test_start
    Save_trained_model(AdaBoostClassifier_pkl_filename, abc)
    print('AdaBoost Accuracy:')
    Accuracy = sklearn.metrics.accuracy_score(y_test, Prediction)
    print(Accuracy)
    print('Error:')
    print(mean_squared_error(y_test, Prediction))
    return [Accuracy, trainig_time, test_time]


def Polynomial_KernalSVM():
    svclassifier = SVC(kernel='poly', degree=2)
    training_start = time.time()
    model = svclassifier.fit(X_train, y_train.astype('int'))
    training_end = time.time()
    trainig_time = training_end - training_start
    test_start = time.time()
    Prediction = model.predict(X_test)
    test_end = time.time()
    test_time = test_end - test_start
    Save_trained_model(PolynomialKernalSVM_pkl_filename, model)
    print('Accuracy for Polynomial :')
    Accuracy = sklearn.metrics.accuracy_score(y_test, Prediction)
    print(Accuracy)
    print('Error:')
    print(mean_squared_error(y_test, Prediction))
    return [Accuracy, trainig_time, test_time]


def Guasian_KernalSVM():
    svclassifier = SVC(kernel="rbf")
    model1 = svclassifier.fit(X_train, y_train.astype('int'))
    Prediction = model1.predict(X_test)
    Save_trained_model(GausianKernalSVM_pkl_filename, model1)
    print('Accuracy for Guasian :')
    Accuracy = sklearn.metrics.accuracy_score(y_test, Prediction)
    print(Accuracy)
    print('Error:')
    print(mean_squared_error(y_test, Prediction))


def drawBarGraph(XName, YName, XValue, YValue, Title):
    plt.bar(XValue, YValue)
    plt.title(Title)
    plt.xlabel(XName)
    plt.ylabel(YName)
    plt.show()


##read data
DF = pd.read_csv('spotify_training_classification.csv')


## generate pkl files
decision_tree_pkl_filename = 'decision_tree_classifier.pkl'
KNNClassifier_pkl_filename = 'KNNClassifier_pkl_filename.pkl'
AdaBoostClassifier_pkl_filename = 'AdaBoostClassifier_pkl_filename.pkl'
PolynomialKernalSVM_pkl_filename = 'PolynomialKernalSVM.pkl'
GausianKernalSVM_pkl_filename = 'GausianKernalSVM_classifier.pkl'

##preprocessing
DF = Update_DataSet(DF)
DF = Encoding(DF)
Correlation()

# data except the key conversion with one_ hot_ encoder and artists conversion with label encoding
features = ['year', 'acousticness', 'energy', 'loudness', 'instrumentalness', 'duration_ms',
            'explicit', 'tempo', 'valence', 'mode', 'liveness', 'speechiness']

##Assign features's value
X = DF[features]
Y = DF['popularity_level']
new_best_features = Feature_Selection()
X = DF[new_best_features]
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=420)

# X_test.to_csv("X_Test_Classification.csv", index=False)
# y_test.to_csv("Y_Test_Classification.csv", index=False)

##Run Classification Models
KNN = KNN_Classifier()
Decisiontree = DecisionTreeClassifieer()
AdaBoost = AdaBoost_Classifier()
polynomial = Polynomial_KernalSVM()
drawBarGraph('Accuracy', 'Value', ['KNN Accuracy', 'Decision tree Accuracy', 'AdaBoost  Accuracy', 'PolynomialSVM Accuracy'],
             [KNN[0], Decisiontree[0], AdaBoost[0], polynomial[0]], 'Accuracy Graph')
drawBarGraph('Training Time', 'Value', ['KNN Training time', 'Decision tree Training time', 'AdaBoost Training time', 'PolynomialSVM Training time'],
             [KNN[1], Decisiontree[1], AdaBoost[1],polynomial[1]], 'Training time Graph')
drawBarGraph('Test time', 'Value', ['KNN Test time', 'Decision tree Test time', 'AdaBoost  Test time', 'PolynomialSVM  Test time'],
             [KNN[2], Decisiontree[2], AdaBoost[2], polynomial[2]], 'Test time Graph')
#Guasian_KernalSVM()