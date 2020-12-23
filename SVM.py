# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn import metrics
# from sklearn.linear_model import LogisticRegression, LinearRegression
# from sklearn.metrics import accuracy_score, roc_auc_score
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder, PolynomialFeatures
# from sklearn.utils import shuffle
# from assignment_functions import *
# import seaborn as sns
#
# df = pd.read_csv('spotify_training.csv')
# pd.set_option("display.max.columns", None)
# pd.set_option("display.precision", 2)
# # print(df.keys())
# # print(pd.isnull(df).sum())
# #
# # label_encoder = LabelEncoder()
# # # Encode labels in column 'Country'.
# # df['artists'] = pd.factorize(df['artists'], sort=True)[0]
# #
# # df['artists']= label_encoder.fit_transform(df['artists'])
# # print(df['artists'])
#
# #
# # label = LabelEncoder()
# # df['popularity'] = pd.factorize(df['artists'], sort=True)[0]
# #
# #
# # onehot_data = OneHotEncoder(sparse=False)
# # onehot_data = onehot_data.fit_transform(df['artists'])
# # print("Categorical data encoded into integer values....\n")
# # print(onehot_data)
#
#
#
#
#
# LR = LinearRegression()
# features = ['year','acousticness','artists','energy','loudness']
# Y = df['popularity']
#
# # integer encode
# label_encoder = LabelEncoder()
# integer_encoded = label_encoder.fit_transform(Y)
# # binary encode
# onehot_encoder = OneHotEncoder(sparse=False)
# integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
# df['artists'] = onehot_encoder.fit_transform(integer_encoded)
# print(df['artists'])
#
#
#
# training = df.sample(frac = 0.8,random_state = 420)
# X_train = training[features]
# y_train = training['popularity']
# X_test = df.drop(training.index)[features]
# X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.2, random_state = 420)
#
# cls = LinearRegression()
# # X=np.expand_dims(X_train, axis=1)
# # Y=np.expand_dims(y_train, axis=1)
# cls.fit(X_train,y_train) #Fit method is used for fitting your training data into the model
# prediction= cls.predict(X_test)
#
# error = metrics.mean_squared_error(y_valid, prediction)
# print(error)
#
#
# # invert first example
# # inverted = label_encoder.inverse_transform([np.argmax(onehot_encoded[0, :])])
# # print(inverted)
#
# plt.figure(figsize=(10,6))
# heatmap = sns.heatmap(df.corr(), vmin=-1,vmax=1, annot=True, cmap='viridis')
# heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12)
# plt.show()
#
# poly_reg = PolynomialFeatures(degree=3)
# # X=np.expand_dims(X, axis=1)
# # Y=np.expand_dims(Y, axis=1)
# # X_poly = poly_reg.fit_transform(np.array(X))
# # prediction= poly_reg.predict(X)
# #
# # error = metrics.mean_squared_error(Y, prediction)
# # print(X_poly)
#
# # onehotencoder = OneHotEncoder()
# # #reshape the 1-D country array to 2-D as fit_transform expects 2-D and finally fit the object
# # X = onehotencoder.fit_transform(df.artists.values.reshape(2,120997)).toarray()
# # #To add this back into the original dataframe
# # dfOneHot = pd.DataFrame(X, columns = ["artists"+str(int(i)) for i in range(df.shape[1])])
# # df = pd.concat([df, dfOneHot], axis=1)
# # #droping the country column
# # df= df.drop(['artist'], axis=1)
# # #printing to verify
# # print(df.head())
# # # print(df.tail(),len(df))
# # plt.figure(figsize=(10,6))
# # heatmap = sns.heatmap(df.corr(), vmin=-1,vmax=1, annot=True, cmap='viridis')
# # heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12)
# # plt.show()
# # target = df['variety']
# # df = df.drop(df.index[rows])
# #
# # x = df['sepal.length']
# # x2 = df['petal.length']
# #
# # setosa_x = x[:50]
# # setosa_x2 = x2[:50]
# #
# # versicolor_x = x[50:]
# # versicolor_x2 = x2[50:]
# #
# # plt.figure(figsize=(8,6))
# # plt.scatter(setosa_x,setosa_x2,marker='+',color='green')
# # plt.scatter(versicolor_x,versicolor_x2,marker='_',color='red')
# # plt.show()
# #
# # ## Drop rest of the features and extract the target values
# # df = df.drop(['sepal.width','petal.width'],axis=1)
# # Y = []
# # target = df['variety']
# # for val in target:
# #     if(val == 'Setosa'):
# #         Y.append(-1)
# #     else:
# #         Y.append(1)
# # df = df.drop(['variety'],axis=1)
# # X = df.values.tolist()
# # ## Shuffle and split the data into training and test set
# # X, Y = shuffle(X,Y)
# #
# # X = np.array(X)
# # Y = np.array(Y)
# #
# # Y = Y.reshape(100,1)
# #
# # y_pred,w = fit(X,Y)
# #
# # ## Predict
# #
# # predictions = []
# # for val in y_pred:
# #     if(val >= 0):
# #         predictions.append(1)
# #     else:
# #         predictions.append(-1)
# #
# # print(accuracy_score(Y,predictions))
# #
# # #plot decision boundary
# # min_f1 = min(df['sepal.length'])
# # max_f1 = max(df['sepal.length'])
# # x_values = [(min_f1 - 1),(max_f1 + 1)]
# # y_values = - (w[0] + np.multiply(w[1], x_values)) / w[2]
# # plt.plot(x_values, y_values, label='Decision Boundary')
# # plt.scatter(setosa_x,setosa_x2,marker='+',color='green')
# # plt.scatter(versicolor_x,versicolor_x2,marker='_',color='red')
# # plt.show()