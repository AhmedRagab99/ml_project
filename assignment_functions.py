import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, PolynomialFeatures
from sklearn.utils import shuffle
import seaborn as sns

df = pd.read_csv('spotify_training.csv')
df.groupby(df['key']).size()

pd.set_option("display.max.columns", None)
pd.set_option("display.precision", 2)

# data preproccesing  exept the artist convertion with one_ hot_ encoder

features = ['year','acousticness','energy','loudness']
Y = df['popularity']
df = df.apply(pd.to_numeric, errors='coerce')
# replace the empty value for the features with the mean of the column  exept the artist column
df['acousticness'] = df['acousticness'].replace(np.nan,np.mean(df['acousticness']))
df['energy'] = df['energy'].replace(np.nan,np.mean(df['energy']))
df['loudness'] = df['loudness'] .replace(np.nan,np.mean(df['loudness']))
df['year'] = df['year'].replace(np.nan,np.mean(df['year']))




#  split the test and train data with 80% for the train and the rest for test

training = df.sample(frac = 0.8,random_state = 420)
# X_train = df[features].values.reshape(-1, len(features))
X_train = df[features]
y_train = training['popularity']
X_test = df.drop(training.index)[features]
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.2, random_state = 0)
#
#

# Logistec regression
logisticRegr = LogisticRegression()
logisticRegr.fit(X_train, y_train)
predictions = logisticRegr.predict(X_test)
score = logisticRegr.score(X_test, y_valid)
print(score)

# Polynomial regression
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X_train)


#  Linear regression
cls = LinearRegression()
X=np.expand_dims(X_train, axis=1)
Y=np.expand_dims(y_train, axis=1)
cls.fit(X_train,y_train) #Fit method is used for fitting your training data into the model
prediction= cls.predict(X_test)
error = metrics.mean_squared_error(y_valid, prediction)




plt.figure(figsize=(10,6))
heatmap = sns.heatmap(df.corr(), vmin=-1,vmax=1, annot=True, cmap='viridis')
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12)
plt.show()