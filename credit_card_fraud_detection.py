import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')

#importing the dataset
dataset = pd.read_csv('creditcard.csv')
X = dataset.iloc[:,0:-1].values
y = dataset.iloc[:,-1].values

# Split the data into training and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=0)
# Feature scaling of data variables
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#using the ann
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

clf = Sequential()
clf.add(Dense(units=128,activation='relu',kernel_initializer='uniform',
              input_dim=30))
clf.add(Dropout(0.5))
clf.add(Dense(units=100,activation='relu'))
clf.add(Dropout(0.2))
clf.add(Dense(units=1,activation='sigmoid',kernel_initializer='uniform'))

#compling our ann
clf.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
clf.fit(X_train, y_train, batch_size=200,epochs=100,verbose=1)

#predicting the model
y_pred = clf.predict(X_test)
y_pred = (y_pred > 0.5)

#measuring the accuracy of the model
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)














