# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 20:34:11 2017

@author: benps
"""

# Importing necessary packages
print('Importing packages...')

import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from sklearn import metrics

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#%%
print('Importing data...')

train = pd.read_csv('Final_train.csv', encoding='utf-8', index_col='Trap')
test = pd.read_csv('Final_test.csv', encoding='utf-8', index_col='Trap')

print('Columns in test set not contain in training set:')
print([x for x in test.columns if x not in train.columns])
print('Columns in training set not contained in test set:')
print([x for x in train.columns if x not in test.columns])

input('Press a key to continue...')


#%%

# Create dummy variables for Species and Zone Type
train = pd.get_dummies(train, columns=['Species','Zone Type'])
# Manually insert dummy column into training set for species value that does not
#  appear in the training set, but does occur in the test set.
train.insert(26,'Species_UNSPECIFIED CULEX', 0)
# We can't make predictions based on the number of mosquitos in each trap, since 
#  that info is not contained in the test set.
train.drop(['Date','NumMosquitos'], axis =1, inplace=True)

# Select all numeric columns
train = train.select_dtypes(include=[np.number])

# Same as above, but for test set.
test = pd.get_dummies(test, columns=['Species','Zone Type'])
test = test.select_dtypes(include=[np.number])


#%%
# Build a certain model architecture based on a number of input dimensions and 
#  output dimensions
print('Building neural network model...')
def build_model(input_dim, output_dim):
    model = Sequential()
    model.add(Dense(32, input_dim=input_dim))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(output_dim))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    return model

# Instanntiate scaler
scaler = StandardScaler()

# Define independent and dependent variables
y = train['WnvPresent']
X = train.drop(['WnvPresent'], axis = 1)

# When we included spray data, we wanted to drop it out and test model performance
# It made no difference.
# X = X.drop(['Sprayed_lastweek_close','Sprayed_lastweek_far', 'Sprayed_last2weeks_close',
#        'Sprayed_last2weeks_far'], axis = 1)

# Scale independent variables.
X = scaler.fit_transform(X)
# Encode dependent variable as needed
y = np_utils.to_categorical(y)

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Save our training set as an array
X_train = np.array(X_train)
y_train = np.array(y_train)

model = build_model(X_train.shape[1], y_train.shape[1])

input('Press any key to fit model...')

# Fitting the model with class weights to balance classes
model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=1, class_weight= {0:1, 1:18})
#%%
# Evaluate model performacne:
print('Model AUC: ')
print(metrics.roc_auc_score(y_test, model.predict(X_test)))
#%%
# Scale test set and save as array
test_array = scaler.transform(test)
test_array = np.array(test_array)

#%%
# Predict probabilities
predictions = model.predict_proba(test_array)
predictions = pd.DataFrame(predictions)

#%%
# Re-join the identifying variables to the predictions:
traps = pd.read_csv('Final_test.csv', encoding='utf-8')[['Trap','Week','Species']]
predictions = predictions.join(traps)

#%%
#save = input("Save predictions to CSV? y/n ")

#if save == 'y':
    #predictions.to_csv('Predictions.csv')

