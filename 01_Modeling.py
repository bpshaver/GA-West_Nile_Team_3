# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 20:34:11 2017

@author: benps
"""
import datetime
import pandas as pd
import numpy as np
#%%
train = pd.read_csv('Final_train.csv', encoding='utf-8', index_col='Trap')
# test = pd.read_csv('Final_test.csv', encoding='utf-8', index_col='Trap')

print([x for x in test.columns if x not in train.columns])
print([x for x in train.columns if x not in test.columns])

#%%
from sklearn.model_selection import train_test_split

model_data = train.select_dtypes(include=[np.number])
X_train, X_test, y_train, y_test = train_test_split(model_data.drop('WnvPresent', axis=1), model_data['WnvPresent'])


#%%

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from sklearn import metrics
from sklearn.model_selection import train_test_split

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

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
y = model_data['WnvPresent']
X = model_data.drop(['NumMosquitos','WnvPresent'], axis = 1)

# X = X.drop(['Sprayed_lastweek_close','Sprayed_lastweek_far', 'Sprayed_last2weeks_close',
#        'Sprayed_last2weeks_far'], axis = 1)
# Keep or drop spray data?

X = scaler.fit_transform(X)
y = np_utils.to_categorical(y)


X_train, X_test, y_train, y_test = train_test_split(X, y)

X_train = np.array(X_train)
y_train = np.array(y_train)


model = build_model(X_train.shape[1], 2)

model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=1)

#%%
print(metrics.roc_auc_score(y_test, model.predict(X_test)))

#%%
from tpot import TPOTClassifier



pipeline_optimizer = TPOTClassifier(max_time_mins=60, population_size=20, cv=5,
                                    random_state=42, verbosity=2, scoring='roc_auc')

pipeline_optimizer.fit(X_train, y_train)
#%%
print(pipeline_optimizer.score(X_test, y_test))
#%%
pipeline_optimizer.export('tpot_exported_pipeline.py')

#%%
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
#%%

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=42)

# Score on the training set was:0.8732657231806037
exported_pipeline = make_pipeline(
    RFE(estimator=ExtraTreesClassifier(criterion="entropy", max_features=0.4, n_estimators=100), step=0.45),
    StackingEstimator(estimator=LogisticRegression(C=0.5, dual=False, penalty="l2")),
    ExtraTreesClassifier(bootstrap=False, criterion="gini", max_features=0.3, min_samples_leaf=16, min_samples_split=16, n_estimators=100)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
#%%
model_data = train.select_dtypes(include=[np.number])
X_train, X_test, y_train, y_test = train_test_split(model_data.drop('WnvPresent', axis=1), model_data['WnvPresent'])

xt = ExtraTreesClassifier(bootstrap=False, criterion="gini", max_features=0.3, min_samples_leaf=16, min_samples_split=16, n_estimators=100)
xt.fit(X_train, y_train)

print(metrics.roc_auc_score(y_test, xt.predict(X_test)))
#%%