# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 19:38:57 2017

@author: benps
"""

import pandas as pd
np.random.seed(6)
#%%

predictions = pd.read_csv('Predictions.csv', index_col=0).rename(columns={'1':'WnvPresent'})

predictions['Id'] = range(1,predictions.shape[0]+ 1)

submission = predictions[['Id','WnvPresent']]

submission.to_csv('submission.csv', index=False)