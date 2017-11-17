# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 09:42:19 2017

@author: benps
"""

print('Importing packages...')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('bmh')
#%%
plt.rcParams['axes.facecolor'] = 'w'
#%%
# Load predictions
predictions = pd.read_csv('predictions.csv', index_col = 0)
#%%
df = predictions.copy()
df = df.rename(columns={'1':'WnvPresent'})
df.drop(['0', 'Species'], axis =1, inplace=True)
df = df.groupby(by=['Trap','Week']).max().reset_index()


# Reshape predictions so we have predictions per week for every trap.
# Rows are weeks, columns are traps.
prediction_by_week = df.pivot_table(index='Week', columns = 'Trap', values = 'WnvPresent')
#%%
# Save to csv
# prediction_by_week.to_csv('prediction_by_week.csv')

#%%

# For a given trap (column), apply treatment if probability of infection is above
#  a certain threshold. Based on our research, we decided not to re-apply treatment if
#  the area around the trap was treated the previous week.

def apply_treatment(trap, risk_threshold):
    trap_col = trap.copy()
    for id in range(23,41):
        if (trap_col[id] > risk_threshold) and (trap_col[id-1] < 1):
            trap_col[id] = 1
        else:
            trap_col[id] = 0
    return(trap_col)
    
#%%
# We assume treatment occurs by spraying per linear mile. Research suggested we would
#  treat a 2km by 2km square around each test point. With a cost of $68 per linear mile
#  and an average of 36 linear miles per square, we compute the cost of treating the area
#  of one trap to be $3,204. In areas with more linear miles of road (denser areas),
#  we simply don't spray along every single road. This makes sense.

cost_of_treatment = 3204

# Apply treatment to every column of our dataset for 100 threshold values. Each
#  threshold value is a level where we decide to classify a trap as infected with
#  WNV. Threshold can be interpreted as risk tolerance.
# Multiply the total number of sprays by the individual cost to come up with
#  a dataframe giving total cost for each risk threshold.
costs = pd.DataFrame(columns=['Threshold','Total Cost of Adulticide'])
for i in range(0,101):
    costs.loc[len(costs)] = (i/100,
             prediction_by_week.apply(apply_treatment, risk_threshold=i/100).sum().sum() * cost_of_treatment)
    
    


#%% 
# Show how total cost responds to our risk threshold.
plt.clf()
plt.figure(facecolor='w')
plt.plot(costs['Threshold'], costs['Total Cost of Adulticide']/1000000)
#plt.title('Risk Tolerance Determines Spraying Cost')
plt.ylabel('Spraying Cost, millions of dollars')
plt.xlabel('Risk Threshold')
plt.show()