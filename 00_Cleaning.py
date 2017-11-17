# Import Statements

print("Chicago West Nile Virus -- Team 3 \n Data Cleaning Script \n Importing packages...")

import pandas as pd
import numpy as np
import datetime
from haversine import haversine

#%%
# Importing Data
print("Importing data...")

# Don't edit raw dataframes after they have been imported:
train_raw = pd.read_csv('assets/train.csv')
test_raw = pd.read_csv('assets/test.csv', index_col=0)
weather_raw = pd.read_csv('assets/weather.csv')
spray_raw = pd.read_csv('assets/spray.csv')

#%%

# Versions of the raw data that we will be cleaning
train = train_raw.copy().drop(['Address','Block','Street','AddressNumberAndStreet','AddressAccuracy'], axis=1)
test = test_raw.copy().drop(['Address','Block','Street','AddressNumberAndStreet', 'AddressAccuracy'], axis=1)
# If we're using latitude and longitude we can drop out the address info.

# Turn date columns into datetime date types
train['Date'] = pd.to_datetime(train['Date'], format='%Y/%m/%d')
test['Date'] = pd.to_datetime(test['Date'], format='%Y/%m/%d')

weather = weather_raw.copy()
spray = spray_raw.copy()

#%%
# Checking for Missing Data and Non-numeric Values
print("Cleaning weather data...")

# Formatting datetime
weather['Date'] = pd.to_datetime(weather['Date'], format='%Y/%m/%d')


# Replacing missing with Null and only keeping certain columns:
weather = weather[['Station', 'Date', 'Tmax', 'Tmin', 'Tavg',
                   'Depart', 'DewPoint', 'WetBulb', 'PrecipTotal','Sunrise','Sunset']].replace('M', np.NaN)
weather = weather[['Station', 'Date', 'Tmax', 'Tmin', 'Tavg',
                   'Depart', 'DewPoint', 'WetBulb', 'PrecipTotal','Sunrise','Sunset']].replace('-', np.NaN)
# Sunrise and sunset data is only available from one station. That is fine. Replace '-' from that station with Null.
weather = weather[['Station', 'Date', 'Tmax', 'Tmin', 'Tavg',
                   'Depart', 'DewPoint', 'WetBulb', 'PrecipTotal','Sunrise','Sunset']].replace('  T', .01)
# Replace trace precipitation with some small number.


weather_consolidated = weather.groupby(by='Date').agg(lambda x: np.nanmean(pd.to_numeric(x)))
weather_consolidated.drop(['Station','Sunrise','Sunset'], axis = 1, inplace=True)
# The two weather stations are at the airport. They aren't exactly on opposite sides of town,
#  so many points are about equidistant from them both. Our weather data will simply be an
#  average of the weather observations. This also has the effect of handling missing data;
#  if a certain variable is missing from one station, it is provided by the other.
# Sunrise and suset could tell us hours of daylight, but we already have that by taking date into account.


# No null values in our averaged and consolidated data set!
print('Null values in combined weather data: \n' + str(weather_consolidated.isnull().sum()))
#%%
# Cleaning Data and EDA, spray

# Missing data and the like:
print('Cleaning spray data...')

spray.drop('Time', axis = 1, inplace=True)
# We don't care exactly what time the spraying occured.

spray['Date'] = pd.to_datetime(spray['Date'], format='%Y/%m/%d')


#%%
# Merging Train/Test Data and Weather Data

print('Merging weather data with train and test data \n using 2 week and 3 day rolling averages...')

two_week_rolling_average = weather_consolidated.rolling(window=14, min_periods=1).mean().reset_index()
three_days_rolling_average = weather_consolidated.rolling(window=7, min_periods=1).mean().reset_index()

# Based on our research of mosquito lifespan, we decided that the average weather 
# for the last two weeks would be an important factor to consider. We also included 
# a three day average, so our model can consider more recent weather conditions as well.

# In[32]:

train = train.merge(two_week_rolling_average, on='Date')
test = test.merge(two_week_rolling_average, on='Date')
train = train.merge(three_days_rolling_average, on='Date', suffixes = ('_2wks','_3dys'))
test = test.merge(three_days_rolling_average, on='Date', suffixes = ('_2wks','_3dys'))

#%%
print('After merging weather data with train and test data, \n there are ' +
      str(train.isnull().sum().sum() + test.isnull().sum().sum()) +
          ' missing values...')
#%%

# The following lines take into account where spraying has occured in the training set.
# First, we define a column with Latitude and Longitude as a tuple.

# train['loc'] = list(zip(train['Latitude'], train['Longitude']))
# test['loc'] = list(zip(test['Latitude'], test['Longitude']))
# spray['loc'] = list(zip(spray['Latitude'], test['Longitude']))

#%%
# Compute whether an iterable of points is within a certain distance of a given point

def bool_vector_haversine(point, series, dist):
    return([haversine(point, i) < dist for i in series])

#%%
# Merging Train/Test Data with Spray Data


print('Defining function to check whether a spray has occured nearby for a particular row...')

def recent_spray(row_tup, dist, days):
    '''Function to determine if, for a given row in training set, there has been a spray within
    a certain lat_long_dist and specified number of days'''

    date = row_tup[0]
    loc  = row_tup[1]

    mask1 = bool_vector_haversine(loc, spray['loc'], dist = dist)
    spray_filt = spray[mask1]
    mask2 = date > spray_filt['Date']
    spray_filt = spray_filt[mask2]
    mask3 = spray_filt['Date'] > (date - datetime.timedelta(days=days))
    spray_filt = spray_filt[mask3]
    
    return spray_filt.shape[0]

# This function will tell you how many sprays have occured within a certain number of 
#  kilometers and number of days from a given point. It is an incredibly time consuming
#  process. 
# The result is that including spraying data does not help the predictive power of a 
#  model fit to the training data. Furthermore, we don't know when spraying occured
#  in the test data, so we can't use spray data in the final model anyway.

#%%
# print('Checking for sprays within 1 week for training set...')
# train['Sprayed_lastweek_close'] = [recent_spray(row_tup, .5, 7) for row_tup in zip(train['Date'],train['loc'])]
# train['Sprayed_lastweek_far'] = [recent_spray(row_tup, 2, 7) for row_tup in zip(train['Date'],train['loc'])]

#%%
# print('Checking for sprays within 2 weeks for training set...')
# train['Sprayed_last2week2_close'] = [recent_spray(row_tup, .5, 14) for row_tup in zip(train['Date'],train['loc'])]
# train['Sprayed_last2weeks_close'] = [recent_spray(row_tup, .5, 14) for row_tup in zip(train['Date'],train['loc'])]


#%%
# Merging All Years Together: Turning Date Column in to 'Week Of' Column
    
# We make an important simplifying assumption here: since each year has data 
#  for a few weeks every year, we assume that the same data generating process 
#  is being repeated basically the same way every year, except for differences in
#  in weather, which we take account of in the model.
# Therefore, we drop out dates and only consider 'week of the year' moving forward

print('Turning \'Date\' into \'Week of\'...')

train['Week'] = train['Date'].dt.week
test['Week'] = test['Date'].dt.week

#%%
# Elevation Data

# This data is courtesy of Steve
print('Importing elevation data...')
train_elev = pd.read_csv('assets/TrainWithAlt.csv')
test_elev = pd.read_csv('assets/TestWithAlt.csv')

train['Altitude (m)'] = train_elev['Altitude (m)']
test['Altitude (m)'] = test_elev['Altitude (m)']

#%%
print('Loading zoning info...')

# Zoning data, fetched and simplified by Steve

zoning = pd.read_csv('assets/Zoning.csv')[['Trap','Zone Type']]
train = train.merge(zoning, on='Trap')
train[train['Zone Type'].isnull()]['Trap'].unique()

#%%
test = test.merge(zoning, on='Trap')
test[test['Zone Type'].isnull()]['Trap'].unique()


#%%

# Do brief EDA, check for missing values, and save to csv. Can be run in terminal.

input("Press enter to display train and test heads: ")
print(train.head())
print(test.head())
input("Missing values: ")
print(train.isnull().sum())
print(test.isnull().sum())

#save = input("Save to CSV? y/n ")

#if save == 'y':
 #   train.to_csv('Final_train.csv', index=False)
 #   test.to_csv('Final_test.csv', index=False)