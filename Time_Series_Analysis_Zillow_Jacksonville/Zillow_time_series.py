#!/usr/bin/env python
# coding: utf-8

# # Zillow house sales time series analysis
# 
# For this project, I am looking at zillow house sales price data for **Jacksonville, FL**to do  from January 2000 to July 2023. I will first collect the data for Jacksonville from all the counties in USA and then clean it. The aim of the project is to **identify the 5 zip codes that are going to see the most increase in sale prices** in the next three years. I will model the prices using ARIMA and use that to predict the future prices. 
# 
# For the exploratory data analysis, I have plotted the Number of cities in Jacksonville Metropilitan Area and number of zipcodes in them. I have also plotted the number of zip codes in each city that fit the budget constraint of in between 250K and 300K. I have also shown price change as a function of year since 2000. We will see that **zip code 32219 is expected to grow the most in Jacksonville by 34.03%.**

# In[1]:


# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Reading the data.
df = pd.read_csv('zillow_new_data.csv')
df.head()


# In[3]:


# Collecting data for Jacksonville
duval = df[df['CountyName'] == 'Duval County']
clay = df[df['CountyName'] == 'Clay County']
baker = df[df['CountyName'] == 'Baker County']
nassau = df[df['CountyName'] == 'Nassau County']
st_johns = df[df['CountyName'] == 'St. Johns County']

jax = pd.concat([duval, clay, baker, st_johns, nassau], ignore_index=True)
jax


# In[4]:


# Some counties were common to other states as well.
# Only want to use for Jacksonville, FL.
jax = jax[jax['State'] == 'FL']
del jax['StateName']
jax.head()


# ### Exploratory Data Analysis

# In[5]:


# Number of cities in Jacksonville Metropilitan Area.
# and number of zipcodes in them.
fig, ax = plt.subplots(figsize = (15,12))

y = [z for z in jax['City'].value_counts()]
x = [z for z in jax['City'].value_counts().keys()]
ax.barh(x,y)
ax.set_title("Cities in Jacksonville Metropolitan area", fontsize = 30)
ax.set_xlabel("Number of zipcodes in the city")
ax.set_ylabel("Cities")


# In[6]:


# Zip codes that are in the budget of 250000 to 300000.
# Otherwise the zip codes list was becoming too long.
jax_budget = jax[(jax['2023-07-31'] <= 300000) & (jax['2023-07-31'] >= 250000)]


# In[7]:


jax_budget_zips = [z for z in jax_budget['RegionName']]
jax_budget_zips


# In[8]:


# Plotting number of zip codes in each city that fit this budget constraint.
from collections import Counter

city_names = []

for zip in jax_budget_zips:
    city_names.append(jax_budget[jax_budget['RegionName'] == zip].iloc[0]['City'])
    
count_dict = {}
for z in Counter(city_names).keys():
    count_dict[z] = Counter(city_names)[z]
 
count_dict = dict(sorted(count_dict.items(), key=lambda item: item[1]))

#Building the bar chart
fig,ax = plt.subplots(figsize=(21,12))

x_labels = [a for a in count_dict.keys()]
x = list(range(1,len(x_labels)+1))
y = [a for a in count_dict.values()]

ax.bar(x,y,width=0.5)

ax.set_xticks(x)
ax.set_xticklabels(x_labels,fontsize='12')
ax.set_title("Cities with zipcodes fitting the budget",fontsize='30')
ax.set_ylabel("Number of Zipcodes",fontsize='20')
ax.set_xlabel("Cities in Jacksonville Metropolitan Area",fontsize='20');


# ### Visualizing the price history of these zip codes.

# In[14]:


jax_budget.head()


# In[15]:


del jax_budget['RegionType']


# In[16]:


zip_dict = {}

for zipcode in jax_budget_zips:
    filtered_data = jax_budget[jax_budget['RegionName'] == zipcode]
    
    melted_data = pd.melt(filtered_data, id_vars=['RegionName', 'RegionID', 'SizeRank', 'City', 'State', 'Metro', 'CountyName'], 
                          var_name='time')
    
    melted_data['time'] = pd.to_datetime(melted_data['time'], infer_datetime_format=True)
    
    # Drop rows with missing 'value'
    melted_data = melted_data.dropna(subset=['value'])
    
    # Group and aggregate the data
    aggregated_data = melted_data.groupby('time').aggregate({'value':'mean'})
    
    # Store the aggregated data in the zip_dict dictionary
    zip_dict[zipcode] = aggregated_data


# In[17]:


fig,ax = plt.subplots(figsize=(20,12))

for zipcode in zip_dict:
    ax.plot(zip_dict[zipcode],)

ax.axhline(y=300000,label = 'Client Budget')   

ax.set_title('ZipCode Price Changes since 2000',fontsize=30)
ax.set_ylabel('Price in US $',fontsize=20)
ax.set_xlabel('Year',fontsize=20)
ax.legend(prop={'size': 25});


# You can clearly see the effects of recession in 2008 on the market. **A person who bought a house in 2006 would not make any profit in the house till 2020.**
# 
# 

# ### Average Price Growth since 2018
# 
# I am checking since 2018, because we happened to buy a house in 2018.

# In[ ]:





# In[18]:


roi_2018 = jax_budget.groupby('RegionName').apply(lambda x: ((x['2023-07-31'] - x['2018-01-31']) / x['2018-01-31']) * 100)
roi_2018 = roi_2018.rename('roi').reset_index()
roi_2018.head(20)


# In[19]:


fig, ax = plt.subplots(figsize=(12, 8))

x_labels = [str(a) for a in roi_2018['RegionName']]
x = list(range(1,12))
y = [a for a in roi_2018['roi']]

ax.bar(x,y)

ax.set_xticks(x)
ax.set_xticklabels(x_labels)


# In[ ]:





# ### Selecting zip code 32205 to build our ARIMA prediction model.

# In[20]:


def melt_data(df):
    melted = pd.melt(df, id_vars=['RegionName', 'RegionID', 'SizeRank', 'City', 'State', 'Metro', 'CountyName'], var_name='time')
    melted['time'] = pd.to_datetime(melted['time'], infer_datetime_format=True)
    melted = melted.dropna(subset=['value'])
    return melted.groupby('time').aggregate({'value':'mean'})



# In[22]:


# melting the data to a time series.
zip32205 = jax[jax['RegionName'] == 32205]
    
melted = pd.melt(zip32205, id_vars=['RegionName', 'RegionID', 'SizeRank', 'City', 'State', 'Metro', 'CountyName'], 
                      var_name='time')
    
melted['time'] = pd.to_datetime(melted['time'], infer_datetime_format=True)
melted = melted.dropna(subset=['value'])

aggregated = melted.groupby('time').aggregate({'value':'mean'})
series32205 = aggregated
series32205


# In[23]:


#Visualizing our Time Series Data for zipcode- 32205

fig, ax = plt.subplots(figsize=(20,12))
ax.plot(series32205)

ax.set_xlabel('Year', fontsize=30)
ax.set_ylabel('Price in 100,000($)',fontsize=20)
ax.set_title('Zip Code 32205 Price History',fontsize=30);


# ### Modelling
# 
# Since the data from 2007 - 2010 was not normal and was caused by very special circumstances. We will use data from 2011 to model and make predictions.

# In[24]:


recent_series = series32205['2011':]
recent_series


# In[25]:


#Visualizing our Time Series Data for zipcode- 32205 from 2011

fig, ax = plt.subplots(figsize=(20,12))
ax.plot(recent_series)

ax.set_xlabel('Year', fontsize=20)
ax.set_ylabel('Price in 100,000($)',fontsize=20)
ax.set_title('Zip Code 32005- Price since 2011',fontsize=30);


# In[26]:


train_series = recent_series[:'2017-12-31']
train_series


# In[27]:


test_series = recent_series['2018-01-31':]
test_series


# In[ ]:





# In[28]:


# running auto ARIMA to get p,d,q values
import pmdarima as pm

auto_model = pm.auto_arima(train_series['value'], start_p=0, start_q=0,
                           test='adf',
                           max_p=5, max_q=5,
                           m=1,
                           d=0,
                           seasonal=True,
                           start_P=0, start_Q=0,
                           D=0,
                           trace=True,
                           error_action='ignore',
                           suppress_warnings=True,
                           stepwise=True, with_intercept=False)

print(auto_model.summary())


# In[29]:


#Identifying the order values for our model
auto_model.order


# In[30]:


#Identifying the Seasonal Order values for our model
auto_model.seasonal_order


# In[31]:


import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Plug the optimal parameter values into a new SARIMAX model
ARIMA_MODEL = SARIMAX(train_series, 
                      order=(1, 0, 1), 
                      seasonal_order=(0, 0, 0, 0), 
                      enforce_stationarity=False, 
                      enforce_invertibility=False)

# Fit the model and print results
output = ARIMA_MODEL.fit()

print(output.summary().tables[1])
auto_model.plot_diagnostics(figsize=(18, 18))
plt.show()


# In[32]:


# Get predictions starting from 04-01-2015 and calculate confidence intervals
pred = output.get_prediction(start=pd.to_datetime('2018-01-31'), end=pd.to_datetime('2023-07-31'), dynamic=False)
pred_conf = pred.conf_int()
pred_conf


# In[33]:


# Plot real vs predicted values along with confidence interval
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 18, 8

# Plot observed values
ax = recent_series.plot(label='observed')

# Plot predicted values
pred.predicted_mean.plot(ax=ax, label='Prediction Series', alpha=0.9)

# Plot the range for confidence intervals
ax.fill_between(pred_conf.index,
                pred_conf.iloc[:, 0],
                pred_conf.iloc[:, 1], alpha=0.5,label = 'Confidence Interval')

# Set axes labels
ax.set_xlabel('Date',fontsize=20)
ax.set_ylabel('Price',fontsize=20)
ax.set_title('Testing Forecasting Model Performance',fontsize=30)
plt.legend()

plt.show()


# ### Price forecasting for 32205.

# In[34]:


# Plug the parameter values from our Auto ARIMA model into a new SARIMAX model that fits the entire series.
ARIMA_MODEL = sm.tsa.statespace.SARIMAX(recent_series, 
                                        order=(1,0,1), 
                                        seasonal_order=(0, 0, 0, 0), 
                                        enforce_stationarity=False, 
                                        enforce_invertibility=False)

# Fit the model and print results
full_output = ARIMA_MODEL.fit()

print(output.summary().tables[1])


# In[35]:


## Getting a forecast for the next 36 months after the last recorded date on our dataset.
forecast = full_output.get_forecast(36)
future_prediction = forecast.conf_int()
future_prediction['value'] = forecast.predicted_mean
future_prediction.columns = ['lower','upper','prediction'] 
future_prediction


# In[36]:


#Plotting our Forecast

fig, ax = plt.subplots()
recent_series.plot(ax=ax,label='Real Values')


future_prediction['prediction'].plot(ax=ax,label='predicted value',ls='--')

ax.fill_between(x= future_prediction.index, y1= future_prediction['lower'], 
                y2= future_prediction['upper'],color='lightpink',
                label='Confidence Interval')
ax.legend() 
plt.ylabel("Average Price")
plt.title('Average Home Price - 32205 - With Forcasted Value & Confidence Intervals')
plt.show()


# ### Making predictions for all zip codes and calculating return on investment.

# In[37]:


def melt_data(df):
    melted = pd.melt(df, id_vars=['RegionName', 'RegionID', 'SizeRank', 'City', 'State', 'Metro', 'CountyName'], 
                      var_name='time')
    
    melted['time'] = pd.to_datetime(melted['time'], infer_datetime_format=True)
    melted = melted.dropna(subset=['value'])
    return melted.groupby('time').aggregate({'value':'mean'})


# In[38]:


zip_predictions = {}

#miami_budget_zips is a list of zipcodes in Jax County with average price under $300000
for zipcode in jax_budget_zips:
    
    series = melt_data(jax[jax['RegionName'] == zipcode])
    
    #Only taking data from 2011 onwards to more accurately reflect current market conditions
    recent_series = series['2011':]
    
    # Splitting the last 36 months of our series as a test dataset.
    train_series = recent_series[:'2017-12-31']
    test_series = recent_series['2018-01-31':]
    print(test_series)
    #Auto ARIMA model
    auto_model = pm.auto_arima(train_series, start_p=0, start_q=0,
                     test='adf',
                     max_p=5, max_q=5,
                     m=1,
                     d=0,
                     seasonal = True,
                     start_P=0,start_Q=0,
                     D=0,
                     trace=True,
                     error_action= 'ignore',
                     suppress_warnings=True,
                     stepwise=True,with_intercept=False)
   
#Plug the optimal parameter values for our Training data into a SARIMAX model that fits our entire series.
    ARIMA_MODEL = sm.tsa.statespace.SARIMAX(recent_series, 
                                            order= auto_model.order, 
                                            seasonal_order= auto_model.seasonal_order, 
                                            enforce_stationarity=False, 
                                            enforce_invertibility=False)

    # Fit the model and print results
    output = ARIMA_MODEL.fit()

    ## Getting a forecast for the next 36 months after the last absrecorded date on our dataset.
    forecast = output.get_forecast(36)
    prediction = forecast.conf_int()
    prediction['value'] = forecast.predicted_mean
    print("###################")
    print(forecast.predicted_mean)
    prediction.columns = ['lower','upper','prediction']
    
    #Adding the Zipcode's ROI to the zip_predictions dictionary
    zip_predictions[zipcode] = ((prediction['prediction'][-1])
                                - (series['value'][-1]))/ (series['value'][-1])
    


# In[39]:


# Sorting our 3 year ROI forecast for zipcodes into descending order
sort_orders = sorted(zip_predictions.items(), key=lambda x: x[1], reverse=True)

sorted_forecast_3yr = {}
for i in sort_orders:
    sorted_forecast_3yr[i[0]] = i[1]
sorted_forecast_3yr

#Selecting only the Top 5 Zips
top_5_zipcodes = list(sorted_forecast_3yr.items())[:5]
top_5_zipcodes

fig, ax = plt.subplots(figsize=(18,12))

x_labels = [top_5_zipcodes[0][0],top_5_zipcodes[1][0],top_5_zipcodes[2][0],
            top_5_zipcodes[3][0],top_5_zipcodes[4][0]]
x = [1,2,3,4,5]
y = [top_5_zipcodes[0][1],top_5_zipcodes[1][1],top_5_zipcodes[2][1],
     top_5_zipcodes[3][1],top_5_zipcodes[4][1]]

ax.bar(x, y, color='mediumslateblue')
remaining_zipcodes = list(sorted_forecast_3yr.items())[5:]
median_ROI_other_zips = remaining_zipcodes[5][1]
ax.axhline(y=median_ROI_other_zips,label = 'Median ROI for remaining Zips')
ax.set_xticks(x)
ax.set_xticklabels(x_labels)
ax.set_yticklabels([str(a)+'%' for a in list(range(0,75,5))])
ax.set_title('Top 5 most growing zip codes in Jacksonville', fontsize=20)
ax.set_ylabel('Average ROI', fontsize=20)
ax.set_xlabel('Zipcodes',fontsize=20)
ax.legend(prop={'size': 15});


# In[40]:


print(f"The Highest Growing Zipcode: {list(sorted_forecast_3yr.items())[:1][0][0]} is expected to grow by {round((list(sorted_forecast_3yr.items())[:1][0][1]) * 100, 2)}% over the next 3 years.")


# In[41]:


print(f'The Highest Growing Zipcode: {list(sorted_forecast_3yr.items())[0][0]} is expected to grow by {round((list(sorted_forecast_3yr.items())[0][1])*100,2)}%')
print(f'The Second Highest Growing Zipcode: {list(sorted_forecast_3yr.items())[1][0]} is expected to grow by {round((list(sorted_forecast_3yr.items())[1][1])*100,2)}%')
print(f'The Third Highest Growing Zipcode: {list(sorted_forecast_3yr.items())[2][0]} is expected to grow by {round((list(sorted_forecast_3yr.items())[2][1])*100,2)}%')
print(f'The Fourth Highest Growing Zipcode: {list(sorted_forecast_3yr.items())[3][0]} is expected to grow by {round((list(sorted_forecast_3yr.items())[3][1])*100,2)}%')
print(f'The Fifth Highest Growing Zipcode: {list(sorted_forecast_3yr.items())[4][0]} is expected to grow by {round((list(sorted_forecast_3yr.items())[4][1])*100,2)}%')


# In[ ]:





# In[ ]:





# In[ ]:




