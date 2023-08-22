#!/usr/bin/env python
# coding: utf-8

# # US Fatal Police Encounters
# 
# In this project, I am looking at the dataset from Kaggle on Fatal US Police encounters. I will first clean the data and then answer a few questions:
# 
# 1. What is the distribution of Fatal encounters with respect to age, gender, mental illness and race? 
# 2. Are black people disproportionately killed?
# 3. Where are shootings happening?
# 4. Are Police Shooting deaths increasing?
# 5. Fatal Police encounters year over year for each race.
# 6. Exploratory analysis on other factors of fatal encounters.
# 7. City adistribution of fatal encounters.
# 8. Race distribution of deaths inside these top cities.

# ### Loading the data and cleaning it.

# In[1]:


# importing libraries.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


# In[2]:


# Loading the datasets
data = pd.read_csv('fatal-police-shootings-data.csv')


# In[3]:


# checking the data shape.
data.shape


# In[4]:


# looking at data types and null values.
data.info()


# In[5]:


# checking if any columns have null values.
data.isnull().sum()


# In[6]:


# converting the object datatype columns to "not specified".
data[["armed", "gender", "race", "flee"]] = data[["armed", "gender", "race", "flee"]].fillna(value="not specified")


# In[7]:


# checking that the changes have taken place.
data.isna().sum()


# In[8]:


#dropping the null values in the ages column.
data.dropna(inplace=True)


# In[9]:


# checking the shape again.
data.shape


# In[10]:


# checking for any duplicate values.
data.duplicated().any()


# In[11]:


# converting the date column into a datetime object.
data["date"] = data["date"].apply(pd.to_datetime)


# ### 1. What is the distribution of Fatal encounters with respect to age, gender, mental illness and race? 

# In[12]:


fig, ax = plt.subplots(2,2, figsize = (12, 8))

sns.countplot(x = 'race', ax = ax[0, 0], data = data)
sns.countplot(x = 'gender', ax = ax[0, 1], data = data)
sns.countplot(x = 'signs_of_mental_illness', ax = ax[1, 0], data = data)
sns.countplot(x = 'age', ax = ax[1,1], data = data)
ax[1,1].set_xticks(range(0,90,10))
ax[1,1].set_xticklabels(range(0,90,10))


# Insights:
# 1. You are less likely to have a fatal police encounter as you get older (60 and above).
# 2. Men are more likely to have fatal encounters. A point could be made that men are driving more but I dont think thats true or atleast not this disproportionately(need to investigate).
# 3. White americans die more in Police encounters.

# ### 2. Are black people disproportionately killed?
# 
# An essential aspect of societal relevance pertains to the potential disproportionate occurrences of police shootings among various racial groups. Our exploration involves generating a bar plot using Seaborn to illustrate the ratio of fatalities to the total population across different races. Subsequently, we will enhance the visualization using Matplotlib. Additionally, we will import another dataset containing information about the racial composition within the US population.

# In[15]:


# Get data
us_census_data = pd.read_csv('acs2017_census_tract_data.csv')

# Get population proportions
total_population = us_census_data['TotalPop'].sum()
race_proportions = pd.DataFrame(['White', 'Hispanic', 'Black', 'Asian', 'Native'], columns=['Race'])
race_proportions['Population'] = race_proportions['Race'].apply(lambda x: us_census_data.apply(lambda y: y['TotalPop'] * y[x] / total_population, axis=1).sum())
race_proportions['Killed In Police Shootings'] = race_proportions['Race'].apply(lambda x: data[data['race'] == x[0]].shape[0] * 100 / data.shape[0])

# Plot proportions
race_proportions = race_proportions.melt(id_vars='Race')
fig, ax = plt.subplots(1, 1, figsize=(10,6))
sns.barplot(x='value', y='Race', hue='variable',data=race_proportions, ax=ax,
            orient='h')

# Annotate with values
for p in ax.patches:
    width = p.get_width()
    plt.text(3+p.get_width(), p.get_y()+0.55*p.get_height(),
             '{:1.2f}%'.format(width),
             ha='center', va='center')

# Customise and show
ax.set_title('Percentage of deaths from police shootings\ncompared to percentage of population by race', fontsize=16)
ax.tick_params(axis='both', labelsize=12)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_xticks([])
plt.legend(frameon=False, fontsize=12, ncol=2)
plt.tight_layout()
plt.show()


# Examining the graph reveals a notable disparity in police-related fatalities, particularly affecting both black and native individuals.

# ### 3. Where are shootings happening?
# 
# Next, we will employ Plotly to construct a choropleth map showcasing the frequency of shootings across different US states. Our visualization illustrates that California, Texas, and Florida stand out as the states with the highest incidence of shootings. Additionally, states neighboring Texas (like Oklahoma) and Florida (like Georgia) exhibit an elevated number of shootings compared to other states in the US.

# In[16]:


import plotly.graph_objects as go
state_counts = data.groupby(by='state').agg({'id' : 'count'}).reset_index()

fig = go.Figure(data=go.Choropleth(
    locations=state_counts['state'],
    z = state_counts['id'],
    locationmode = 'USA-states',
    colorscale = 'Reds',
    colorbar_title = "Deaths"
))

fig.update_layout(
    title_text = 'Police Shooting Deaths by US States',
    geo_scope='usa'
)

fig.show()


# The initial choropleth map we created does not factor in the population of each state, which is crucial since **states with larger populations would logically have a higher incidence of police shooting deaths.** As a result, we will incorporate a dataset containing the 2018 population figures for US states and integrate this information into our choropleth visualization. Upon making this adjustment, you will notice a notable transformation in the appearance of our updated map.

# In[18]:


state_pops = pd.read_csv('State Populations.csv')
state_codes = {'California' : 'CA', 'Texas' : 'TX', 'Florida' : 'FL', 'New York' : 'NY', 'Pennsylvania' : 'PA',
       'Illinois' : 'IL', 'Ohio' : 'OH', 'Georgia' : 'GA', 'North Carolina' : 'NC', 'Michigan' : 'MI',
       'New Jersey' : 'NJ', 'Virginia' : 'VA', 'Washington' : 'WA', 'Arizona' : 'AZ', 'Massachusetts' : 'MA',
       'Tennessee' : 'TN', 'Indiana' : 'IN', 'Missouri' : 'MO', 'Maryland' : 'MD', 'Wisconsin' : 'WI',
       'Colorado' : 'CO', 'Minnesota' : 'MN', 'South Carolina' : 'SC', 'Alabama' : 'AL', 'Louisiana' : 'LA',
       'Kentucky' : 'KY', 'Oregon' : 'OR', 'Oklahoma' : 'OK', 'Connecticut' : 'CT', 'Iowa' : 'IA', 'Utah' : 'UT',
       'Nevada' : 'NV', 'Arkansas' : 'AR', 'Mississippi' : 'MS', 'Kansas' : 'KS', 'New Mexico' : 'NM',
       'Nebraska' : 'NE', 'West Virginia' : 'WV', 'Idaho' : 'ID', 'Hawaii' : 'HI', 'New Hampshire' : 'NH',
       'Maine' : 'ME', 'Montana' : 'MT', 'Rhode Island' : 'RI', 'Delaware' : 'DE', 'South Dakota' : 'SD',
       'North Dakota' : 'ND', 'Alaska' : 'AK', 'District of Columbia' : 'DC', 'Vermont' : 'VT',
       'Wyoming' : 'WY'}
state_pops['State Codes'] = state_pops['State'].apply(lambda x: state_codes[x])
state_counts['Pop'] = state_counts['state'].apply(lambda x: state_pops[state_pops['State Codes'] == x].reset_index()['2018 Population'][0])

fig = go.Figure(data=go.Choropleth(
    locations=state_counts['state'],
    z = state_counts['id'] / state_counts['Pop'] * 100000,
    locationmode = 'USA-states',
    colorscale = 'Reds',
    colorbar_title = "Deaths Per 100,000"
))

fig.update_layout(
    title_text = 'Police Shooting Deaths by US States per 100,000 People',
    geo_scope='usa'
)

fig.show()


# Upon considering population size, a different perspective emerges. Notably, **New Mexico and Alaska** emerge as the states with the highest number of police shooting deaths. While several states that ranked prominently on the prior map exhibit lower figures here, the prominence of Oklahoma and Arizona in terms of police shooting incidents remains evident.

# ### 4. Are Police Shooting deaths increasing?
# Let's now turn our attention to assessing the evolving frequency of police shooting deaths. For this analysis, we will once again employ Seaborn, utilizing the regplot function to generate a regression line that encapsulates our data. To facilitate this examination, we will categorize the data into monthly groups. Upon reviewing the resulting visualization, it becomes apparent that there is no significant alteration in the number of shootings per month over the observed period.

# In[20]:


# Get date month data
data['date'] = pd.to_datetime(data['date'])
newd = data.groupby(pd.Grouper(key='date', freq='M')).count().reset_index()[['date', 'id']]
newd['date_ordinal'] = newd['date'].apply(lambda x: x.toordinal())

# Plot
fig, ax = plt.subplots(1, 1, figsize=(12, 4))
sns.regplot(x='date_ordinal', y='id', ax=ax, data=newd)

# Customise
year_labels = [newd['date_ordinal'].min() + (x * 365) for x in range(6)]
ax.set_xticks(year_labels)
ax.set_xticklabels([2015, 2016, 2017, 2018, 2019, 2020])
ax.set_xlabel('Year')
ax.set_ylabel('Deaths')

plt.title('US Police Shooting Deaths Over Time', fontsize=16)
plt.show()


# ### 5. Fatal Police encounters year over year for each race.
# 

# In[23]:


data["year"] = data["date"].apply(lambda x: x.year)
time_group = data.groupby(["year", "race"], as_index=False).agg({"id": pd.Series.count})


# In[24]:


import plotly.express as px
time_g = px.line(x=time_group.year, y=time_group.id, color=time_group.race, title="Number of killed people over time with race distribution")
time_g.update_layout(xaxis_title="year", yaxis_title="Number of death")
time_g.show()


# ### 6. Exploratory analysis on other factors of fatal encounters.

# In[25]:


sns.set_style('whitegrid')

plt.subplots(3, 2, figsize=(13, 10))
cols = ['manner_of_death', 'flee', 'body_camera', 'gender', 'signs_of_mental_illness', 'threat_level']
for i in range(len(cols)):
    plt.subplot(3, 2, i + 1)
    
    # Calculate percentage values
    percentage_values = data[cols[i]].value_counts(normalize=True) * 100
    
    # Plot the barplot with percentage values
    sns.barplot(x=percentage_values.index, y=percentage_values.values, palette='bright')
    
    plt.ylabel('Percent')
    
    if len(percentage_values) >= 3:
        plt.xticks(rotation=75)
    
    plt.title(cols[i])  # Add title to the subplot

plt.tight_layout()

# Display the plots
plt.show()


# ### 7. City adistribution of fatal encounters.

# In[26]:


city = data["city"].value_counts()
city = city.sort_values(ascending=False)


# In[27]:


cit = go.Figure(data=[go.Table(
    header=dict(values=["City", "Count"], fill_color="lavender", font=dict(size=14, color='black')),
    cells=dict(values=[city.index[:30], city.values[:30]], fill_color="#F5F5DC"))
                     ])
cit.update_layout(title_text="Top 30 cities with the number of death")
cit.show()


# In[29]:


ci = px.bar(x=city.index[:10], y=city.values[:10], title="Top 10 cities with the highest deaths")
ci.update_layout(xaxis_title="City", yaxis_title="Number of death")
ci.show()


# ### 8. Race distribution of deaths inside these top cities.

# In[30]:


race_city = data.groupby(["city", "race"])["name"].count().reset_index()
race_city = race_city.sort_values("name", ascending=False)
race_city.head()


# In[32]:


rc = px.bar(race_city[:20], x="city", y="name", color="race", title="Share of each race in the top 20 cities")
rc.update_layout(yaxis_title="Number of death")
rc.show()


# We see that out in the west, there are more fatal encounters with hispanic people. Wherewas in the east, the fatal encounters is disproportionaly with black people. Need to investigate the population distribution in these cities. Logically, if there are only asian people in a city then obviously any fatal encounter in this city will be with an asian person.

# In[ ]:




