#!/usr/bin/env python
# coding: utf-8

# # US Baby Names 1880-2022

# The United States Social Security Administration has a database with the frequency of all baby names from 1880-2022. I am going to analyze this data to find out any interesting pattern in the data. In particular, I want to do the following with the dataset:
# 
# -  Visualize the proportion of babies given a particular name over time.
# -  Determine the relative rank of a name.
# -  Determine the names whose popularity has advanced or declined the most.
# -  Determine the most popular names in each year.
# -  Analyze trends in names: vowels, consonants, overall diversity, changes in spelling, first and last letters.
# -  Analyze external sources of trends: biblical names, celebrities, demographic changes.
# 
# The dataset was downloaded from:  http://www.ssa.gov/oact/babynames/limits.html

# There were individual files like yob1880.txt for each year. The first step therefore was to load the dataset for each year and put it into a dataframe.

# In[21]:


import pandas as pd

years = range(1880, 2023)
    
pieces = []
columns = ['name', 'sex', 'births']

for year in years:
    path = 'baby_names_data/yob%d.txt' % year 
    frame = pd.read_csv(path, names=columns)
        
    frame['year'] = year
    pieces.append(frame)
    
# Concatenate everything into a single DataFrame
names = pd.concat(pieces, ignore_index=True)


# In[22]:


names


# ### Total births by sex and year.

# We can aggregate the data at the year and sex level using groupby or pivot_table and then plot it.

# In[23]:


total_births = names.pivot_table('births', index='year', columns='sex', aggfunc=sum)
total_births.tail()


# In[24]:


get_ipython().run_line_magic('matplotlib', 'notebook')
total_births.plot(title='Total births by sex and year')


# There is a very interesting trend here where the number of baby girls were always higher from 1880 to mid 1930s. Beyond that the number of baby boys have always been higher. Why did this shift happen?

# ### Proportion of baby names.
# 
# I will now insert a column prop with a fraction of babies given each name relative to the total number of births.

# In[25]:


def add_prop(group):
    group['prop'] = group.births / group.births.sum() 
    return group

names = names.groupby(['year', 'sex']).apply(add_prop)
names.head()


# In[26]:


## doing a sanity check to make sure that the sum of all proportions equals 1 as it should.
names.groupby(['year', 'sex']).prop.sum()


# Extracting a subset of data: the top 1000 names for each sex/year combination.

# In[27]:


pieces = []
for year, group in names.groupby(['year', 'sex']):
    pieces.append(group.sort_values(by='births', ascending=False)[:1000])
    
top1000 = pd.concat(pieces, ignore_index=True)
top1000.head()


# In[28]:


sorted_top1000 = top1000.sort_values(by=['year', 'prop'], ascending=[True, False])
most_popular_names_df = sorted_top1000.groupby('year').first().reset_index()
most_popular_names_df


# In[29]:


import matplotlib.pyplot as plt

selected_years = most_popular_names_df[most_popular_names_df['year'].isin([1880, 1919, 1946, 1971, 2000, 2016])]

pivot_table = selected_years.pivot(index='year', columns='name', values='prop')

ax = pivot_table.plot(kind='bar', colormap='Paired', figsize=(10, 6))
plt.title("Most Popular Names for the Selected Years")
plt.xlabel("Names")
plt.ylabel("Proportion")
plt.xticks(rotation=45)
ax.set_xticklabels(pivot_table.index)
ax.set_xlabel("Names")
ax.set_ylabel("Proportion")
plt.tight_layout()


# ### Analyzing Naming Trends

# In[30]:


## Splitting the top 1000 names into boy and girl first.
boys = top1000[top1000.sex == 'M']
girls = top1000[top1000.sex == 'F']


# In[31]:


## Making a pivot table of the total number of births by year and name.
total_births = top1000.pivot_table('births', index='year',
                                  columns='name', aggfunc=sum)
total_births.info()


# In[32]:


## plotting a subset of data. The number of John, Harry, Mary and Marilyn names for each year.
subset = total_births[['John', 'Harry', 'Mary', 'Marilyn']]
subset.plot(subplots=True, figsize=(12, 10), grid=False, title="Number of births per year")


# #### By looking at the above plot you might conclude that these names have grown out of favor, but the story is a bit more complicated as will be explored now.

# ### Measuring the increase in naming diversity.
# 
# One explanation for the decrease is that fewer parents are choosing common names for their children. This hypothesis can be explored and confirmed by the data. One measure is the proportion of births represented by the top1000 popular names, aggregated by year and sex.

# In[33]:


import numpy as np
table = top1000.pivot_table('prop', index='year', columns='sex', aggfunc=sum)

table.plot(title='Sum of table1000.prop by year and sex', 
           yticks=np.linspace(0, 1.2, 13), xticks=range(1880, 2030, 15))


# ### Number of names to get to 50% of births.
# There is decreasing total proportion in the top 1000 implying increasing name diversity. Another interesting metric is the number of distinct names, taken in order of popularity from highest to lowest in the top 50% of births.

# In[34]:


df = boys[boys.year == 2010]
df


# After sorting prop in descending order, we want to know how many of the most popular names it takes to reach 50%.

# In[35]:


prop_cumsum = df.sort_values(by='prop', ascending=False).prop.cumsum()
prop_cumsum.values.searchsorted(0.5)


# In[36]:


## in 1990 this number was much smaller.
df = boys[boys.year == 1900]
in1900 = df.sort_values(by='prop', ascending=False).prop.cumsum()
in1900.values.searchsorted(0.5) + 1


# In[37]:


## Now we can do this operation for each year and show it in a plot.
def get_quantile_count(group, q=0.5):
    group = group.sort_values(by='prop', ascending=False) 
    return group.prop.cumsum().values.searchsorted(q) + 1
    
diversity = top1000.groupby(['year', 'sex']).apply(get_quantile_count)
diversity = diversity.unstack('sex')
diversity.head()


# In[38]:


diversity.plot(title="Number of popular names in top 50%")


# #### Girl names have always been more diverse than boy names, and they have become increasingly more so over time. Also, the number of names it takes to get to 50% has increased dramatically implying that people are naming their kids more diverse names.

# ### The last letter revolution.
# 
# In 2007, baby name researcher Laura Wattenberg pointed out on her website that the distribution of boy names by final letter has changed significantly over the last 100 years.

# In[39]:


# extract last letter from name column
get_last_letter = lambda x: x[-1] 
last_letters = names.name.map(get_last_letter) 
last_letters.name = 'last_letter'
    
table = names.pivot_table('births', index=last_letters,
                              columns=['sex', 'year'], aggfunc=sum)


# In[40]:


subtable = table.reindex(columns=[1910, 1960, 2010], level='year')
subtable.head()


# In[41]:


subtable.sum()


# In[42]:


letter_prop = subtable / subtable.sum()
letter_prop


# In[43]:


## With the letter proportion now in hand, I can make a bar plot for each sex broken down by year.
import matplotlib.pyplot as plt
    
fig, axes = plt.subplots(2, 1, figsize=(10, 8))
letter_prop['M'].plot(kind='bar', rot=0, ax=axes[0], title='Male')
letter_prop['F'].plot(kind='bar', rot=0, ax=axes[1], title='Female', legend=False)


# #### Boy names ending in n have experienced significant growth since the 1960s.

# ### Letter proportional of boy names.

# In[44]:


letter_prop = table / table.sum()
dny_ts = letter_prop.loc[['d', 'n', 'y'], 'M'].T
dny_ts.head()


# In[46]:


dny_ts.plot()


# ### Boy names that became girl names.
# 
# One such example is Lesley or Leslie.

# In[47]:


## Get the count of all "Leslie" like names.
all_names = pd.Series(top1000.name.unique())
lesley_like = all_names[all_names.str.lower().str.contains('lesl')]
lesley_like


# In[48]:


filtered = top1000[top1000.name.isin(lesley_like)]
filtered.groupby('name').births.sum()


# In[49]:


## Aggregate by sex and year and normalize withing year.
table = filtered.pivot_table('births', index='year', columns='sex', aggfunc='sum')
table = table.div(table.sum(1), axis=0)
table.tail()


# In[50]:


## Now plotting a breakdown by sex over time.
table.plot(style={'M': 'k-', 'F': 'k--'})


# #### As you can see, Leslie started becoming a predominantly Female name in the 1940s.

# In[ ]:




