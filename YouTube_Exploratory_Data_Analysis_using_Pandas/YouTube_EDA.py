#!/usr/bin/env python
# coding: utf-8

# # Data Cleaning and Exploratory Data Analysis with Pandas
# 
# For this project, I have looked at data from YouTube containing information about the published date, trending date, category, views, likes, dislikes and comment counts for trending videos in 10 countries. 
# 
# As a first step of the data analysis process, I have looked at the data and converted data types such that they are easily manipulated during analysis. After that I examined for any missing, N/A values. Once, that was done, I combined the different data frames from all the countries into one combined data frame. I then inserted a category column into the data frame by reading the categories from a JSON file. This concluded the **data cleaning** portion of the data analysis.
# 
# Once the data was cleaned, I performed **exploratory data analysis (EDA)** and answered questions like:
# 1. What was the ratio of likes and dislikes in different categories?
# 2. What was the category of the trending videos in each country?
# 3. What were the top 5 videos trending in each country?
# 4. Is the most liked video also the most trending video?
# 5. What was the number of days between publishing a video and when it became trending?
# 6. What were the most liked categories?
# 7. Which categories got the most comments?
# 8. What were the most frequently words occurring in tags and descriptions?
# 9. Were there any correlations between views, likes, dislikes and comments?

# ### Data Cleaning

# In[1]:


# importing the necessary libraries.
import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set()


# In[2]:


# Since there were different csv files for each country.
# Importing all the csv files.
import glob
all_csv = [i for i in glob.glob('*.{}'.format('csv'))]
all_csv


# In[3]:


# Reading all the csv files and putting them into individual dataframes.
all_dataframes = []
for csv in all_csv:
    df = pd.read_csv(csv, encoding='ISO-8859-1')
    df['country'] = csv[0:2]
    all_dataframes.append(df)
    


# In[4]:


# Looking at the data frame for one country: US.
all_dataframes[8].head()


# In[5]:


# Getting info on the data types of the columns.
all_dataframes[8].info()


# **Fixing the data type of the columns to make them easier to manipulate.**
# 
# 1. Converting video_id, title, channel_title, category_id, tags, thumbnail_link and description as str.
# 2. Converting comments_disabled, ratings_disabled, video_error_or_removed into category.
# 3. Converting trending date into date time.
# 4. Splitting publish_date into publish_date(datetime) and publish_time(str).

# In[6]:


for df in all_dataframes:
    # video_id 
    df['video_id'] = df['video_id'].astype('str') 
    
    # trending date
    df['trending_date'] = df['trending_date'].astype('str') 
    date_pieces = (df['trending_date']
                   .str.split('.')
                  )
    df['Year'] = date_pieces.str[0].astype(int)
    df['Day'] = date_pieces.str[1].astype(int)
    df['Month'] = date_pieces.str[2].astype(int)
    updatedyear = []
    for i in range(len(df)) : 
        y = df.loc[i, "Year"]
        newy = y+2000
        updatedyear.append(newy)
    for i in range(len(df)):
        newy = updatedyear[i]
        tr = df.loc[i, "Year"]
        df['Year'].replace(to_replace = tr, value = newy, inplace=True)
    del df['trending_date']
    df['trending_date'] = pd.to_datetime(df[['Year', 'Month', 'Day']], format = "%Y-%m-%d")
    del df['Year']
    del df['Day']
    del df['Month']
    
    #title
    df['title'] = df['title'].astype('str')
    #channel_title
    df['channel_title'] = df['channel_title'].astype('str')
    #category_id
    df['category_id'] = df['category_id'].astype(str) 
    
    #tags
    df['tags'] = df['tags'].astype('str')
    
    #thumbnail_link
    df['thumbnail_link'] = df['thumbnail_link'].astype('str') 
    
    #description
    df['description'] = df['description'].astype('str')
    
    # Changing comments_disabled, ratings_disabled, video_error_or_removed from bool to categorical
    df['comments_disabled'] = df['comments_disabled'].astype('category') 
    df['ratings_disabled'] = df['ratings_disabled'].astype('category') 
    df['video_error_or_removed'] = df['video_error_or_removed'].astype('category') 
    
    # publish_time 
    df['publish_time'] = pd.to_datetime(df['publish_time'], errors='coerce', format='%Y-%m-%dT%H:%M:%S.%fZ')


# In[ ]:





# In[7]:


# Separating publish time into publish date and publish time.
for df in all_dataframes:
    df.insert(4, 'publish_date', df['publish_time'].dt.date)
    df['publish_time'] = df['publish_time'].dt.time
    
# Changing data type for 'publish_date' from object to 'datetime64[ns]'
for df in all_dataframes:
     df['publish_date'] = pd.to_datetime(df['publish_date'], format = "%Y-%m-%d")


# In[8]:


# Now checking that the changes to data types has taken place.
all_dataframes[1].dtypes


# In[9]:


# Looking at the data.
all_dataframes[8].head()


# In[10]:


# Setting the index of the data frame as video_id which is unique.
for df in all_dataframes:
    df.set_index('video_id', inplace=True)


# In[11]:


# Checking for any null values via a heatmap.
for df in all_dataframes:
    sns.heatmap(df.isnull(), cbar=False)
    plt.figure()


# In[12]:


# checking for any na values via heatmap.
for df in all_dataframes:
    sns.heatmap(df.isna(), cbar=False)
    plt.figure()


# We do not have any missing or N/A values in our data. We checked this using heatmap. **Any missing value in a column would appears as an orange square against the black background of the heat-map**. We do not see any.

# In[13]:


# combining the individual data frames into one data frame.
combined_df = pd.concat(all_dataframes)


# Next, the data was further cleaned by **sorting the entries of the data set by trending_date**. This would result in the latest trending videos to be moved to the top of the data set. This was done so that we can view the current trends of the trending videos of each country, as they are more relevant to our project.
# 
# I created a duplicate copy of the data frame as a safety precaution and to keep a copy of the original data frame at hand. **Duplicate video entries were removed** while sorting the videos from the other data frame.

# In[14]:


backup_df = combined_df.reset_index().sort_values('trending_date', ascending=False).set_index('video_id')

combined_df = combined_df.reset_index().sort_values('trending_date', ascending=False).drop_duplicates('video_id',keep='first').set_index('video_id')

for df in all_dataframes:
    df = df.reset_index().sort_values('trending_date', ascending=False).set_index('video_id')

combined_df[['publish_date','publish_time','trending_date', 'country']].head()


# In[15]:


# Reading the JSON file to see if it contains any useful information.
import json

with open('US_category_id.json', 'r') as f:
    data = f.read()

obj = json.loads(data)
obj


# In[16]:


# mapping the category number to category name and putting it into the data frame.
category_id = {}
with open('US_category_id.json', 'r') as f:
    d = json.load(f)
    for category in d['items']:
        category_id[category['id']] = category['snippet']['title']

combined_df.insert(2, 'category', combined_df['category_id'].map(category_id))
backup_df.insert(2, 'category', backup_df['category_id'].map(category_id))

for df in all_dataframes:
    df.insert(2, 'category', df['category_id'].map(category_id))

combined_df.head()


# In[17]:


# Seeing how many unique categories there are.
combined_df['category'].unique()


# ### Exploratory Data Analysis.

# #### What was the ratio of likes and dislikes in different categories?

# In[18]:


likesdf = combined_df.groupby('category')['likes'].agg('sum')
dislikesdf = combined_df.groupby('category')['dislikes'].agg('sum')
ratiodf = likesdf/dislikesdf 
ratiodf = ratiodf.sort_values(ascending=False).reset_index()

ratiodf.columns = ['category','ratio']
plt.subplots(figsize=(10, 15))
sns.barplot(x="ratio", y="category", data=ratiodf,
            label="Likes-Dislikes Ratio", color="b")


# We see that videos belonging to the pets and animals categories have the highest ratio of likes to dislikes videos among the trending categories whereas new and politics videos have the least. This implies that **people are less divided on the content of videos based on pets and animals** than compared to topics such as news, whose content can lead to a division of opinions among the user.

# #### What was the category of the trending videos in each country?

# In[ ]:





# In[20]:


countries = []
allcsv = [i for i in glob.glob('*.{}'.format('csv'))]
for csv in allcsv:
    c = csv[0:2]
    countries.append(c)
    
for country in countries:
        tempdf = combined_df[combined_df['country']==country]['category'].value_counts().reset_index()
        ax = sns.barplot(y=tempdf['index'], x=tempdf['category'], data=tempdf, orient='h')
        plt.xlabel("Number of Videos")
        plt.ylabel("Categories")
        plt.title("Catogories of trend videos in " + country)
        plt.figure()


# Category most liked by the users in each of the other countries is ‘Entertainment’, apart from Russia and Great Britain.
# **Viewers from Russia prefer the category ‘People and Blogs’ the most.
# Viewers from Great Britain prefer the category ‘Music’ the most.**

# #### What were the top 5 videos trending in each country?

# In[21]:


temporary = []
for df in all_dataframes:
    temp = df
    temp = temp.reset_index().sort_values(by = ['views'], ascending=False)
    temp.drop_duplicates(subset ="video_id", keep = 'first', inplace = True)
    temp.set_index('video_id', inplace=True)
    temp = temp.head(5) # top 5 that are on trending
    temporary.append(temp)

# printing it for one random country.
temporary[6][['title', 'channel_title', 'category', 'views', 'likes']]


# **Music and Entertainment were the most popular categories for trending videos.**

# #### Is the most liked video also the most trending video?

# In[22]:


temporary = [] 
for df in all_dataframes:
    temp = df
    temp = temp.reset_index().sort_values(by = ['likes'], ascending=False)
    temp.drop_duplicates(subset ="video_id", keep = 'first', inplace = True)
    temp.set_index('video_id', inplace=True)
    temp = temp.head(5) # top 5 that are most liked
    temporary.append(temp)

# Printing a radom result.
temporary[1][['views', 'likes']]


# Most viewed video doesn't necessarily mean the most liked video. See rows 4 and 5.

# #### What was the number of days between publishing a video and when it became trending?

# In[23]:


# Calculating days between publish and trending date
temporary = []
for data in all_dataframes:
    temp = data
    temp['timespan'] = (temp['trending_date'] - temp['publish_date']).dt.days
    temporary.append(temp)

# Plotting
to_trending = temporary[0].sample(1000).groupby('video_id').timespan.max() # CA
sns_ax = sns.boxplot(y = to_trending)
_ = sns_ax.set(yscale = "log")
plt.show()
_ = sns.distplot(to_trending.value_counts(),bins='rice',kde=False)


# **Most videos take less a 100 days to reach the trending page.**

# #### What were the most liked categories?

# In[24]:


temp = combined_df
temp = temp.groupby('category')['views', 'likes'].apply(lambda x: x.astype(int).sum())
temp = temp.sort_values(by='likes', ascending=False).head()
temp


# #### Which categories got the most comments?

# In[25]:


temp = combined_df
temp = temp.groupby('category')['views','likes', 'comment_count'].apply(lambda x: x.astype(int).sum())
temp = temp.sort_values(by='comment_count', ascending=False).head()
temp


# #### What were the most frequently words occurring in tags and descriptions?

# In[26]:


#! pip3 install wordcloud


# In[ ]:


from wordcloud import WordCloud

plt.figure(figsize=(15, 15))
stopwords = set(STOPWORDS)
wordcloud = WordCloud(background_color='black', stopwords=stopwords, max_words=1000,
                      max_font_size=120, random_state=42).generate(str(combined_df['tags']))
plt.imshow(wordcloud)
plt.title('WORD CLOUD for Tags', fontsize=20)
plt.axis('off')
plt.show()


# ![Screenshot%202023-07-31%20at%2010.02.31%20AM.png](attachment:Screenshot%202023-07-31%20at%2010.02.31%20AM.png)

# In[ ]:


plt.figure(figsize = (15, 15))
stopwords = set(STOPWORDS)
wordcloud = WordCloud(
 background_color = 'black',
 stopwords = stopwords,
 max_words = 1000,
 max_font_size = 120,
 random_state = 42
 ).generate(str(combined_df['description']))
plt.imshow(wordcloud)
plt.title('WORD CLOUD for Description', fontsize = 20)
plt.axis('off')
plt.show()


# ![Screenshot%202023-07-31%20at%2010.02.39%20AM.png](attachment:Screenshot%202023-07-31%20at%2010.02.39%20AM.png)

# #### Were there any correlations between views, likes, dislikes and comments?

# In[27]:


col = ['views', 'likes', 'dislikes', 'comment_count']
corr = combined_df[col].corr()
corr


# In[28]:


sns.heatmap(corr, cmap = 'viridis')


# **There is a positive relation between views and likes, likes and comment_count, dislikes and comment_count.**

# **End of the project.**

# In[ ]:




