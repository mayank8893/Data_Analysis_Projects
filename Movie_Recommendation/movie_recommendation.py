#!/usr/bin/env python
# coding: utf-8

# # Movie Recommendation
# 
# For this project, I will look at the movie dataset from https://files.grouplens.org/datasets/movielens/ml-25m.zip. This dataset contains two main files: **movies** with information on title, genre and movie id of the movie and the **ratings** file which contains information about rating, user id and movie id. First I will design a **search engine using tfidf** which gives the 5 most similar movie names to what you have input. 
# 
# Then I have designed a **movie recommendation engine**, where:
# 1. I have found users who liked the same movie as us (similar users).
# 2. Combined a list of other movies liked by these users.
# 3. Made a list of movies that were liked by 10% or more of similar users.
# 4. Found out how much all users like these movies.
# 5. Created a score by **calculating the differential** between movies liked by similar users and by all users.
# 6. Used this score to **recommend 10 movies** that you could watch.
# 
# So just plug in a movie and get a recommendation to watch your next movie tonight.

# ### Search Engine

# In[29]:


# Reading the data

import pandas as pd

movies = pd.read_csv("movies.csv")


# In[30]:


movies


# In[3]:


# defining a function to clean the title and remove all special charachters.
import re

def clean_title(title):
    return re.sub("[^a-zA-Z0-9 ]", "", title)


# In[31]:


# cleaning the title.
movies["clean_title"] = movies["title"].apply(clean_title)
movies


# movies

# In[ ]:





# In[6]:


# vectorizing the data using tf-idf

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(ngram_range=(1,2))

tfidf = vectorizer.fit_transform(movies["clean_title"])


# In[7]:


# creating the search engine

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def search(title):
    title = clean_title(title)
    query_vec = vectorizer.transform([title])
    similarity = cosine_similarity(query_vec, tfidf).flatten()
    indices = np.argpartition(similarity, -5)[-5:]
    results = movies.iloc[indices][::-1]
    return results


# In[8]:


# displaying the search engine
import ipywidgets as widgets
from IPython.display import display

movie_input = widgets.Text(
    value = "Toy Story",
    description = "Movie Title:",
    disabled = False
)
movie_list = widgets.Output()

def on_type(data):
    with movie_list:
        movie_list.clear_output()
        title = data["new"]
        if(len(title) > 5):
            display(search(title))
            
movie_input.observe(on_type, names = "value")

display(movie_input, movie_list)


# ### Movie Recommendation engine

# In[32]:


# reading the ratings file.
ratings = pd.read_csv("ratings.csv")


# In[33]:


ratings


# In[11]:


ratings.dtypes


# In[12]:


movie_id = 1


# In[13]:


# Finding similar users.
# people who watched the same movie and rated it 4 or greater.
similar_users = ratings[(ratings["movieId"] == movie_id) & (ratings["rating"] > 4)] ["userId"].unique()


# In[14]:


similar_users


# In[34]:


# Finding movies liked by these similar users.
# Movies to which they gave greater than 4 ratings.
similar_user_recs = ratings[(ratings["userId"].isin(similar_users)) & (ratings["rating"] > 4)]["movieId"]


# In[16]:


similar_user_recs


# In[17]:


# converting the number of people liking a movie to percentage.
# only selecting movies that were liked by 10% or more of similar users. 
similar_user_recs =  similar_user_recs.value_counts()/len(similar_users)

similar_user_recs = similar_user_recs[similar_user_recs > 0.1]


# In[18]:


similar_user_recs


# In[19]:


# Checking how many of all users liked a particular movie.
all_users = ratings[(ratings["movieId"].isin(similar_user_recs.index)) & (ratings["rating"] > 4)]


# In[20]:


# converting it to percentage.
all_users_recs = all_users["movieId"].value_counts() / len(all_users["userId"].unique( ))


# In[21]:


all_users_recs


# In[22]:


# combing the percentage of similar users liking a movie and all users liking a movie.
rec_percentages = pd.concat([similar_user_recs, all_users_recs], axis = 1)
rec_percentages.columns = ["similar", "all"]


# In[23]:


rec_percentages


# In[24]:


# Created a score as the ration of the two percentages.
rec_percentages["score"] = rec_percentages["similar"]/rec_percentages["all"]
rec_percentages = rec_percentages.sort_values("score", ascending=False)


# In[25]:


rec_percentages


# In[26]:


# listing the top 10 movies based on the above score.
rec_percentages.head(10).merge(movies, left_index=True, right_on="movieId")


# In[27]:


# putting everything in a function.
def find_similar_movies(movie_id):
    similar_users = ratings[(ratings["movieId"] == movie_id) & (ratings["rating"] > 4)] ["userId"].unique()
    similar_user_recs = ratings[(ratings["userId"].isin(similar_users)) & (ratings["rating"] > 4)]["movieId"]
    
    similar_user_recs =  similar_user_recs.value_counts()/len(similar_users)
    similar_user_recs = similar_user_recs[similar_user_recs > 0.1]
    
    all_users = ratings[(ratings["movieId"].isin(similar_user_recs.index)) & (ratings["rating"] > 4)]
    all_users_recs = all_users["movieId"].value_counts() / len(all_users["userId"].unique( ))
    
    rec_percentages = pd.concat([similar_user_recs, all_users_recs], axis = 1)
    rec_percentages.columns = ["similar", "all"]
    
    rec_percentages["score"] = rec_percentages["similar"]/rec_percentages["all"]
    rec_percentages = rec_percentages.sort_values("score", ascending=False)
    
    return rec_percentages.head(10).merge(movies, left_index=True, right_on="movieId")[["score", "title", "genres"]]


# In[ ]:





# In[28]:


# displaying the widget.
# plug in a movie and enjoy your next movie based on the recommendation.
movie_name_input = widgets.Text(
    value = "Toy Story",
    description = "Movie Title:",
    disabled = False
)
recommendation_list = widgets.Output()

def on_type(data):
    with recommendation_list:
        recommendation_list.clear_output()
        title = data["new"]
        if(len(title) > 5):
            results = search(title)
            movie_id = results.iloc[0]["movieId"]
            display(find_similar_movies(movie_id))
            
movie_name_input.observe(on_type, names = "value")

display(movie_name_input, recommendation_list)


# In[ ]:




