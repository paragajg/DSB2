
# coding: utf-8

# ## Recommendation Engine: Introduction to Building Recommendation Engine

# ### Import the libraries

# In[1]:


import numpy as np
import pandas as pd


# ### Get the data

# In[2]:


column_names = ['user_id', 'item_id', 'rating', 'timestamp']


# In[3]:


df = pd.read_csv('u.data', sep='\t', names=column_names)


# In[ ]:


df.head()


# In[5]:


movie_titles = pd.read_csv('Movie_Id_Titles')


# In[ ]:


movie_titles.head()


# In[ ]:


print("Total number of Movies in the database is %s"%len(movie_titles))


# Merge them together:

# In[7]:


df = pd.merge(df, movie_titles, on='item_id')


# In[ ]:


print(df.head())
print("\nSize of data set is {}".format(df.shape))


# ### Import vizualisation libraries

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


sns.set_style('white')


# In[ ]:


df.groupby('title')['rating'].mean().sort_values(ascending=False).head(10)


# In[ ]:


df.groupby('title')['rating'].count().sort_values(ascending=False).head(10)


# ##### create a ratings dataframe with average rating and number of ratings:

# In[ ]:


ratings =pd.DataFrame(df.groupby('title')['rating'].mean())


# In[ ]:


ratings.head()


# #####  Set the number of ratings column:

# In[ ]:


ratings['rating_numbers'] = pd.DataFrame(df.groupby('title')['rating'].count())


# In[ ]:


ratings.head()


# ##### Number of ratings histogram

# In[ ]:


ratings['rating_numbers'].hist(bins=70)


# #### Average rating per movie histogram

# In[ ]:


ratings['rating'].hist(bins=70)


# ##### Relationship between the average rating and the actual number of ratings
# ###### The larger the number of ratings, the more likely the rating of a movie is

# In[ ]:


sns.jointplot(x='rating', y='rating_numbers', data=ratings, alpha=0.5)


# ## Recommending Similar Movies using Collaborative Filtering

# Let's create a matrix that has the user ids on one access and the movie title on another axis. Each cell will then consist of the rating the user gave to that movie. The NaN values are due to most people not having seen most of the movies.

# In[ ]:


moviemat = df.pivot_table(index='user_id', columns='title', values='rating')
moviemat.head()


# ##### Most rated movies

# In[ ]:


ratings.sort_values('rating_numbers', ascending=False).head(10)


# #### Let's choose two movies for our system: Starwars, a sci-fi movie. And Liar Liar, a comedy.

# What are the user ratings for those two movies?

# In[ ]:


def recommendMovies(name , min_rating_count = 50):
    user_rating = moviemat[name]
    similar_movies = moviemat.corrwith(user_rating)
    corr_movies = pd.DataFrame(similar_movies, columns=['Correlation'])
    corr_movies.dropna(inplace=True)
    #corr_movies.sort_values('Correlation', ascending=False).head(10)
    # Joining the rating_number list so that we can filter basis minimum count of ratings to be considered
    # for recommending a movie
    corr_movies = corr_movies.join(ratings['rating_numbers'], how='left', lsuffix='_left', rsuffix='_right')
    
    final = corr_movies[corr_movies['rating_numbers']>min_rating_count].sort_values('Correlation', ascending=False)
    return final


# In[ ]:


recommendations = recommendMovies('GoldenEye (1995)')
recommendations.head()


# ### What's Next
# - We will be exploring advanced metrics to study performance our movie recommendation engine
# - We will explore the famous Netflix data
# - Explore the context of the data given on below kaggle link
# https://www.kaggle.com/netflix-inc/netflix-prize-data
