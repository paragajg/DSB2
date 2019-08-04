
# coding: utf-8

# ## Netflix Recommendation Competition Dataset

# In[ ]:


get_ipython().system(' pip install surprise')


# In[2]:


import pandas as pd
import numpy as np
import math
import re
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from surprise import Reader, Dataset, SVD, evaluate
sns.set_style("white")


# ### Load data set

# In[3]:


df = pd.read_csv("netflix-prize-data/netflix.csv")
df.drop(labels= ["Unnamed: 0"],inplace= True,axis =1)


# In[ ]:


df.head()


# ### Remove movies with less count of ratings

# In[4]:


f = ['count','mean']

df_movie_summary = df.groupby('Movie_Id')['Rating'].agg(f)
df_movie_summary.index = df_movie_summary.index.map(int)
movie_benchmark = round(df_movie_summary['count'].quantile(0.8),0)
drop_movie_list = df_movie_summary[df_movie_summary['count'] < movie_benchmark].index

print('Movie minimum times of review: {}'.format(movie_benchmark))

df_cust_summary = df.groupby('Cust_Id')['Rating'].agg(f)
df_cust_summary.index = df_cust_summary.index.map(int)
cust_benchmark = round(df_cust_summary['count'].quantile(0.8),0)
drop_cust_list = df_cust_summary[df_cust_summary['count'] < cust_benchmark].indexbbb

print('Customer minimum times of review: {}'.format(cust_benchmark))


# ### Use Singular Value Decompostion (SVD) to predict move preference
# 
# - Using surprise package to train SVD and predict movies for a user
# ref: https://surprise.readthedocs.io/en/stable/

# In[ ]:


reader = Reader()

# get just top 100K rows for faster run time
data = Dataset.load_from_df(df[['Cust_Id', 'Movie_Id', 'Rating']][:500000], reader)
data.split(n_folds=3)

svd = SVD()
evaluate(svd, data, measures=['RMSE', 'MAE'])


# ### Load Movies title dataset

# In[ ]:


df_title = pd.read_csv('netflix-prize-data/movie_titles.csv', encoding = "ISO-8859-1", header = None, names = ['Movie_Id', 'Year', 'Name'])
df_title.set_index('Movie_Id', inplace = True)
print (df_title.head(10))


# In[ ]:


df_785314 = df[(df['Cust_Id'] == 785314) & (df['Rating'] == 5)]
df_785314 = df_785314.set_index('Movie_Id')
df_785314 = df_785314.join(df_title)['Name']
print(df_785314[:10])


# In[ ]:


# Let's predict which movies user 785314 would love to watch:
user_785314 = df_title.copy()
user_785314 = user_785314.reset_index()
user_785314 = user_785314[~user_785314['Movie_Id'].isin(drop_movie_list)]

# getting full dataset
data = Dataset.load_from_df(df[['Cust_Id', 'Movie_Id', 'Rating']][:1000000], reader)


trainset = data.build_full_trainset()
#svd = SVD()
svd.fit(trainset)

user_785314['Estimate_Score'] = user_785314['Movie_Id'].apply(lambda x: svd.predict(785314, x).est)
v
user_785314 = user_785314.drop('Movie_Id', axis = 1)

user_785314 = user_785314.sort_values('Estimate_Score', ascending=False)
print(user_785314.head(10))


# ### Writing a function to automate the above steps

# In[29]:


def predict_movie(user_id,movie_list,df):
    # view historical preference of the user
    temp_usr = df[(df['Cust_Id'] == user_id) & (df['Rating'] == 5)]
    temp_usr = temp_usr.set_index('Movie_Id')
    temp_usr = temp_usr.join(df_title)['Name']
    print("Movies Previously liked by user.....................")
    print(temp_usr[:10])
    
    # create svd model to predict movies for user
    user = movie_list.copy()
    user = user.reset_index()
    user = user[~user['Movie_Id'].isin(drop_movie_list)]

    # getting dataset
    reader = Reader()
    data = Dataset.load_from_df(df[['Cust_Id', 'Movie_Id', 'Rating']][:1000000], reader)

    trainset = data.build_full_trainset()
    svd = SVD()
    svd.fit(trainset)

    user['Estimate_Score'] = user['Movie_Id'].apply(lambda x: svd.predict(user_id, x).est)

    user = user.drop('Movie_Id', axis = 1)

    user = user.sort_values('Estimate_Score', ascending=False)
    print("Recommended Movies for User are as follows.........\n")
    print(user.head(10))


# In[ ]:


predict_movie(user_id=1488844, movie_list= df_title, df = df)


# In[31]:


## Use below customer ids to verify the preference

df["Cust_Id"].tail()

