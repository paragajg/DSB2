
# coding: utf-8

# ## Boston Housing Data Set
# 
# ### Assignment Goals:
# - Build Machine Learning model to predict Category of Income of an individual
# - Use pipeline and grid search to build strategy for experimenting your ML model 

# ### Load Libraries

# In[5]:


# Data manipulation libraries
import pandas as pd
import numpy as np

##### Scikit Learn modules needed for Logistic Regression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder,MinMaxScaler , StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Library to store and load models
import joblib

# Plotting libraries
from IPython.display import SVG
from graphviz import Source
from IPython.display import display
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes = True)
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Load Data

# In[ ]:


#df = pd.read_csv()
#df.head()


# In[ ]:


#print(df.describe())
#df.isna().sum()


# ### Visualize Data
# 
# - Use correlation plot (as shown in Decision Tree & Regression Models in class) to study correlation between numerical variables

# ### Build Strategy for your Machine Learning Pipeline
# - Define transformation of categorical variables
# - Define scaling for numerical variables

# In[3]:


# We create the preprocessing pipelines for both numeric and categorical data.

numeric_features = [] # add names of numerical variables which you want to add for building model
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])## Add your choice of scaler

categorical_features = [] #  add names of categorical variables which you want to add for building model
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))]) # Experiment with other label encoding techiques as well
 
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Append classifier to preprocessing pipeline.
# Now we have a full prediction pipeline.
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', DecisionTreeClassifier())])# Change classifier and try RandomForest & Logistic Regession as well


# ### Split your data

# In[4]:


#X_train, X_test, y_train, y_test = train_test_split(,test_size=0.2,random_state=42) # add your X & y


# In[ ]:


# Fit your model to check accuracy
clf.fit(X_train, y_train)
#print("model score: %.3f" % clf.score(X_test, y_test))


# ### Experiment with Hyper Parameters using Grid Search
# 
# Reference on Grid Search https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV
# 
# - Refer to individual models on scikit learn to know more about options in hyper parameters associated with Decision Trees , Logistic Regression and Random Forest

# In[ ]:


param_grid = {
    'preprocessor__num__imputer__strategy': ['mean', 'median'],
    'classifier__criterion': ["gini","entropy"]
}

grid_search = GridSearchCV(clf, param_grid, cv=10, iid=False)
grid_search.fit(X_train, y_train)

print(("best Model from grid search: %.3f"
       % grid_search.score(X_test, y_test)))


# In[ ]:


# Print your best combination of hyper parameters
grid_search.best_params_


# ### Store your model using joblib Library

# In[ ]:


joblib.dump(grid_search,"my_model.model")

