
# coding: utf-8

# ## Regression Analysis : First Machine Learning Algorithm !!

# __Why use linear regression?__
# 
# 1. Easy to use
# 2. Easy to interpret
# 3. Basis for many methods
# 4. Runs fast
# 5. Most people have heard about it :-) 
# 
# ### Libraries in Python for Linear Regression
# 
# The two most popular ones are
# 
# 1. `scikit-learn`
# 2. `statsmodels`
# 
# Highly recommend learning `scikit-learn` since that's also the machine learning package in Python.

# ### Linear regression 
# 
# Let's use `scikit-lean` for this example. 
# 
# Linear regression is of the form:
# 
# $y = \beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_nx_n$
# 
# - $y$ is what we have the predict/independent variable/response variable
# - $\beta_0$ is the intercept/slope
# - $\beta_1$ is the coefficient for $x_1$ (the first feature/dependent variable)
# - $\beta_n$ is the coefficient for $x_n$ (the nth feature/dependent variable)
# 
# The $\beta$ are called *model coefficients*
# 
# The model coefficients are estimated in this process. (In Machine Learning parlance - the weights are learned using the algorithm). The objective function is least squares method. 
# <br>
# 
# **Least Squares Method** : To identify the weights so that the overall solution minimizes the sum of the squares of the errors made in the results of every single equation. [Wiki](https://en.wikipedia.org/wiki/Least_squares)
# 
# <img style="float: left;" src = "./img/lin_reg.jpg" width="600" height="600">

# <h2> Model Building & Testing Methodology </h2>
# <img src="./img/train_test.png" alt="Train & Test Methodology" width="700" height="600">

# In[26]:


# Step1: Import packages
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(color_codes = True)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Step2:  Load our data
df = pd.read_csv('./data/Mall_Customers.csv')
df.rename(columns={'CustomerID':'id','Spending Score (1-100)':'score','Annual Income (k$)':'income'},inplace=True)
df.head() # Visualize first 5 rows of data


# In[ ]:


# Step3: Feature Engineering - transforming variables as appropriate for inputs to Machine Learning Algorithm
# transforming categorical variable Gender using One hot encodding
gender_onhot = pd.get_dummies(df['Gender'])
gender_onhot.head()


# In[ ]:


# Create input dataset aka X
X = pd.merge(df[['Age','income']], gender_onhot, left_index=True, right_index=True)
X.head()


# In[ ]:


# Create target variable
Y = df['score']
Y.head()


# In[ ]:


# Step3: Split data in train & test set
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.20, random_state=42)
print('Shape of Training Xs:{}'.format(X_train.shape))
print('Shape of Test Xs:{}'.format(X_test.shape))


# In[38]:


# Step4: Build Linear Regression Analysis Model
learner = LinearRegression(); #initializing linear regression model

learner.fit(X_train,y_train); #training the linear regression model
y_predicted = learner.predict(X_test)
score=learner.score(X_test,y_test);#testing the linear regression model


# In[ ]:


print(score)
print(y_predicted)


# In[ ]:


# Step5: Check Accuracy of Model
df_new = pd.DataFrame({"true_score":y_test,"predicted_score":y_predicted})
df_new


# In[ ]:


# Step6: Diagnostic analysis

from sklearn.metrics import mean_squared_error, r2_score
print("Intercept is at: %.2f"%(learner.intercept_))
# The coefficients
print('Coefficients: \n', learner.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_predicted))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_predicted))

