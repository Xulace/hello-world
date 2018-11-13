
# coding: utf-8

# In[1]:


import pandas as pd


# In[6]:


data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', index_col = 0)
data.head()


# In[7]:


data.tail()


# In[8]:


data.shape


# In[9]:


import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[16]:


sns.pairplot(data, x_vars=['TV', 'radio', 'newspaper'], y_vars='sales', size=7, aspect=0.7, kind='reg')


# In[19]:


feature_cols = ['TV', 'radio', 'newspaper']
X = data[feature_cols]
X.head()


# In[20]:


X = data[['TV', 'radio', 'newspaper']]
X.head()


# In[22]:


print(type(X))
print(X.shape)


# In[23]:


y = data['sales']
y = data.sales
y.head()


# In[24]:


print(type(y))
print(y.shape)


# In[25]:


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


# In[26]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[27]:


from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_train, y_train)


# In[28]:


print(linreg.intercept_)
print(linreg.coef_)


# In[35]:


zipped = zip(feature_cols, linreg.coef_)
list(zipped)


# In[36]:


y_pred = linreg.predict(X_test)


# In[37]:


true = [100, 50, 30, 20]
pred = [90, 50, 50, 30]


# In[38]:


print((10+0+20+10)/4)

from sklearn import metrics
print(metrics.mean_absolute_error(true, pred))


# In[39]:


print((10**2+0**2+20**2+10**2)/4)
print(metrics.mean_squared_error(true, pred))


# In[42]:


import numpy as np
print(np.sqrt((10**2+0**2+20**2+10**2)/4))
print(np.sqrt(metrics.mean_squared_error(true,pred)))


# In[43]:


print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[45]:


feature_cols = ['TV', 'radio']
X = data[feature_cols]
y = data.sales
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
linreg.fit(X_train, y_train)
y_pred = linreg.predict(X_test)
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

