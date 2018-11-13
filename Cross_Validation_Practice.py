
# coding: utf-8

# In[3]:


from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


# In[4]:


iris = load_iris()
X = iris.data
y = iris.target


# In[5]:


X


# In[3]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 4)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))


# In[6]:


from sklearn.cross_validation import KFold
kf = KFold(25, n_folds = 5, shuffle = False)
print('{} {:^61} {}'.format('Iteration', 'Training set observations', 'Testing set observations'))
for iteration, data in enumerate(kf, start = 1):
    print('{:^9} {} {:^25}'.format(iteration, data[0], data[1]))


# In[7]:


from sklearn.cross_validation import cross_val_score


# In[8]:


scores = cross_val_score(knn, X, y, cv = 10, scoring='accuracy')
print(scores)


# In[9]:


print(scores.mean())


# In[10]:


k_range = range(1, 31)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv = 10, scoring='accuracy')
    k_scores.append(scores.mean())
print(k_scores)


# In[11]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')


# In[12]:


knn20 = KNeighborsClassifier(n_neighbors=20)
print(cross_val_score(knn20, X, y, cv=10, scoring='accuracy').mean())


# In[13]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
print(cross_val_score(logreg, X, y, cv=10, scoring = 'accuracy').mean())


# In[14]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


# In[15]:


data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', index_col = 0)


# In[25]:


data


# In[16]:


feature_cols = ['TV', 'radio', 'newspaper']
X = data[feature_cols]
y = data.sales


# In[17]:


lm = LinearRegression()
scores = cross_val_score(lm, X, y, cv=10, scoring='mean_squared_error')
print(scores)


# In[19]:


mse_scores = -scores
print(mse_scores)


# In[20]:


rmse_scores = np.sqrt(mse_scores)
print(rmse_scores)


# In[21]:


print(rmse_scores.mean())


# In[22]:


feature_cols = ['TV', 'radio']
X = data[feature_cols]
print(np.sqrt(-cross_val_score(lm, X, y, cv=10, scoring='mean_squared_error')).mean())

