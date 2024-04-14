#!/usr/bin/env python
# coding: utf-8

# In[35]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[36]:


df=pd.read_csv("D:\placement.csv")
df


# In[37]:


df.shape


# In[38]:


df.iloc[:,1:]


# In[39]:


df2 = plt.scatter(df['cgpa'],df['iq'])


# In[40]:


df2 = plt.scatter(df['cgpa'],df['iq'],c=df['placement'])


# In[41]:


#logistic regresion thats are the cut the data 
X = df.iloc[:,1:3]
y = df.iloc[:,-1]


# In[42]:


X


# In[43]:


y


# In[44]:


y.shape


# In[45]:


from sklearn.model_selection import train_test_split


# In[46]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1)


# In[47]:


X_train,X_test,y_train,y_test


# In[48]:


X_train


# In[49]:


y_train


# In[50]:


X_test


# In[51]:


y_test


# In[52]:


from sklearn.preprocessing import StandardScaler


# In[53]:


scaler = StandardScaler()


# In[54]:


X_train = scaler.fit_transform(X_train)


# In[55]:


X_train


# In[56]:


X_test = scaler.transform(X_test)


# In[57]:


X_test


# In[58]:


from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()


# In[59]:


clf.fit(X_train,y_train)


# In[60]:


y_pred = clf.predict(X_test)


# In[61]:


y_pred


# In[62]:


y_test


# In[63]:


from sklearn.metrics import accuracy_score


# In[1]:


accuracy_score(y_test,y_pred)
# they are given 10 to 9 good prediction in this case 


# In[67]:


from mlxtend.plotting import plot_decision_regions


# In[66]:


plot_decision_regions(X_train.values,y_train,clf=svm, legend=2)


# In[ ]:




