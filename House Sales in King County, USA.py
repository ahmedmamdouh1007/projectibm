#!/usr/bin/env python
# coding: utf-8

# In[30]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df= pd.read_csv("Desktop/kc_house_data.csv")


# In[3]:


df.head()


# In[4]:


print(df.dtypes)


# In[5]:


df["id"].drop


# In[6]:


df.describe()


# In[7]:


df["floors"].value_counts().to_frame()


# In[8]:


import seaborn as sns


# In[9]:


sns.boxplot(x= "waterfront", y= "price", data= df)


# In[10]:


sns.regplot(x= "sqft_above", y= "price", data= df)
plt.ylim(0,)


# In[11]:


from sklearn.linear_model import LinearRegression


# In[12]:


LR= LinearRegression()
x= df[["sqft_living"]]
y= df["price"]
LR.fit(x,y)


# In[13]:


LR.intercept_ , LR.coef_


# In[14]:


y_pred= LR.predict(x)  


# In[15]:


LR.score(x, y)


# In[16]:


import sklearn.metrics as ms


# In[17]:


ms.mean_absolute_error(y , y_pred)


# In[24]:


features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]
x= df[features]
y= df["price"]
LR.fit(x, y)
LR.score(x, y)


# In[33]:


Input= [("scale",StandardScaler()),("polynomial", PolynomialFeatures()),("model",LinearRegression())]
pipe=Pipeline(Input)
pipe
pipe.fit(x, y)
pipe.score(x, y)


# In[35]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
X = df[features]
Y = df['price']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)


print("number of test samples:", x_test.shape[0])
print("number of training samples:",x_train.shape[0])


# In[36]:


from sklearn.linear_model import Ridge
RigeModel=Ridge(alpha=0.1)
RigeModel.fit(x_train, y_train)
RigeModel.score(x_test, y_test)


# In[37]:


pr = PolynomialFeatures(degree=2)
x_train_pr = pr.fit_transform(x_train)
x_test_pr = pr.fit_transform(x_test)

RigeModel=Ridge(alpha=0.1)
RigeModel.fit(x_train_pr, y_train)
RigeModel.score(x_test_pr, y_test)


# In[ ]:




