#!/usr/bin/env python
# coding: utf-8

# In[1]:


### Importing dependencies

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, normalize,MinMaxScaler
from sklearn.cluster import KMeans
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore") 
# %matplotlib inline


# In[19]:


df = pd.read_csv('test_data.csv')
df.head()


# In[23]:


df.isna().sum()


# In[21]:


numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
Numerical_columns=list(df.select_dtypes(include=numerics).columns)
Categorical_columns=list(df.select_dtypes('object').columns)

Categorical_columns


# In[22]:


### Treating missing values

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
Numerical_columns=list(df.select_dtypes(include=numerics).columns)
Categorical_columns=list(df.select_dtypes('object').columns)

for i in Numerical_columns:
    df[i]=df[i].fillna(0)
    
for i in Categorical_columns:
    df[i]=df[i].fillna("Unknown")
    


# In[25]:


df_center=df[["Cost Center"]]


# In[26]:


df.drop("Cost Center",inplace=True,axis=1)


# In[27]:


df.head()


# In[28]:



from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

# Create some toy data in a Pandas dataframe

class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)


# In[29]:


### Fetching out the numerical and categorical columns

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
Numerical_columns=list(df.select_dtypes(include=numerics).columns)
Categorical_columns=list(df.select_dtypes('object').columns)

df_transform=MultiColumnLabelEncoder(columns = Categorical_columns).fit_transform(df)


# In[30]:


df_transform.head()


# In[31]:


### Normalizing the data points and running K-Means to find the best elbow point
std=MinMaxScaler()
arr1=std.fit_transform(df_transform)

SSE = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, random_state = 99)
    kmeans.fit(arr1)
    SSE.append(kmeans.inertia_)
  

plt.figure(figsize=(12,5))
plt.plot(range(1, 11), SSE,marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Sum of Squares')
plt.show()


# In[32]:


kmeans_cluster = KMeans(n_clusters = 3, random_state = 23)
result_cluster=kmeans_cluster.fit_predict(arr1)

df['clusters']=result_cluster
df['clusters'].value_counts()


# In[33]:


df_final_cluster=pd.concat([df_center,df],axis=1)
### The final Dataset
df_final_cluster


# In[49]:


df_final_cluster.to_csv("test_control_final.csv",index=False)


# In[34]:


df_cluster_0=df_final_cluster[df_final_cluster['clusters']==0]
df_cluster_1=df_final_cluster[df_final_cluster['clusters']==1]
df_cluster_2=df_final_cluster[df_final_cluster['clusters']==2]


# In[42]:


print(df_cluster_2.shape)
print(df_cluster_1.shape)
print(df_cluster_0.shape)


# In[46]:


df_cluster_1.sample(2)


# In[47]:


df_cluster_2.sample(1)


# In[48]:


df_cluster_0.sample(1)

