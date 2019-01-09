
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import collections, re
import copy

from pandas.tools.plotting import scatter_matrix
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('bmh')

get_ipython().run_line_magic('matplotlib', 'inline')

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
print(train.describe())


# In[4]:


## exctract cabin letter
def extract_cabin(x):
    return x!=x and 'other' or x[0]
train['Cabin_l'] = train['Cabin'].apply(extract_cabin)


# In[5]:


plain_features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'Cabin_l']
fig, ax = plt.subplots(nrows = 2, ncols = 3 ,figsize=(20,10))
start = 0
for j in range(2):
    for i in range(3):
        if start == len(plain_features):
            break
        sns.barplot(x=plain_features[start], y='Survived', data=train, ax=ax[j,i])
        start += 1

