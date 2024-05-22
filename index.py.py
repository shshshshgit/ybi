#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
df=load_digits()
_,axes=plt.subplots(nrows=1,ncols=4,figsize=(10,3))
for ax,image,label in zip(axes,df.images,df.target):
    ax.set_axis_off()
    ax.imshow(image,cmap=plt.cm.gray_r,interpolation="nearest")
    ax.set_title("Training: %i" % label)


# In[6]:


df.images.shape


# In[7]:


df.images[0]


# In[8]:


len(df.images)


# In[10]:


n_samples=len(df.images)
data=df.images.reshape((n_samples, -1))


# In[11]:


data[0]


# In[12]:


data[0].shape


# In[13]:


data.shape


# In[14]:


data.min()


# In[15]:


data.max()


# In[16]:


data=data/16


# In[17]:


data.min()


# In[18]:


data.max()


# In[19]:


data[0]


# In[21]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(data,df.target,test_size=0.3)
x_train.shape,x_test.shape,y_train.shape,y_test.shape


# In[22]:


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(x_train,y_train)


# In[23]:


y_pred=rf.predict(x_test)
y_pred


# In[24]:


from sklearn.metrics import confusion_matrix,classification_report
confusion_matrix(y_test,y_pred)


# In[25]:


print(classification_report(y_test,y_pred))


# In[ ]:




