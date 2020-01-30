#!/usr/bin/env python
# coding: utf-8

# In[1]:


import scipy.stats as st
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


# In[7]:


data = pd.read_csv('Downloads/crime.csv',encoding = 'unicode_escape', sep=',')


# In[11]:


df = pd.DataFrame(data)
df.head(10)


# In[83]:


district = df['DISTRICT'].head(1000)
district.describe()


# In[84]:


# Remove NaN

pd.isnull(district)
district.dropna()


# In[85]:


count, district_names = 0,[]

for test in district:
    if test=='D14':
        count = count+1
    
    indicator = True
    for check in district_names:
        if check==test:
            indicator = False
    
    if indicator:
        district_names.append(test)

print(count)
print(district_names)

# Idk why still got Nan, will remove with conventional methods


# In[97]:


def remove_nan(a):
    for freq in range(100):
        for x in district_names:
            if type(x)==float:
                district_names.remove(x)


# In[98]:


remove_nan(district_names)
district_names


# In[99]:


crime_freq = []

for test in district_names:
    count = 0
    for x in district:
        if x==test:
            count = count + 1
    crime_freq.append(count)
    print(f'In {test} there is/are {count} case(s) of criminal activity.')


# In[100]:


length = np.arange(len(district_names))

plt.figure(figsize=(10,8))
plt.bar(length, crime_freq, align='center', color='crimson', alpha=0.8)

plt.xticks(length, district_names, size=12); plt.yticks(size=12)

plt.title('Crime Rate in 11 Districts of Boston')
plt.xlabel('Districts'); plt.ylabel('Crime Rate')

plt.plot(length, crime_freq, lw=4, alpha=0.5, c='black')

