#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
from sklearn import svm
import pandas as pd


# ### Using SVM to predict quality of wine

# In[2]:


# Dataset
dataset = pd.read_csv(r'C:\Users\User\Downloads\wineQualityReds.csv')

dataset.head(100)


# Quality is dependant on all the other features, so lets plot all the relationship graphs to better visualise their correlationship

# In[3]:


for x in dataset.columns[1:-1]:
    quality = dataset['quality'].sort_values()
    xx = dataset[x].sort_values()
    
    plt.scatter(dataset[x], quality, s=1, alpha=1, c='green')
    
    fit = np.poly1d(np.polyfit(dataset[x], quality, 1))
    xx = np.linspace(min(dataset[x]),max(dataset[x]),100)
    yy = [fit(i) for i in xx]
    
    plt.plot(xx, yy, c='red', lw=10, alpha=0.5)
    
    plt.title(f'{x} vs Quality')
    
    plt.xlabel(str(x))
    plt.ylabel('Quality')
    
    plt.show()
    print(f'Correlation: {np.corrcoef(dataset[x], quality)[0][1]}')
    


# We can see that they rarely form a strong correlationship with the quality, but all of them together might produce some results. Lets try with some prediction models to see if they work well together in predicting the quality.

# In[4]:


# Make sure that any null values are replaced

for i in dataset.columns:
    dataset[i].fillna(np.mean(dataset[i]))
    print(i, '\nFilled all NaN values with mean of ',i,':',np.mean(dataset[i]),'.\n\n')


# In[5]:


from sklearn.model_selection import train_test_split


# In[6]:


# Get rid of unsuitable subsets
x_ = pd.DataFrame(dataset.drop(columns='quality')).head(120)
x_ = pd.DataFrame(x_.drop(columns='Unnamed: 0'))

y_ = dataset['quality'].head(120)


x_train, x_test, y_train, y_test = train_test_split(x_, y_, test_size=0.4, shuffle=False)


# In[7]:


# Testing best model to fit this data with

c = [1,10,100]
kernels = ['linear','poly','rbf']

for c_ in c:
    for k_ in kernels:
        model = svm.SVC(C=c_,kernel=k_,gamma='auto').fit(x_train,y_train)
        score = model.score(x_test, y_test)
        print(f'C value of {c_}, \nKernel: {k_}, Score={score}')


# In[8]:


# We will use C-value of 1, linear kernel, with score of 0.75.
model = svm.SVC(gamma='auto',C=1,kernel='linear').fit(x_train,y_train)


# In[11]:


# now try to predict quality

# There are many variables that affect quality:
labels = [i for i in dataset.columns]

labelss = []
for x in labels:
    if labels.index(x)==0 or labels.index(x)==len(labels):
        print('Removed ',x)
    else:
        labelss.append(x)
labels = labelss
        
print(labels[:-1])

# Only from index 1 to 2nd last we will use, so the input has to have:
inputt = []


choice = int(input('\n\nDo you want to paste formatted features or type manually? (0/1)  '))

if choice==0:
    inputted = (input('Paste Features  '))
    inputt.append(eval(inputted))
elif choice==1:
    for x in labels[:-1]:
        inputted = input(f'Key in {x}  (Ranging from {dataset[x].min()} to {dataset[x].max()})  ')
        if not dataset[x].min() < float(inputted) < dataset[x].max():
            print('\033[93m Warning: Inappropriate Value. May interfere with results \033[0m')
        else:
            print('\033[92m Success! \033[0m')
        
        inputt.append(inputted)


# In[16]:


# Final Prediction
print('When',inputt,'\nQuality is: ',(model.predict(np.array(inputt).reshape(-1,11))[0]))


# In[15]:


print(model.score(x_train,y_train))
print(model.score(x_test,y_test))


# Relatively high score for this prediction model for both test and training data. May not be very viable as range of quality is very small, an inaccurate prediction may err wildly from the expected value.
