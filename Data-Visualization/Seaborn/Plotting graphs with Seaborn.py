#!/usr/bin/env python
# coding: utf-8

# # Plotting graphs with Seaborn

# In[2]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


iris = sns.load_dataset('iris')
iris.shape


# In[4]:


iris.head()


# In[5]:


iris['species'].unique()


# # Univariate Analysis

# ## Histogram 
# 
# #### Definition 
# 
# Histogram indicates how numerical data are distributed. or it is a **graphical representation of data points grouped into ranges** that the user specifies. or Histograms illustrate the frequency distribution of data sets.
# 
# #### Key Pointers 
# 
#   - Histograms show data in a bar graph-like manner by grouping outcomes along a vertical axis.
#   - The y-axis of a histogram represents occurrences, either by number or by percentage or in simple words the height of each bar represents either a number count or a percentage.
#   - MACD (Moving Average Convergence Divergence) histograms are used in technical analysis to indicate changes in momentum in the market. 
#   
# #### Difference between histograms and bar charts 
# 
# Often, histograms are confused with bar charts.Generally, a histogram is used to plot continuous data, in which each bin represents a range of values, while a bar chart shows categorical data
# 
# More techincally,Histograms depict the frequency distribution of variables in a data set. A bar graph, by contrast, represents a graphical comparison of discrete or categorical variables.
# 
# #### A Histogram's Function
# 
# In statistics, histograms are used to indicate the frequencies of particular types of variables occurring within a defined range. As an example, a census that focuses on the demography of a country may show a histogram showing the number of people aged 0 - 10, 11 - 20, 21 - 30, 31 - 40, 41 - 50, etc.
# 
# A histogram can be customized in several ways by the analyst. As an alternative to frequency, one could use percent of total or density as labels.
# Another factor to consider would be bucket size. In the above example, there are 5 buckets with an interval of ten. You could change this, for example, to 10 buckets with a 5-minute interval instead.
# 
# A histogram gives a rough idea of the underlying distribution of data, and is often used for density estimation: estimating the probability density function of underlying variables.
# 

# In[6]:


iris['petal_length']


# In[9]:


sns.histplot(data=iris,x = 'petal_length',bins = 30)


# In[13]:


sns.histplot(data=iris,x = 'petal_length',bins = 30,hue='species')


# ## KDE - Kernel Density Estimation
# 
# contiuous probability density function (PDF)

# In[11]:


sns.kdeplot(data = iris, x='petal_length')


# In[12]:


sns.kdeplot(data = iris, x='petal_length',hue='species')
plt.show()


# ## Distribution plot

# In[14]:


sns.displot(data=iris,x='sepal_length',bins=40,hue='species')


# ## Bivariante Analysis

# ## Scatter plot - using seaborn

# In[18]:


sns.scatterplot(data=iris,x='petal_length',y = 'petal_width',hue='species')
#plt.xlim(-5,10)
plt.show()


# ## scatter plot using matplotlib 

# In[20]:


setosa = iris[iris['species'] == 'setosa']
versicolor = iris[iris['species'] == 'versicolor']
virginica = iris[iris['species'] == 'virginica']


# In[22]:


plt.scatter(x=setosa['petal_length'],y=setosa['petal_width'], c = 'blue')
plt.scatter(x=versicolor['petal_length'],y=versicolor['petal_width'], c = 'orange')
plt.scatter(x=virginica['petal_length'],y=virginica['petal_width'], c = 'green')


# ## Joint plot (Default it joins displot and scatterplot)

# In[25]:


sns.jointplot(data=iris,x='petal_length',y='petal_width',hue='species')
plt.show


# In[33]:


# ['scatter', 'hist', 'hex', 'kde', 'reg', 'resid']
sns.jointplot(data=iris,x='petal_length',y='petal_width',kind='reg')
plt.show


# In[38]:


sns.jointplot(data=iris,x='petal_length',y='petal_width',hue='species',kind='kde')
plt.show


# In[41]:


sns.jointplot(data=iris,x='sepal_length',y='sepal_width',kind='hex')
plt.show


# ## Multivariate Analysis

# In[42]:


iris.head()


# In[45]:


sns.pairplot(data = iris, hue='species')


# # Categorial variables

# ## Univariate Analysis

# In[46]:


iris.head()


# In[47]:


sns.countplot(data=iris,x = 'species')


# ## Box plot

# In[49]:


sns.boxplot(data=iris, y = 'petal_length')


# # Boxplot for Multi-variant Analysis

# In[51]:


sns.boxplot(data=iris,x = 'species', y = 'sepal_length')


# ## violinplot shows distribution and outliers.

# In[53]:


sns.violinplot(data=iris,x = 'species', y = 'sepal_length')


# ## Matrix Plots

# In[56]:


corr = iris.corr()
corr


# In[60]:


sns.heatmap(corr,annot = True)


# In[61]:


car_crashes = sns.load_dataset('car_crashes').corr()


# In[62]:


sns.heatmap(car_crashes.corr(),annot = True)


# In[ ]:




