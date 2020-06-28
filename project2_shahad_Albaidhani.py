#!/usr/bin/env python
# coding: utf-8

# 
# 
# # Project: Investigate a Dataset (The Movie Database (TMDb))
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# <a id='intro'></a>
# # Introduction
# 
# > I will analyze The Movie Database .
# This data set contains information about 10,000 movies collected from The Movie Database (TMDb),
# including user ratings and revenue etc.
# 
# ><b>Our goal in the analysis is to find an answer to these questions.</b>
# 
# >  <ul> 
#    <li>What are the most profitable movies over the years ?</li>
#     <li> Are comedy-type movies more popular than horror-type movies?</li>
#     <li>comedy-type movies more productive than horror-type movies over years</li>
#     <li>Who are the companies that are most productive and profitable ?</li>
#     
# </ul>
# 
# 

# In[1]:



#importing import files

import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# <a id='wrangling'></a>
# ## Data Wrangling
# 
# 
# 
# ### General Properties

# In[2]:


# Loading the csv file and storing in the variable "df" 
df=pd.read_csv('tmdb-movies.csv')


# In[3]:


#print out a few lines
df.head()


# In[4]:


#types and look for instances of missing or possibly errant data.
df.info()


# 
# 
# ### Data Cleaning (Replace this with more specific notes!)

# #### To Data processing
#  
# 
# 
# 
# <ul>
# <li>We need to delete unused columns</li>
# <li>then,Delete duplicate rows </li>
# <li>finally, Deal with a null value and zero value</li>
# </ul>
# 

#  ##### delete unused columns

# In[5]:


#delete unused columns
df.drop(['imdb_id','cast','homepage','keywords','overview','runtime','budget_adj','revenue_adj'],axis=1,inplace=True)

#print out a few lines after delete unused columns
df.head()


# ##### Delete duplicate rows

# In[6]:


#Check for duplicate rows

sum(df.duplicated())


# we have one duplicate row

# In[7]:


# delete duplicate row
df.drop_duplicates(inplace=True)


# In[8]:


#Check if the duplicate row was deleted

sum(df.duplicated())


# ##### Deal with a null value (missing value)and zero value

# In[9]:


#Check for null value


df.isnull().sum()


# We find that null values 
# 
# so we delete null values because the data type is a string

# In[10]:


# delete null values because the data type is a string
df.dropna(inplace=True)


# In[11]:


#Check if the null values was deleted
df.isnull().sum()


# In[12]:


#Check for zero values
df.describe()


# In[13]:


# replace zero value with null value
df['budget'].replace(0,np.NAN,inplace=True)
df['revenue'].replace(0,np.NAN,inplace=True)

# delete null values
df.dropna(axis=0,inplace=True)


# In[14]:



#Check if the zero values was deleted

df.describe()


# In[15]:


#see dataset after cleaning 
df.info()


# <b> We find that budget and revenue data types change from int to float </b>

# In[16]:


#changing data type
change_type=['budget', 'revenue']
df[change_type]=df[change_type].applymap(np.int64)
#see dataset after
df.info()


# <a id='eda'></a>
# ## Exploratory Data Analysis
# 
# 
# ### Research Question 1 (What are the most profitable movies over the years ?)

# First calculating the profit of the each movie and Profit must be
# greater than zero if Profit less than 0 meaning is Loss
# 

# In[17]:


#function calculating the profit 

def profit(revenue,budget) :
    #calculating the profit 
    profit=df[revenue]-df[budget]
    return profit



# In[18]:


#call function profit and Add a column profit in dataset
df['profit']=profit('revenue','budget')

#Profit must be greater than zero
df=df.query('profit >= 0')

#print out a few lines after add a column profit
df.head()


# *Now we added profit column in dataset*

# 
# **second**, we calculating most profitable movies over the years

# In[19]:


#calculating most profitable movies over the years
profit_years=df.groupby(['release_year'])['profit','release_year','original_title'].max()
#arrange of ascending values
profit_years.sort_values(by='profit',ascending=False ,inplace=True)

#print
profit_years


# *Now we have most profitable movies over the years*

# In[20]:


#plotting a bar graph of The most profitable movies
profit_years.plot( x='original_title',y='profit',kind='bar',title='The most profitable movies',figsize=(13,13));
plt.xlabel("movies");
plt.ylabel("profit");
plt.legend();


# **This graph show that highest profitable movies are zombieland and women in gold**

# **comparison the highest profitable year movies over the years**

# In[21]:


#plotting a scatter graph of The most profitable movies over the years

profit_years.plot( x='release_year',y='profit',kind='scatter',title='The most profitable movies over the years',figsize=(13,13));
 


# **This graph show that percentage of movie profits increases more than in previous years**
# 
# **You can see that in 2010 the profits for the year increased more than in previous year**

# ### Research Question 2  (Are comedy-type movies more popular than horror-type movies ?)

# **first get comedy-type rows and horror-type rows form genres column**

# In[22]:


#function get special type of genres
def genresTyep (x) :
    return df.genres==x


# In[23]:


#collecting the comedy-type movies
Comedy=genresTyep('Comedy')

#collecting the Horror-type movies
Horror=genresTyep('Horror')


# **second get popularity for comedy-type rows and horror-type rows**

# In[24]:


#plotting a histogram  graph of A comparison of the most popular between the two types
df.popularity[Comedy].plot(kind="hist",alpha=0.5,bins=20,label='Comedy',title='The popular between comedy movies and horror movies');
df.popularity[Horror].plot(kind="hist",alpha=0.5,bins=20,label='Horror');
plt.ylabel("popularity")
plt.xlabel("genres")
plt.legend();


# **This graph show that The most popular of the two type are comedy movies**
# 
# **comparison the Produce of years between the two types**
# 

# In[25]:


#plotting a bar graph of A comparison of the Produce of years between the two types
df.release_year[Comedy].value_counts().plot(kind='bar',figsize=(20,20),label='Comedy',title=' The most Produce of years between the two types');
df.release_year[Horror].value_counts().plot(kind='bar',color='red',label='Horror',figsize=(20,20));
plt.xlabel("years")
plt.ylabel("productivity")
plt.legend();


# **This graph show that The most Produce of years of the two genres are comedy movies**
# 

# ### Research Question 3  (Who are the companies that are most productive and profitable ?)

# **first we calculate the productivity of production companies over the years**

# In[26]:


#calculating the most productive of production companies over the years
max_productivity_of_production_companies=df.groupby('release_year').production_companies.max()
#count productive of production companies 
counts_production_companies=max_productivity_of_production_companies.value_counts()


# In[27]:


#print
counts_production_companies


# **second, We define largest of productive for production companies**

# In[36]:


#largest number of productive production company 
highest_production=counts_production_companies.nlargest()


#show largest number of productive production company 
highest_production


# In[29]:


#plotting a pie chart of A comparison of The highest productive production companies
highest_production.plot.pie(figsize=(8, 8),autopct='%1.1f%%', startangle=90, shadow=False,legend = False, fontsize=10,title="The highest productive production companies");



# **This graph shows that the highest production companies is Warner Bros by 41.2%**
# 
# **we calculation the profit production companies**

# In[30]:


#The max profit from production companies
df.groupby('production_companies').profit.max()


# **we calculation maximum profit of production companies**

# In[31]:


#calculation_maximum profit of production companies
max_productivity_of_production_companies=df.groupby('profit').production_companies.max()
#counts profit of production companies
highest_profit_production=max_productivity_of_production_companies.value_counts()


# In[32]:


#show most profit of production companies

highest_profit_production


# In[33]:


#largest number of profit production company 

highest_profit_production.nlargest(keep='last')


# In[34]:


#plotting a pie chart of A comparison of The highest profit production companies


highest_profit_production.nlargest().plot.pie(figsize=(8, 8),autopct='%1.1f%%', startangle=90, shadow=False,legend = False, fontsize=10,title='The highest profit production companies');


# **This graph shows that most profit of production companies is Paramount Pictures by 31.4%**
# 

# <a id='conclusions'></a>
# ## Conclusions
# 
# > This was a very interesting data analysis. We came out with some very interesting facts about movies. After this analysis we can conclude following:
# >  <ul> 
#    <li>The Zombieland movie is most profitable over the years</li>
#     <li>Every year the percentage of movie profits increases more than in previous years</li>
#     <li>we find comedy-type movies more popular than horror-type movies</li>
#     <li>comedy-type movies more productive than horror-type movies over years</li>
#     <li>Warner Bros production company most productive over years</li>
#     <li>Paramount Pictures  production company most profitable over years </li>
# </ul>
# 
# >**Limitations**: This analysis was done considering the movies which had a Some of the titles of movies written in a language other than English are difficult to recognize and this affects the analysis . Dropping the rows with missing values also affected the overall analysis.

# In[35]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Investigate_a_Dataset.ipynb'])


# In[ ]:





# In[ ]:





# In[ ]:




