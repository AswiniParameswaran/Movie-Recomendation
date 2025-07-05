#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# movies_data=pd.read_csv(r"C:\Users\Aswini\Downloads\movies.csv")

# In[2]:


movies_data=pd.read_csv(r"C:\Users\Aswini\Downloads\movies.csv")


# In[3]:


movies_data.head()


# In[4]:


movies_data.info


# In[5]:


movies_data.isnull().sum()


# In[6]:


movies_data.shape


# In[7]:


selected_features=['genres','keywords','tagline','cast','director']


# In[8]:


for feature in selected_features:
    movies_data[feature]=movies_data[feature].fillna('')


# In[9]:


combined_features=movies_data['genres']+''+movies_data['keywords']+''+movies_data['tagline']+''+movies_data['cast']+''+movies_data['director']


# In[10]:


print(combined_features)


# In[11]:


vectorizer=TfidfVectorizer()
feature_vectors=vectorizer.fit_transform(combined_features)
print(feature_vectors)


# In[12]:


similarity=cosine_similarity(feature_vectors)
print(similarity)


# In[13]:


print(similarity.shape)


# In[14]:


movie_name=input('Enter your Favorite Movie Name:')


# In[15]:


list_of_all_titles=movies_data['title'].tolist()
print(list_of_all_titles)


# In[16]:


find_close_match=difflib.get_close_matches(movie_name,list_of_all_titles)
print(find_close_match)


# In[17]:


close_match=find_close_match[0]
print(close_match)


# In[18]:


index_of_the_movie=movies_data[movies_data.title== close_match]['index'].values[0]
print(index_of_the_movie)


# In[20]:


similarity_score=list(enumerate(similarity[index_of_the_movie]))
print(similarity_score)


# In[21]:


len(similarity_score)


# In[23]:


sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

print(sorted_similar_movies)


# In[25]:


print("Movies suggested for you:\n")
i=1
for movie in sorted_similar_movies:
    index=movie[0]
    title_from_index=movies_data[movies_data.index==index]['title'].values[0]
    if (i<30):
        print(i,'.',title_from_index)
        i+=1


# In[ ]:




