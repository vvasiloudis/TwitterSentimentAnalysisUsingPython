#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import nltk
tweets=pd.read_csv('Tweets.csv')
tweets.head()


# In[2]:


tweets.shape


# In[3]:


tweets_df=tweets.drop(tweets[tweets['airline_sentiment_confidence']<0.5].index,axis=0)
tweets_df.shape


# In[4]:


Variable_X=tweets_df['text']
Variable_Y=tweets_df['airline_sentiment']


# In[5]:


from nltk.corpus import stopwords
nltk.download('stopwords')
import string
from nltk.stem import PorterStemmer


# In[6]:


stop_words=stopwords.words('english')
punct=string.punctuation
stopper=PorterStemmer()


# In[7]:


import re
cleaned_data=[]
for i in range(len(Variable_X)):
    tweet_topic=re.sub('[^a-zA-Z]',' ',Variable_X.iloc[i])
    tweet_topic=tweet_topic.lower().split()
    tweet_topic=[stopper.stem(word) for word in tweet_topic if (word not in stop_words) and (word not in punct)]
    tweet_topic=' '.join(tweet_topic)
    cleaned_data.append(tweet_topic)


# In[8]:


cleaned_data


# In[9]:


Variable_Y


# In[10]:


sentiment_ordering = ['negative', 'neutral', 'positive']
Variable_Y = Variable_Y.apply(lambda variable_x: sentiment_ordering.index(variable_x))


# In[11]:


Variable_Y.head()


# In[12]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=3000,stop_words=['virginamerica', 'unit'])
Variable_X_fin=cv.fit_transform(cleaned_data).toarray()
Variable_X_fin.shape


# In[13]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
model=MultinomialNB()


# In[14]:


Variable_X_train,Variable_X_test,Variable_Y_train,Variable_Y_test=train_test_split(Variable_X_fin,Variable_Y,test_size=0.3)


# In[15]:


model.fit(Variable_X_train,Variable_Y_train)


# In[16]:


Variable_Y_pred=model.predict(Variable_X_test)


# In[17]:


from sklearn.metrics import classification_report
cf=classification_report(Variable_Y_test,Variable_Y_pred)
print(cf)


# In[ ]:




