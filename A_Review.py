#!/usr/bin/env python
# coding: utf-8

# In[68]:


import requests   # Importing requests to extract content from a url
from bs4 import BeautifulSoup as bs # Beautifulsoup is for web scrapping...used to scrap specific content 
import pandas as pan
Total_reviews=[]


# In[69]:


SG_M52_reviews=[]


# In[70]:


for i in range (1,31):
    op=[]
    url ="https://www.amazon.in/Samsung-Storage-Snapdragon-sAMOLED-Display/product-reviews/B09CV6FJ62/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews"+str(i)
    response = requests.get(url)
    soup = bs(response.content,"html.parser")
    reviews = soup.findAll("span",attrs = {"class","a-size-base review-text review-text-content"})
    for i in range(len(reviews)):
        op.append(reviews[i].text)
    SG_M52_reviews = SG_M52_reviews+op


# In[71]:


SG_M52_reviews = list(set(SG_M52_reviews))


# In[72]:


SG_M52_reviews


# In[73]:


import pandas as pd
import numpy as np
import tweepy
import re 
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
wordnet = WordNetLemmatizer()
import re
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import requests
from bs4 import BeautifulSoup as bs
from selenium import webdriver


# In[74]:


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


# In[75]:


#Cleaning
txt_upd = ' '.join(SG_M52_reviews)


# In[76]:


txt_upd = re.sub("[^A-Za-z" "]+"," ",txt_upd).lower() #remove special character
txt_upd = re.sub("[0-9" "]+"," ",txt_upd).lower() #remove numbers
txt_upd = re.sub(r'^https?:\/\/.*[\r\n]*', '', txt_upd).lower() #remove hyperlink


# In[77]:


text_tokens = word_tokenize(txt_upd)


# In[78]:


tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]


# In[79]:


tf = TfidfVectorizer()
text_tf = tf.fit_transform(tokens_without_sw)


# In[80]:


feature_names = tf.get_feature_names()
dense = text_tf.todense()
denselist = dense.tolist()
df =pd.DataFrame(denselist, columns=feature_names)


# In[81]:


df


# In[82]:


word_list = ' '.join(df)


# In[83]:


wordcloud = WordCloud(background_color='black',
                      width=1800,
                      height=1400).generate(word_list)


# In[84]:


plt.imshow(wordcloud)


# In[85]:


#Sentimental Analysis


# In[86]:


with open("C:/Users/prate/Downloads/Assignment/Text Mining/positive-words.txt","r") as pw:
    positive_words = pw.read().split("\n")


# In[87]:


positive_words = positive_words[35:]


# In[88]:


with open("C:/Users/prate/Downloads/Assignment/Text Mining/negative-words.txt","r", encoding='latin-1') as nw:
    negative_words = nw.read().split("\n")


# In[89]:


negative_words = negative_words[35:]


# In[90]:


txt_neg_in_nw = ' '.join([word for word in df if word in negative_words])


# In[91]:


wordcloud_neg = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(txt_neg_in_nw)


# In[92]:


txt_pos_in_pw = ' '.join([word for word in df if word in positive_words])


# In[93]:


wordcloud_pos = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(txt_pos_in_pw)


# In[94]:


plt.imshow(wordcloud_neg)


# In[95]:


plt.imshow(wordcloud_pos)


# In[ ]:




