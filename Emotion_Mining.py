#!/usr/bin/env python
# coding: utf-8

# In[44]:


import numpy as np 
import pandas as pd
import string # special operations on strings
import spacy # language models

from matplotlib.pyplot import imread
from matplotlib import pyplot as plt
from wordcloud import WordCloud
get_ipython().run_line_magic('matplotlib', 'inline')


# In[45]:


import pandas
tweets=pd.read_csv('C:/Users/prate/Downloads/Assignment/Text Mining/Elon_musk.csv',error_bad_lines=False)


# In[46]:


tweets


# In[47]:


tweets.drop(tweets.filter(regex="Unname"),axis=1, inplace=True)


# In[48]:


tweets.head()


# In[49]:


tweets=[Text.strip() for Text in tweets.Text] # remove both the leading and the trailing characters
tweets=[Text for Text in tweets if Text] # removes empty strings, because they are considered in Python as False
tweets[0:10]


# In[50]:


##Part Of Speech Tagging
nlp = spacy.load("en_core_web_sm")

one_block = tweets[2]
doc_block = nlp(one_block)
spacy.displacy.render(doc_block, style='ent', jupyter=True)


# In[51]:


one_block


# In[52]:


for token in doc_block[0:20]:
    print(token, token.pos_)


# In[53]:


#Filtering for nouns and verbs only
nouns_verbs = [token.text for token in doc_block if token.pos_ in ('NOUN', 'VERB')]
print(nouns_verbs[5:25])


# In[54]:


#Counting tokens again
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

X = cv.fit_transform(nouns_verbs)
sum_words = X.sum(axis=0)
words_freq = [(word, sum_words[0, idx]) for word, idx in cv.vocabulary_.items()]
words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
wf_df = pd.DataFrame(words_freq)
wf_df.columns = ['word', 'count']

wf_df[0:10]


# In[55]:


##Visualizing results
#Barchart for top 10 nouns + verbs
wf_df[0:10].plot.bar(x='word', figsize=(12,8), title='Top verbs and nouns')


# #### Emotion Mining

# In[57]:


#Sentiment analysis
afinn = pd.read_csv('C:/Users/prate/OneDrive/Desktop/DS_ExLr _/Afinn.csv', sep=',', encoding='latin-1')
afinn.shape


# In[58]:


afinn.head()


# In[61]:


from nltk import tokenize
sentences = tokenize.sent_tokenize(" ".join(tweets))
sentences[5:15]


# In[62]:


sent_df = pd.DataFrame(sentences, columns=['sentence'])
sent_df


# In[63]:


affinity_scores = afinn.set_index('word')['value'].to_dict()


# In[65]:


#Custom function :score each word in a sentence in lemmatised form, 
#but calculate the score for the whole original sentence.
nlp = spacy.load("en_core_web_sm")
sentiment_lexicon = affinity_scores

def calculate_sentiment(text: str = None):
    sent_score = 0
    if text:
        sentence = nlp(text)
        for word in sentence:
            sent_score += sentiment_lexicon.get(word.lemma_, 0)
    return sent_score


# In[71]:


# test that it works
calculate_sentiment(text = 'good')


# In[72]:


sent_df['sentiment_value'] = sent_df['sentence'].apply(calculate_sentiment)


# In[73]:


# how many words are in the sentence?
sent_df['word_count'] = sent_df['sentence'].str.split().apply(len)
sent_df['word_count'].head(10)


# In[74]:


sent_df.sort_values(by='sentiment_value').tail(10)


# In[75]:


# Sentiment score of the whole review
sent_df['sentiment_value'].describe()


# In[76]:


# Sentiment score of the whole review
sent_df[sent_df['sentiment_value']<=0].head()


# In[78]:


sent_df[sent_df['sentiment_value']<=20].head()


# In[79]:


sent_df['index']=range(0,len(sent_df))


# In[80]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.distplot(sent_df['sentiment_value'])


# In[81]:


plt.figure(figsize=(15, 10))
sns.lineplot(y='sentiment_value',x='index',data=sent_df)


# In[82]:


sent_df.plot.scatter(x='word_count', y='sentiment_value', figsize=(8,8), title='Sentence sentiment value to sentence word count')


# In[ ]:




