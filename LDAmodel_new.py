
# coding: utf-8

# In[1]:


import re
import numpy as np
import pandas as pd
from pprint import pprint
import nltk; 
nltk.download('stopwords')

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  
import matplotlib.pyplot as plt


# In[2]:


#get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)


# In[4]:


import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)


# In[5]:


from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])


# In[6]:


df = pd.read_json('English_data.json')
print(df.Narrative.unique())
df.head()


# In[7]:


# Convert to list
data = df.Narrative.values.tolist()

# Remove Emails
data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]

# Remove new line characters
data = [re.sub('\s+', ' ', sent) for sent in data]

# Remove single quotes
data = [re.sub("\'", "", sent) for sent in data]

pprint(data[:1])


# In[8]:


def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(sentence.encode('utf-8'), deacc=True))  # deacc=True removes punctuations

data_words = list(sent_to_words(data))

print(data_words[:1])


# In[9]:


# Build bigram and trigram
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

print(trigram_mod[bigram_mod[data_words[0]]])


# In[10]:


def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


# In[11]:


#remove stop words
data_words_nostops = remove_stopwords(data_words)

#getting bigrams
data_words_bigrams = make_bigrams(data_words_nostops)


nlp = spacy.load('en', disable=['parser', 'ner'])

data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

print(data_lemmatized[:1])


# In[12]:


id2word = corpora.Dictionary(data_lemmatized)

texts = data_lemmatized

corpus = [id2word.doc2bow(text) for text in texts]

print(corpus[:1])


# In[13]:


lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=50, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=30,
                                           alpha='auto',
                                           per_word_topics=False)


# In[14]:


num_of_topics = 50
num_of_words_per_topic = 15
topics = lda_model.print_topics(num_of_topics, num_of_words_per_topic)
doc_lda = lda_model[corpus]
pprint(topics)


# In[15]:


#coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
#coherence_lda = coherence_model_lda.get_coherence()
#print('\nCoherence Score: ', coherence_lda)


# In[16]:


mallet_path = 'C:/mallet-2.0.8/mallet-2.0.8/bin/mallet' # update this path
ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=50, id2word=id2word)


# In[17]:


# Compute Coherence Score
#coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
#coherence_ldamallet = coherence_model_ldamallet.get_coherence()
#print('\nCoherence Score: ', coherence_ldamallet)


# In[18]:


topics = ldamallet.print_topics(num_of_topics, num_of_words_per_topic)
pprint(topics)


# In[19]:


index = 0
chunks = [None] * num_of_topics
for chunk in topics:
    chunk = chunk[1]
    percentages = re.findall(r"[-+]?\d*\.\d+|\d+", chunk) #credit to miku on Stackoverflow
    keywords = re.findall('"([^"]*)"', chunk) #credit to jspcal on Stackoverflow
    result = [None] * 2 * num_of_words_per_topic
    result[::2] = percentages
    result[1::2] = keywords
    result = [str(index)] + result
    chunks[index] = result # A array stroing arrays of keywords and corresponding percentages
    index += 1
print(chunks)
    


# In[20]:


import csv
#header = [None] * 2 * num_of_words_per_topic

#for i in range(0, 2 * num_of_words_per_topic):
    #if i % 2 == 0:
               #header[i] = 'Percentage'
    #else:
               #header[i] = 'Keyword'

#header = ['Topic No.'] + header
#with open('topics.csv', 'w', newline='') as f:
    #writer = csv.writer(f)
    #writer.writerows([header])
    #for chunk in chunks:
        #writer.writerows([chunk])


# In[21]:


#pyLDAvis.enable_notebook()
#vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
#vis


# In[22]:


def format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data):
    sent_topics_df = pd.DataFrame()

    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)

        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


df_topic_sents_keywords = format_topics_sentences(ldamodel=ldamallet, corpus=corpus, texts=data)

df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Index', 'Dominant Topic', 'Topic Contribution', 'Keywords', 'Text']

#df_dominant_topic.to_csv('dominant_topic.csv')


# In[23]:


import operator
repre = [None] * 50
for i, row in enumerate(lda_model[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        for j, (topic_num, prop_topic) in enumerate(row):
            if repre[topic_num] is None:
                repre[topic_num] = dict()
            repre[topic_num][str(i)] = prop_topic
sorted_repre = [None] * 50
for i in range(0, len(repre)):
    sorted_repre[i] = sorted(repre[i].items(), key=operator.itemgetter(1), reverse = True)

#pprint(sorted_repre)
allTheTopics = [None] * 50
index = 0
for item in sorted_repre:
    allTheTopics[index] = [index]
    for i in range(0, 8):
        allTheTopics[index] += [data[int(sorted_repre[index][i][0])]]
    index += 1
print(allTheTopics)
with open('representative_narrative.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for topic in allTheTopics:
        writer.writerows([topic])


# In[24]:


sent_topics_sorteddf_mallet = pd.DataFrame()

sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

sent_topics_outdf_grpd.head()


# In[25]:



for i, grp in sent_topics_outdf_grpd:
    sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet, 
                                             grp.sort_values(['Perc_Contribution'], ascending=[0]).head(1)], 
                                            axis=0)
   
sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)

sent_topics_sorteddf_mallet.columns = ['Topic Index', "Topic Contribution", "Keywords", "Text"]

sent_topics_sorteddf_mallet.head(10)


# In[27]:




