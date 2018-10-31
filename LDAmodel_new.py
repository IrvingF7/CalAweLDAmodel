
# coding: utf-8

# In[143]:



# In[144]:


# For output
import os
import pathlib
import csv


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
import matplotlib.pyplot as plt
import wordcloud # Package by Andreas Mueller

# General usage
import math


# In[148]:


import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)


# In[149]:


import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)


# In[145]:

from nltk.corpus import stopwords
stop_words = stopwords.words('English')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])


def process(dataName):

    num_of_representatives = 8  # The number of representative narrative you want to show for each topics.
    num_of_topics = 20  # The number of topics you want to generate from the data.

    # Prepare the output folder, according to the input data's language type
    if not os.path.exists('output'):
        os.mkdir('output')
    outputPath = 'output/' + dataName
    if not os.path.exists(outputPath):
        os.mkdir(outputPath)
    outputPath = outputPath + '/'

    # In[146]:

    # Prepare the input data's path, according to the input data's language type
    inputPath = 'input/' + dataName + '/'

    # In[147]:

    # In[150]:

    # In[151]:

    df = pd.read_json(inputPath + dataName + '_data.json', encoding='utf-8')

    # In[152]:

    # Convert to list
    # if df.X.values[0] == 'English':
    data = df.Narrative.values.tolist()
    # else:
    # data = df.translation.values.tolist()
    # Remove Emails
    data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]

    # Remove new line characters
    data = [re.sub('\s+', ' ', sent) for sent in data]

    # Remove single quotes
    data = [re.sub("\'", "", sent) for sent in data]

    #pprint(data[:1])

    # In[153]:

    def sent_to_words(sentences):
        for sentence in sentences:
            yield (gensim.utils.simple_preprocess(sentence.encode('utf-8'), deacc=True))  # deacc=True removes punctuations

    data_words = list(sent_to_words(data))


    # In[154]:

    # Build bigram and trigram
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)


    # In[155]:

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

    # In[156]:

    # remove stop words
    data_words_nostops = remove_stopwords(data_words)

    # getting bigrams
    data_words_bigrams = make_bigrams(data_words_nostops)

    nlp = spacy.load('en', disable=['parser', 'ner'])

    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])


    # In[157]:

    id2word = corpora.Dictionary(data_lemmatized)

    texts = data_lemmatized

    corpus = [id2word.doc2bow(text) for text in texts]


    # In[158]:

    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=num_of_topics,
                                                random_state=100,
                                                update_every=1,
                                                chunksize=100,
                                                passes=30,
                                                alpha='auto',
                                                per_word_topics=False)

    # In[159]:

    num_of_words_per_topic = 15
    topics = lda_model.print_topics(num_of_topics, num_of_words_per_topic)
    doc_lda = lda_model[corpus]
    #pprint(topics)

    # In[160]:

    #coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
    #coherence_lda = coherence_model_lda.get_coherence()
    #print('\nCoherence Score: ', coherence_lda)

    # In[161]:

    mallet_path = 'C:/mallet-2.0.8/mallet-2.0.8/bin/mallet'  # update this path
    ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_of_topics, id2word=id2word)

    # In[162]:

    # Compute Coherence Score
    coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=data_lemmatized, dictionary=id2word,
                                               coherence='c_v')
    #coherence_ldamallet = coherence_model_ldamallet.get_coherence()
    #print('\nCoherence Score: ', coherence_ldamallet)

    # In[163]:

    topics = ldamallet.print_topics(num_of_topics, num_of_words_per_topic)
    #pprint(topics)

    # In[164]:

    # Prepare the keywords and percentages for futrue useage
    # allKeywords[i] will give a array of keywords for topic i
    # allPercentages[i] will give a array of percentages for topic i
    index = 0
    chunks = [None] * num_of_topics
    allKeywords = [None] * num_of_topics
    allPercentages = [None] * num_of_topics
    for chunk in topics:
        chunk = chunk[1]
        percentages = re.findall(r"[-+]?\d*\.\d+|\d+", chunk)  # credit to miku on Stackoverflow
        keywords = re.findall('"([^"]*)"', chunk)  # credit to jspcal on Stackoverflow
        allKeywords[index] = keywords
        allPercentages[index] = percentages
        result = [None] * 2 * num_of_words_per_topic
        result[::2] = percentages
        result[1::2] = keywords
        result = [str(index)] + result
        chunks[index] = result  # A array stroing arrays of keywords and corresponding percentages
        index += 1

    # In[165]:


    # In[166]:

    header = [None] * 2 * num_of_words_per_topic

    for i in range(0, 2 * num_of_words_per_topic):
        if i % 2 == 0:
            header[i] = 'Percentage'
        else:
            header[i] = 'Keyword'

    header = ['Topic No.'] + header
    topicsPath = outputPath + dataName + '_' + 'topics.csv'
    with open(topicsPath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows([header])
        for chunk in chunks:
            writer.writerows([chunk])

    # In[167]:

    # For generating the most dominant topic for each narrative
    def format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data):
        sent_topics_df = pd.DataFrame()

        for i, row in enumerate(ldamodel[corpus]):
            row = sorted(row, key=lambda x: (x[1]), reverse=True)

            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:  # => dominant topic
                    wp = ldamodel.show_topic(topic_num)
                    topic_keywords = ", ".join([word for word, prop in wp])
                    sent_topics_df = sent_topics_df.append(
                        pd.Series([int(topic_num), round(prop_topic, 4), topic_keywords]), ignore_index=True)
                else:
                    break
        sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

        contents = pd.Series(texts)
        sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
        return (sent_topics_df)

    df_topic_sents_keywords = format_topics_sentences(ldamodel=ldamallet, corpus=corpus, texts=data)

    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic.columns = ['Narrative No.', 'Dominant_Topic', 'Topic Contribution', 'Keywords', 'Text']

    dominantTopicsPath = outputPath + dataName + '_' + 'dominant_topic.csv'
    df_dominant_topic.to_csv(dominantTopicsPath, encoding='utf-8', index=False)

    # In[168]:


    # In[169]:

    # For generating top 'num_of_representatives' most representative narratives
    import operator
    repre = [None] * num_of_topics
    for i, row in enumerate(ldamallet[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        for j, (topic_num, prop_topic) in enumerate(row):
            if repre[topic_num] is None:
                repre[topic_num] = dict()
            repre[topic_num][str(i)] = prop_topic
    sorted_repre = [None] * num_of_topics
    for i in range(0, len(repre)):
        sorted_repre[i] = sorted(repre[i].items(), key=operator.itemgetter(1), reverse=True)

    # pprint(sorted_repre)
    allTheTopics = [None] * num_of_topics

    index = 0
    for item in sorted_repre:
        allTheTopics[index] = [index]
        allTheTopics[index] += [allKeywords[index]]
        for i in range(0, num_of_representatives):
            allTheTopics[index] += [data[int(sorted_repre[index][i][0])]]
        index += 1
    representativeNarraPath = outputPath + dataName + '_' + 'representative_narratives.csv'
    header = (2 + num_of_representatives) * [None]
    header[0] = 'Topic No.'
    header[1] = 'Keywords'
    for i in range(1, num_of_representatives + 1):
        header[i + 1] = i
    with open(representativeNarraPath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows([header])
        for topic in allTheTopics:
            writer.writerows([topic])

    # In[170]:

    sent_topics_sorteddf_mallet = pd.DataFrame()

    sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')


    # In[171]:

    # For generating the frequency of each topic being the most dominant topic
    frequency = dict()  # A dictionary. The key is the string of topic number. The item is the number of the appearence of this topic
    for topicIndex in range(0, num_of_topics):
        frequency[str(topicIndex)] = sent_topics_outdf_grpd.get_group(float(topicIndex)).count().at['Dominant_Topic']

    total_appearance = 0  # The number of narratives that have a dominant topic
    for i in range(0, num_of_topics):
        total_appearance += frequency[str(i)]

    frequency_rows = [None] * num_of_topics
    for i in range(0, num_of_topics):
        temp = [None] * 4  # Each row always has 4 elements: Topic Number, Topic Keywords, Topic Appearance, Frequency
        float_index = float(i)  # The float version of the index for Dataframe Usage
        temp[0] = float_index
        temp[1] = sent_topics_outdf_grpd.get_group(float_index).iat[0, 2]
        # 0 here means that the first rows(each topic will be the most domimant topic for at least one narrative by the nature of LDA)
        # 2 here means we get the value at the third column, which is the Topic_Keywords
        temp[2] = frequency[str(i)]
        temp[3] = float(temp[2]) / float(total_appearance)
        frequency_rows[i] = temp

    frequencyPath = outputPath + dataName + '_' + 'frequency_topics.csv'  # Handle the output path

    # This part handles the header
    header = [None] * 4
    header[0] = "Topic No."
    header[1] = "Keywords"
    header[2] = "Appearance"
    header[3] = "Frequency"

    with open(frequencyPath, 'w', newline='', encoding = 'utf-8') as f:
        writer = csv.writer(f)
        writer.writerows([header])
        for frequency in frequency_rows:
            writer.writerows([frequency])

    # In[172]:

    for i, grp in sent_topics_outdf_grpd:
        sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet,
                                                 grp.sort_values(['Perc_Contribution'], ascending=[0]).head(1)],
                                                axis=0)

    sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)

    sent_topics_sorteddf_mallet.columns = ['Topic Index', "Topic Contribution", "Keywords", "Text"]


    # In[173]:

    fontpath = 'font/SFCompact/SFCompactDisplay-Light.otf'  # Use a local font.
    cloud = wordcloud.WordCloud(font_path=fontpath, width=700, height=600,
                                background_color=None, mode='RGBA', relative_scaling=0.5,
                                normalize_plurals=False)  # The object for generating wordcloud.

    # The folder that stores these visulization.
    imgPath = outputPath + 'visualizations'
    if not os.path.exists(imgPath):
        os.mkdir(imgPath)
    imgPath += '/'
    for topic in range(0, num_of_topics):
        cloudict = dict()
        for i in range(0, 15):
            cloudict[allKeywords[topic][i]] = float(
                allPercentages[topic][i])  # Generate the frequency for the cloud object to use.

        img = cloud.generate_from_frequencies(cloudict, max_font_size=None)  # Generate the image.
        img.to_file(imgPath + 'Topic' + ' ' + str(topic) + '.png')

    if (dataName is 'Overall'):
        topicsArray = [None] * num_of_topics  # An array that contains the distribution of countries for a topic.
        for i in range(len(data)):
            index = int(df_dominant_topic.iat[i, 1])
            if topicsArray[index] is None:
                topicsArray[index] = dict()
            country = df.iat[i, 0]
            if country in topicsArray[index]:
                topicsArray[index][country] += 1
            else:
                topicsArray[index][country] = 1

        header = [None]
        header[0] = 'Topic No.'
        header = header + countryList
        distributionRow = [None] * num_of_topics
        for i in range(len(topicsArray)):
            temp = [None] * len(header)
            temp[0] = float(i)
            j = 1
            for country in countryList:
                if country in topicsArray[i]:
                    temp[j] = topicsArray[i][country]
                else:
                    temp[j] = 0
                j += 1
            distributionRow[i] = temp

        distributionPath = outputPath + dataName + '_' + 'distribution.csv'
        with open(distributionPath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows([header])
            for distribution in distributionRow:
                writer.writerows([distribution])

countryList = ['Argentina','Australia','Austria','Brazil','Canada','Chile', 'China', 'France','Germany','India','Indonesia','Ireland','Japan','Korea','Mexico','Netherlands','Norway','Russia','Singapore','South_Africa','Spain','Sweden','Switzerland','Turkey','UK','USA']

#for country in countryList:
process('Overall')
#dataName = 'Brazil' # The name of the data file. Now it is just for different language.

