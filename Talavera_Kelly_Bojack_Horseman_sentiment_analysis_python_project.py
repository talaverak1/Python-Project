# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 22:05:19 2020

@author: kma5
"""

import nltk #Make sure you are using a 64-bit system or else nltk will not work. The next two lines will update the relevant modules.
nltk.download('vader_lexicon') #Calculates negative, positive, and neutral values for the text.
nltk.download('punkt') #Word tokenizer, which will split the csv file into sentences or words.

# import the relevant modules from the NLTK libraries
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import sentiment
from nltk import word_tokenize

sid = SentimentIntensityAnalyzer() #VADER has to be initialized so it can be used it within the Python script.

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle') #Initialize 'english.pickle' word tokenizer function and give it a short name, 'tokenizer'.

f = open(r'C:\Users\kma5\Desktop\CleanSubtitles\ULTIMATEFINAL.csv') #Open up the csv file. You may need to specify the path. 

horseman = f.read() #Designate the file as 'horseman.'
         
sentences = tokenizer.tokenize(horseman) #Tell the tokenizer to break up the csv file, 'horseman,' into a list of sentences.

#Find all sentences in the csv file that include a specific keyword and designate these sentences as a list. The "*"s are wildcards which match everything before and after the word itself. 
import re

r = re.compile(".* word .*")
wordlist = list(filter(r.match, sentences))
print(wordlist[:10]) #The last part of the code prints the first ten sentences in the list, so we can see that it worked.

#Run the sentiment analysis on those sentences that include the word.
for sentence in sentences:
    print(sentence)
    scores = sid.polarity_scores(sentence)
    for key in sorted(scores):
       print('{0}:{1}, '.format(key,scores[key]), end='')
       print()
   

#Once we hit "Run," we can see the calculated positive, negative, and neutral scores for each sentence in the text source.        
#The information output has limited value, and we require a deeper analysis of the text of the show. 
#The output highlights the limitations of computer language processing, where ambigous words are difficult to train.
#Once a computer program records a value to a word, that value is set. Mapping context to a word or phrase is likely doable, but outside the scope of this project. 
#Analyzing the data using TextBlob, a different natural language processor, can give us some further insight into how the show uses language and has additional graphing applications. 
#First, we import some data and graphing modules we need to analyze the script.
import string
import numpy as np	
import pandas as pd
import matplotlib.pyplot as plt
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer, PorterStemmer
from wordcloud import WordCloud, STOPWORDS
from textblob import TextBlob
from itertools import islice
import seaborn as sns

# In order to manage the data, we will create a new file that will allow us to manipulate data based on sentiment, subjectivity and polarity. 	
bjscript = pd.read_csv(r"C:\Users\kma5\Desktop\CleanSubtitles\ULTIMATEFINAL.csv")

COLS = ['Character', 'Text', 'sentiment', 'subjectivity', 'polarity']

df = pd.DataFrame(columns=COLS)

#Here, we add the information for each row in the script which gives us a file with analyzable data. 
for index, row in islice(bjscript.iterrows(), 0, None): 
    
    new_entry = []
    text_lower = str(row['TEXT'])
    blob = TextBlob(text_lower)
    sentiment = blob.sentiment
    
    polarity = sentiment.polarity
    subjectivity = sentiment.subjectivity 
    
    new_entry += [row['TEXT'], text_lower, sentiment, subjectivity, polarity]
    
    line_sentiment = pd.DataFrame([new_entry], columns=COLS)
    df = df.append(line_sentiment, ignore_index=True)

df.to_csv('BJ_Script_Sentiment_Analysis.csv', mode = 'w', columns=COLS, index=False, encoding="utf-8" )

#Having too many neutral scores can skew the data and makes certain conclusions more difficult. 
dffilter = df.loc[(df.loc[:, df.dtypes != object] != 0).any(1)]
dffilter.describe()
# Here we create a boxplot tracking the polarity of the sentiment scores. 

boxplot = dffilter.boxplot(column=['subjectivity', 'polarity'], fontsize = 15, grid = True, vert = True, figsize = (10,10,))

plt.ylabel('Range')

plt.show()

#Here we can create a scatterplot to attempt to see if there is any consitency in the data. 

sns.lmplot(x='subjectivity', y= 'polarity', data=dffilter, fit_reg = True, height =10, palette= "mute" )

plt.show()

# To get a sense of whether the show leans positive or negative, we can use a covariance matrix. 
from numpy.random import randn
from numpy.random import seed
from numpy import cov
from scipy.stats import pearsonr

# prepare data
data1 = dffilter['subjectivity']
data2 = data1 + dffilter['polarity']
# calculate covariance matrix
covariance = cov(data1, data2) 
print()
print(covariance)

corr, _ = pearsonr(data1, data2)
print('Pearsons correlation: %.5f' % corr)
#Values above .5 are considered positive. Since the show has a value close to .5, the show is slightly positive. 
#This is likely due to the heavy nature of the show as well as the dramatic language that the show uses. 


# A polarity distribution chart gives us an indication that most of the scored sentences are closer to neutral, with a spike at .5 indictating a strong positive. 
plt.hist(dffilter['polarity'], color = 'darkred', edgecolor = 'black', density=False,
         bins = int(30))
plt.title('Polarity Distribution')
plt.xlabel("Polarity")
plt.ylabel("Number of Times")
plt.show()

from pylab import rcParams
rcParams['figure.figsize'] = 10,15


# The density chart of the subjectivity shows a binomial distribution of subjectivity at .5 and 1. This is likely due to the simplicity of some of the dialouge, increasing the modules ability to determine the sentiment. 
sns.distplot(dffilter['subjectivity'], hist=True, kde=True, 
             bins=int(30), color = 'darkred',
             hist_kws={'edgecolor':'black'},axlabel ='Subjectivity')
plt.title('Subjectivity Density')
plt.show()


rcParams['figure.figsize'] = 10,15

#Using a stopwards corpus, we can eliminate stop words to simplify text and analyze the most commonly used words in the text.
stopwords = nltk.corpus.stopwords.words('english')


RE_stopwords = r'\b(?:{})\b'.format('|'.join(stopwords))
words = (df.Text
           .str.lower()
           .replace([r'\|', r'\&', r'\-', r'\.', r'\,', r'\'', r'\?', RE_stopwords], [' ', '', '','','','','',''], regex=True)
           .str.cat(sep=' ')
           .split()
)



from collections import Counter

# We can use a df as a counter to keep track of the most common words. 
rslt = pd.DataFrame(Counter(words).most_common(10),
                    columns=['Word', 'Frequency']).set_index('Word')
rslt

rslt_wordcloud = pd.DataFrame(Counter(words).most_common(100),
                    columns=['Word', 'Frequency'])

#BAR CHART of the most common words excluding stopwords and proper nouns.
rslt.plot.bar(rot=40, figsize=(16,10), width=0.8,colormap='tab10')
plt.title("Most Common Words in BoJack Script")
plt.ylabel("Count")
plt.show()


rcParams['figure.figsize'] = 10,15

#PIE CHART percentage of appearance of the top 10 common words as compared to all common words. 

explode = (0.1, 0.12, 0.122, 0,0,0,0,0,0,0)  # explode 1st slice
labels=["I'm",
        'dont',
        'know',
        'Oh',
        'like',
        'get',
        'youre',
        'right',
        'okay',
        'thats',]

plt.pie(rslt['Frequency'], explode=explode,labels =labels , autopct='%1.1f%%',
        shadow=False, startangle=90)
plt.legend( labels, loc='upper left',fontsize='x-small',markerfirst = True)
plt.tight_layout()
plt.title(' Ambigous Words and their frequency')
plt.show()


import matplotlib as mpl
mpl.rcParams['font.size'] = 15.0

from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import random

#Finally, a wordcloud to represent the 100 most frequent words whose size is based on their usage. 
#This wordcloud includes the normally excluded stopwords and proper nouns excluded in the previous part. 
wordcloud = WordCloud(max_font_size=60, max_words=100, width=480, height=380,colormap="brg",
                      background_color="white").generate(' '.join(rslt_wordcloud['Word']))
                      
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.figure(figsize=[10,10])
plt.show()




