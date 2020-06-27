# Python-Project
Sentiment Analysis on Bojack Horseman

Goal

This project aims to complete a sentiment analysis of the Netflix show, Bojack Horseman. 
The show revolves around the use of dark humor and tackles difficult subjects such as depression and addiction. 
As a result, the language used in the show should reflect the sentiments of the characters. 
The code I created attempts to use both nltk, TextBlob and other data tools to do a sentiment analysis of the show's script. 

Instructions

This code requires the installation of several modules, including: nltk, re, string, numpy, pandas, matplotlib, wordcloud, textblob, itertools, and seaborn. 
All these files can be installed through the command prompt using "pip install [module name]." 
If there are any problems with the installation, please refer to the installation guides for the individual modules. 
I recommend using anaconda to install and spyder to run the code. 

The code requires the download of the "ULTIMATEFINAL.csv" file. This file is the clean subtitles obtained from Netflix. 
There is an area in the code that requires you to provide a path for the csv file. Please insert the path where indicated. 

Dataset

The data used in this project was downloaded from netflix itself. Using a commandprompt and online tutorials, I was able to download raw XML files from the netflix site. 
These files were converted to txt and combined into a single file. Using a code included in the documentation, I was able to 
remove xml markings from the subtitles and take the raw text. 

Originally, the project was intended to do a character analysis for each  individual character. Howerver, due to time constraints, this became impossible to complete. 
Characters are added for most lines, however, certain lines in the text are delimited with  "--:" as a character instead. 

Data was converted to a csv file using excel to delete extra lines. 

Given additional time, the data would be divided by Season and Episode and a histogram analyzing changes through time and character would be possible. 
As it stands, the data still requires additional work to reach this stage. 

The code 

The main classes and functions serve several different purposes. 
The first function conducts a search for a certain word in the text and shows all instances of the text as well as the nltk sentiment analysis. 
The next function conducts the analysks for each line. 
The next part of the module revolves around the creation of data and analysis using TextBlob and sentiment analysis. 
Many of the functions here serve to analyze the data and present different plots depending on the needs of the data. 
Conclusions, descriptions, and functionality are included within the documentation of the code. 

Much of the code base was obtained from Sarah McTavish at cloud.archivesunleashed.org/derivatives/text-sentiment, as well as
Dilan Jayasekara at https://towardsdatascience.com/statistical-sentiment-analysis-for-survey-data-using-python-9c824ef0c9b0
