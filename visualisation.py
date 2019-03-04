import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import warnings
from wordcloud import WordCloud

def create_worcloud_visualisation(data) :
	data = data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
	data = data.rename(columns={"v1":"label", "v2":"text"})
	data["label_num"] = data.label.map({"ham":0,"spam":1})
	ham_words = ''
	spam_words = ''
	spam = data[data.label_num == 1]
	ham = data[data.label_num ==0]

	for val in spam.text:
	    text = val.lower()
	    tokens = nltk.word_tokenize(text)
	    #tokens = [word for word in tokens if word not in stopwords.words('english')]
	    for words in tokens:
	        spam_words = spam_words + words + ' '
	        
	for val in ham.text:
	    text = val.lower()
	    tokens = nltk.word_tokenize(text)
	    for words in tokens:
	        ham_words = ham_words + words + ' '

	# Generate a word cloud image
	spam_wordcloud = WordCloud(width=600, height=400).generate(spam_words)
	ham_wordcloud = WordCloud(width=600, height=400).generate(ham_words)

	#Spam Word cloud
	plt.figure( figsize=(10,8), facecolor='k')
	plt.imshow(spam_wordcloud)
	plt.title("spam worcloud")
	plt.axis("off")
	plt.tight_layout(pad=0)
	plt.show()


	#Ham word cloud
	plt.figure( figsize=(10,8), facecolor='k')
	plt.imshow(ham_wordcloud)
	plt.title("ham worcloud")
	plt.axis("off")
	plt.tight_layout(pad=0)
	plt.show()

def bar_plot(dict_acc):
	plt.bar(np.arange(len(dict_acc)), dict_acc.values(), align='center', alpha=0.5)
	plt.xticks(np.arange(len(dict_acc)), dict_acc.keys())
	plt.ylabel('Accuracy score')
	plt.title('Distribution by classifier')
	plt.show()