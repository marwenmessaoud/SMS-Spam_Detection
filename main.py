import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import operator
import warnings
from wordcloud import WordCloud
warnings.filterwarnings('ignore')
from visualisation import *


data = pd.read_csv("spam.csv",encoding='latin-1')
#print(data.head())


def preprocess(data) :
	# drop inefficent columns and change names of the useful columns
	data = data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
	data = data.rename(columns={"v1":"label", "v2":"text"})

	# convert label to numerical values
	data["label_num"] = data.label.map({"ham":0,"spam":1})


	# split data into train and test
	X_train,X_test,y_train,y_test = train_test_split(data["text"],data["label"], 
		test_size = 0.2, random_state = 10)

	original_x_test = X_test
	# transformation of text
	vect = CountVectorizer()
	vect.fit(X_train)
	X_train_df = vect.transform(X_train)
	X_test_df = vect.transform(X_test)

	return(X_train_df, y_train, X_test_df, y_test, original_x_test)



def ML_model(model, X_train_df, y_train, X_test_df, y_test) :
	model.fit(X_train_df,y_train)
	prediction = model.predict(X_test_df)
	#print("type of prediction : ", len(prediction))
	acc_score = accuracy_score(y_test,prediction)
	return(prediction, acc_score)

def compare_eval(models, X_train_df, y_train, X_test_df, y_test) :
	acc_scores = dict()
	predictions = dict()
	for model in models : 
		prediction, acc_score = ML_model(model,  X_train_df, y_train, X_test_df, y_test)
		acc_scores[model.__class__.__name__] = acc_score
		predictions[model.__class__.__name__] = prediction
		print("accuracy of {} : {}".format(model.__class__.__name__, acc_score))

	best_model = max(acc_scores.items(), key=operator.itemgetter(1))[0]
	#print("the best model is : {} with accuracy {}".format(best_model, acc_scores[best_model]))
	return(predictions, acc_scores, best_model, acc_scores[best_model])

def csv_predictions(predictions, test_x, test_y, file_name) : 
	new_df = pd.DataFrame()
	new_df["mail_text"] = test_x
	new_df["original_label"] = test_y
	for key, item in predictions.items():
		new_df[key] = item

	new_df.to_csv(file_name)

def main(data):

	# preprocess data
	X_train_df, y_train, X_test_df, y_test, original_x_test = preprocess(data)

	# visualisation 
	create_worcloud_visualisation(data)

	# compare performance of diffrents 
	models = [MultinomialNB(), LogisticRegression(), KNeighborsClassifier(n_neighbors=5), 
	RandomForestClassifier(), AdaBoostClassifier()]

	predictions, acc_scores, best_model, best_acc = compare_eval(models, X_train_df, y_train, X_test_df, y_test)
	csv_predictions(predictions, original_x_test, y_test, "predictions_file.csv")
	print("the best model is : {} with accuracy {}".format(best_model, best_acc))
	bar_plot(acc_scores)

main(data)