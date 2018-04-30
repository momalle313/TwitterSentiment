#!/usr/bin/python
import sys
import time
import string
import pandas as pd
from naive_bayes_model import NaiveBayesText
from textblob_model import TextBlobModel


### Michael O'Malley, Michael Burke
### Machine Learning
### Semester Project
### Tweet Scoring script


### Twitter data cleaning and formatting


class TweetScorer:


	# Initialize class values
	def __init__(self, keyword, modelFile=None, train_data='Sentiment Analysis Dataset.csv', n=1000, train_n=500000, k=10, naive_bayes=True):
		self.data = data = pd.read_csv('../database/' + keyword + '_Tweets.txt', sep='\t', names=['UserID','SentimentText','Time','Location', 'TweetID'], nrows=n)
		self.data = self.data.dropna()
		for i in range(0, len(self.data.index)):
			self.data.iat[i, 1] = self.data.iat[i, 1].translate(None,string.punctuation)
		if naive_bayes:
			self.model = NaiveBayesText(train_data, train_n, k)
		else:
			self.model = TextBlobModel(train_data, train_n, k)
		self.modelFile = modelFile


	# Train NaiveBayesText with new data or data from file
	def trainModel(self):
		
		# If modelFile is given, load that
		if self.modelFile != None:
			self.model.readModel(self.modelFile)

		# If not, use default values
		else:
			self.model.trainModel(reset=True)


	# Records scores of event based tweets based on model
	def scoreTweets(self):
		
		# Train model
		self.trainModel()

		# Add blank new column to database
		size = len(self.data.index)
		scores = [-1] * size
		self.data['Sentiment'] = scores

		# Set up row drop list
		drop_list = []

		# Score each tweet according to model prediction
		loc1 = self.data.columns.get_loc('SentimentText')
		loc2 = self.data.columns.get_loc('Sentiment')
		for i in range(0, len(self.data.index)):

			prediction =  self.model.predict(self.data.iat[i, loc1])
			
			# Throw out bad data
			if prediction == -1:
				drop_list.append(index)
				continue

			# Set column value to prediction
			self.data.iat[i,loc2] = prediction
		
		# Drop bad data
		self.data = self.data.drop(self.data.index[drop_list])


	# Returns scored data
	def getData(self):
		self.scoreTweets()
		return self.data

	
	# Returns dicts used in model
	def getDicts(self):
		return self.model.returnDicts()


### Testing ###


if __name__ == "__main__":

	start_time = time.time()
	
	TS = TweetScorer(str(sys.argv[1]), 'n100000k10.txt')
	data = TS.getData()
	print data.head(20)

	print("Runtime: %s seconds" % (time.time() - start_time))

