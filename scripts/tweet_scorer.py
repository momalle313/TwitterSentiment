#!/usr/bin/python
import sys
import time
import pandas as pd
from naive_bayes_model import NaiveBayesText


### Michael O'Malley, Michael Burke
### Machine Learning
### Semester Project
### Tweet Scoring script


### Twitter data cleaning and formatting


class TweetScorer:


	# Initialize class values
	def __init__(self, twitter_data, modelFile=None, train_data='Sentiment Analysis Dataset.csv', n=500000, k=10):
		self.data = data = pd.read_csv('../database/' + twitter_data, sep='\t', names=['UserID','SentimentText','Time','Location'])
		self.model = NaiveBayesText(train_data, n, k)
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
		loc = self.data.columns.get_loc('Sentiment')
		for index, row in self.data.iterrows():
			prediction =  self.model.predict(row['SentimentText'])
			
			# Throw out bad data
			if prediction == -1:
				drop_list.append(index)
				continue

			# Set column value to prediction
			self.data.iat[index,loc] = prediction
		
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
	
	TS = TweetScorer('trump_Tweets.txt', 'n100000k10.txt')
	data = TS.getData()
	print data.head(20)

	print("Runtime: %s seconds" % (time.time() - start_time))

