#!/usr/bin/python3
import sys
sys.dont_write_bytecode = True
import time
import string
import pandas as pd
from event_data_naive_bayes import EventDataNaiveBayes
from event_data_neural_net import EventDataNeuralNet


### Michael O'Malley, Michael Burke
### Machine Learning
### Semester Project
### Tweet Scoring script


### Scored tweets according to given model ###


class TweetScorerV2:


	# Initialize class values
	def __init__(self, keyword, n=1000, train_n=500000, k=10, model=1):
		self.data = pd.read_csv('../database/' + keyword + '_Tweets.txt', sep='\t', names=['UserID','SentimentText','Time','Location', 'TweetID'], nrows=n)
		self.data = self.data.dropna()
		for i in range(0, len(self.data.index)):
			translation = {None:string.punctuation}
			self.data.iat[i, 1] = self.data.iat[i, 1].translate(translation)
		if model == 1:
			self.model = EventDataNaiveBayes(keyword, train_n, k, False, 1)
		else:
			self.model = EventDataNeuralNet(keyword, train_n, k, False, 1, 50)


	# Records scores of event based tweets based on model
	def scoreTweets(self):
		
		# Train model
		self.model.trainModel(self.data, reset=True)

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
		return self.model.getDicts()


### Testing ###


if __name__ == "__main__":

	start_time = time.time()
	
	TS = TweetScorerV2(str(sys.argv[1]))
	data = TS.getData()
	print(data.head(5))

	print("Runtime: %s seconds" % (time.time() - start_time))

