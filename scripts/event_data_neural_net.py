#!/usr/bin/python3
import sys
sys.dont_write_bytecode = True
import time
import string
import random
import numpy as np
import pandas as pd
from naive_bayes_model import NaiveBayesText
from tweet_scorer import TweetScorer


### Michael O'Malley, Michael Burke
### Machine Learning
### Semster Project
### Event Data Model


### Neural Net Secondary Model ###


class EventDataNeuralNet(NaiveBayesText):


	# Initialize class variables
	def __init__(self, datafile, n=10000, k=10, graded=False, model=1, max_iter=100):
		NaiveBayesText.__init__(self, datafile + '_graded_Tweets.txt', n, k)
		NaiveBayesText.readModel(self, 'n100000k10.txt')
		self.pos_dict, self.neg_dict = NaiveBayesText.getDicts(self)
		self.max_iter = max_iter

		# If data is not graded, score it
		if graded:
			self.data = pd.read_csv('../database/' + str(datafile) + '_graded_Tweets.txt', error_bad_lines=False, warn_bad_lines=False, sep='\t', names=['UserID','SentimentText','Time','Location', 'TweetID', 'Sentiment'], nrows=n)
		else:
			self.model = TweetScorer(datafile, 'n100000k10.txt', model=model)
			self.data = self.model.getData()
			self.tweet_num = len(self.data.index)

		self.resetMetrics()



	# Reset all the values
	def resetMetrics(self):
		self.weights = {}
		self.lr = .01
		NaiveBayesText.resetMetrics(self)


	# Train given DataFrame
	def trainModel(self, df, reset=False):
		
		# Reset values if indicated
		if reset:
			self.resetMetrics()
			df = self.data

		# Randomly assign weights to words
		for index, row in df.iterrows():

			# Add user id to weights
			if row['UserID'] not in self.weights:
				self.weights[row['UserID']] = random.uniform(-1, 1)
		
			# Split into words
			translation = {None:string.punctuation}
			words = row['SentimentText'].translate(translation).split()

			# Go word by word
			for word in words:
				if word not in self.pos_dict and word not in self.neg_dict:
					if word not in self.weights:
						self.weights[word] = random.uniform(-.001, .001)

		# Implement the delta rule
		iter_error = pd.DataFrame(columns=['x', 'y'])
		for loop in range(0,self.max_iter): 
		
			# Change in weight set to zero
			w_change = [0] * len(self.weights)
			part_error = float(0.0)
		
			# Compute change in weight per row of data
			for index, row in df.iterrows():
			    
				# Set target
				target = row['Sentiment']
				    
				# Find actual output
				output = self.predict(row['SentimentText'], row['UserID'])
				
				# Record partial errors
				part_error += (target-output)**2
				
				# Do nothing if output is correct, otherwise adjust weights
				if target == output:
					continue
				else:
					# Strip punctuation and split sentence
					translation = {None:string.punctuation}
					words = row['SentimentText'].translate(translation).split()

					# Go through every word
					for word in words:
						if word in self.weights:
							self.weights[word] += (self.lr*(target-output))

			# Record error
			iter_error.loc[len(iter_error)] = [loop, part_error/2]
		
			# If no error, break loop
			if (part_error / 2) == 0:
			    break


	# Make prediction for text
	def predict(self, sentence, usr=None):

		# Intitialize final score
		score = 0
		
		# Throw out non strings
		if not isinstance(sentence, str):
			return -1

		# Strip punctuation and split sentence
		translation = {None:string.punctuation}
		words = sentence.translate(translation).split()

		# Go through every word
		for word in words:
			if word in self.weights:
				if self.weights[word] > 0:
					score += 1
				else:
					score -= 1
			else:
				if word in self.pos_dict:
					score += 1
				if word in self.neg_dict:
					score -= 1

		# Add in user weight
		if usr in self.weights:
			#score *= self.weights[usr]
			if self.weights[usr]>0:
				score += 1
			else:
				score -= 1

		# If greater than 0, positive sentiment
		if score > 0:
			prediction = 1
		else:
			prediction = 0
		return prediction


	# Evaluate Naive Bayes algorithm with k-fold cross-validation
	def fullEval(self, output=True):

		# Split the data according to k
		self.splitData()

		# Set list for eval metrics
		acc_list = []
		pre_list = []
		rec_list = []
		f1_list = []

		# Loop through each dataframe
		for i in range(0, len(self.split_data)):

			# Train data
			if i == 0:
				self.trainModel(pd.concat(self.split_data[i+1:len(self.split_data)]))
			elif i == len(self.split_data)-1:
				self.trainModel(pd.concat(self.split_data[0:i]))
			else:
				self.trainModel(pd.concat(self.split_data[0:i]))
				self.trainModel(pd.concat(self.split_data[i+1:len(self.split_data)]))

			# Test data
			self.testModel(self.split_data[i])
		
			# Add to metric lists
			acc_list.append(self.accuracy)
			pre_list.append(self.precision)
			rec_list.append(self.recall)
			f1_list.append(self.f1)

			# Reset all values
			self.resetMetrics()

		# Print results
		self.accuracy = sum(acc_list)/len(acc_list)
		self.precision = sum(pre_list)/len(pre_list)
		self.recall = sum(rec_list)/len(rec_list)
		self.f1 = sum(f1_list)/len(f1_list)
		if output:
			NaiveBayesText.printEval(self)


	# Return compiled dictionaries
	def getWeights(self):
		return self.weights


# Usage function
def usage(program):
    print('Usage: {} keyword'.format(program))


### Main Execution ###


if __name__=="__main__":

	# Assert proper usage
	if len(sys.argv) != 2:
	        usage(sys.argv[0])
        	sys.exit(1)

	start_time = time.time()

	model = EventDataNeuralNet(str(sys.argv[1]), 100000, 10, False, 2, 50)
	model.fullEval()

	print("Runtime: %s seconds" % (time.time() - start_time))

