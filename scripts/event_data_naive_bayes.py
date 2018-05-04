#!/usr/bin/python3
import sys
sys.dont_write_bytecode = True
import time
import string
import numpy as np
import pandas as pd
from naive_bayes_model import NaiveBayesText
from tweet_scorer import TweetScorer


### Michael O'Malley, Michael Burke
### Machine Learning
### Semster Project
### Event Data Model


### Naive Bayes Secondary Model ###


class EventDataNaiveBayes(NaiveBayesText):


	# Initialize class variables
	def __init__(self, datafile, n=10000, k=10, graded=False, model=1):
		NaiveBayesText.__init__(self, datafile + '_graded_Tweets.txt', n, k)
		NaiveBayesText.readModel(self, 'n100000k10.txt')
		self.pos_dict, self.neg_dict = NaiveBayesText.getDicts(self)

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
		self.pos_usr = {}
		self.neg_usr = {}
		NaiveBayesText.resetMetrics(self)


	# Train given DataFrame
	def trainModel(self, df, reset=False):
		
		# Reset values if indicated
		if reset:
			self.resetMetrics()
			df = self.data

		# Count words unique to the event
		for index, row in df.iterrows():

			# Add users to dicts
			if row['Sentiment'] == 1:
				self.tot_pos += 1
				if row['UserID'] in self.pos_usr:
					self.pos_usr[row['UserID']] += 1
				else:
					self.pos_usr[row['UserID']] = 1
			else:
				self.tot_neg += 1
				if row['UserID'] in self.neg_usr:
					self.neg_usr[row['UserID']] += 1
				else:
					self.neg_usr[row['UserID']] = 1
		
			# Split into words
			translation = {None:string.punctuation}
			words = row['SentimentText'].translate(translation).split()

			# Go word by word
			for word in words:

				# Add word to total count
				self.tot_word += 1

				# If word is positive and unique to event, record it
				if row['Sentiment'] == 1:
					if word not in self.pos_dict:
						self.tot_pos+=1
						if word in self.pos:
							self.pos[word] += 1
						else:
							self.pos[word] = 1

				# If word is negative and unique to event, record it
				else:
					if word not in self.neg_dict:
						self.tot_neg+=1
						if word in self.neg:
							self.neg[word] += 1
						else:
							self.neg[word] = 1

		# Fix zero-frequency problem
		self.pos = {key:value+1 for key, value in self.pos.items()}
		self.neg = {key:value+1 for key, value in self.neg.items()}
		self.pos_usr = {key:value+1 for key, value in self.pos_usr.items()}
		self.neg_usr = {key:value+1 for key, value in self.neg_usr.items()}
		for key in self.pos.keys():
			if key not in self.neg:
				self.neg[key] = 1
		for key in self.neg.keys():
			if key not in self.pos:
				self.pos[key] = 1
		for key in self.pos_usr.keys():
			if key not in self.neg_usr:
				self.neg_usr[key] = 1
		for key in self.neg_usr.keys():
			if key not in self.pos_usr:
				self.pos_usr[key] = 1

		# Normalize totals so ratios don't get too small
		self.tot_neg = self.tot_neg/(self.tweet_num/float(10))
		self.tot_pos = self.tot_pos/(self.tweet_num/float(10))
		self.tot_word = self.tot_word/(self.tweet_num/float(10))

		# Calculate word ratios
		self.pos_ratio = self.tot_pos/self.tot_word
		self.neg_ratio = self.tot_neg/self.tot_word
		self.pos = {key:(self.pos[key]/self.tot_pos) for key in self.pos.keys()}
		self.neg = {key:(self.neg[key]/self.tot_neg) for key in self.neg.keys()}
		self.pos_usr = {key:(self.pos_usr[key]/self.tot_pos) for key in self.pos_usr.keys()}
		self.neg_usr = {key:(self.neg_usr[key]/self.tot_neg) for key in self.neg_usr.keys()}


	# Make prediction for text
	def predict(self, sentence, usr=None):
		
		# Set scores to 1 
		pos_score = 1
		neg_score = 1

		# Throw out non strings
		if not isinstance(sentence, str):
			return -1

		# Strip punctuation and split sentence
		translation = {None:string.punctuation}
		words = sentence.translate(translation).split()

		# Go through every word
		for word in words:
			
			# Multiply pos and neg ratios to scores
			if word in self.pos:
				pos_score *= self.pos[word]
			if word in self.neg:
				neg_score *= self.neg[word]
			
			if word in self.pos_dict:
				pos_score *= self.pos_dict[word]
			if word in self.neg_dict:
				neg_score *= self.neg_dict[word]

		# Add in user score
		if usr in self.pos_usr:
			pos_score *= self.pos_usr[usr]
		if usr in self.neg_usr:
			neg_score *= self.neg_usr[usr]
				
		# Include overall ratios
		pos_score *= self.pos_ratio
		neg_score *= self.neg_ratio
		
		# Normalize
		div = pos_score+neg_score
		if div == 0:
			div = 1
		pos_score = pos_score/div
		neg_score = neg_score/div

		# Make prediction and return output
		if pos_score > neg_score:
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
	def getDicts(self):
		return self.pos, self.neg


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

	model = EventDataNaiveBayes(str(sys.argv[1]), 100000, 10, False, 1)
	model.fullEval()

	print("Runtime: %s seconds" % (time.time() - start_time))

