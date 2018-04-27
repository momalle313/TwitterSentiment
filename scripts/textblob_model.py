#!/usr/bin/python
import sys
import time
import string
import random
import numpy as np
import pandas as pd
from textblob import TextBlob
from base_model import BaseModel


### Michael O'Malley, Michael Burke
### Machine Learning
### Semster Project
### Text Blob Model


### TextBlob training model ###


class TextBlobModel(BaseModel):


	# Initialize class values
	def __init__(self, datafile, n=500000, k=10):
		BaseModel.__init__(self, datafile, n, k)


	# Make prediction for sentence
	def predict(self, sentence):
		
		# Run string through text blob
		score = TextBlob(sentence.translate(None,string.punctuation).decode('ascii', 'ignore')).sentiment[0]
		
		# Make prediction and return output
		if score >= 0:
			prediction = 1
		else:
			prediction = 0
		return prediction


	# Evaluate algorithm with k-fold cross-validation
	def fullEval(self):

		# Split the data according to k
		self.splitData()

		# Set list for eval metrics
		acc_list = []
		pre_list = []
		rec_list = []
		f1_list = []

		# Loop through each dataframe
		for i in range(0, len(self.split_data)):

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
		self.printEval()


	# Print evaluation metrics
	def printEval(self):
		print "NaiveBayesText Evaluation"
		print "Number of Tweets: %d" % self.tweet_num
		print "Acurracy: %.2f" % self.accuracy
		print "Precision: %.2f" % self.precision
		print "Recall: %.2f" % self.recall
		print "F1: %.2f\n" % self.f1


### Testing ###


if __name__=="__main__":

	start_time = time.time()
	n = int(sys.argv[1])
	k = 10

	TB = TextBlobModel('Sentiment Analysis Dataset.csv', n, k)
	TB.fullEval()

	print("Runtime: %s seconds" % (time.time() - start_time))

	
