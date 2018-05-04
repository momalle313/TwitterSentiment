#!/usr/bin/python3
import io
import sys
sys.dont_write_bytecode = True
import time
import string
import random
import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from base_model import BaseModel


### Michael O'Malley, Michael Burke
### Machine Learning
### Semster Project
### Text Blob Model


### TextBlob training model ###


class VaderModel(BaseModel):


	# Initialize class values
	def __init__(self, datafile, n=500000, k=10):
		BaseModel.__init__(self, datafile, n, k)
		self.model = SentimentIntensityAnalyzer()


	# Make prediction for sentence
	def predict(self, sentence, usr=None):
		
		# Run string through text blob
		translation = {None:string.punctuation}
		score = self.model.polarity_scores(sentence.translate(translation))['compound']
		
		# Make prediction and return output
		if score >= 0:
			prediction = 1
		else:
			prediction = 0
		return prediction


	# Evaluate algorithm with k-fold cross-validation
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
			BaseModel.printEval(self)


### Testing ###


if __name__=="__main__":

	start_time = time.time()
	n = int(sys.argv[1])
	k = 10

	TB = VaderModel('Sentiment Analysis Dataset.csv', n, k)
	TB.fullEval()

	print("Runtime: %s seconds" % (time.time() - start_time))

	
