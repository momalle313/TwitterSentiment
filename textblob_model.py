#!/usr/bin/python
import sys
import time
import string
import random
import numpy as np
import pandas as pd
from textblob import TextBlob


### Michael O'Malley, Michael Burke
### Machine Learning
### Semster Project
### Text Blob Model


### TextBlob training model ###


class TextBlobModel:


	# Initialize class values
	def __init__(self, datafile, n=500000, k=10):
		self.data = pd.read_csv('database/' + str(datafile), error_bad_lines = False, warn_bad_lines=False, nrows=n)
		self.tweet_num = n
		self.k_folds = k
		self.resetValues()


	# Reset all the values
	def resetValues(self):

		# Eval metrics
		self.TP = 0
		self.TN = 0
		self.FP = 0
		self.FN = 0
		self.accuracy = 0
		self.precision = 0
		self.recall = 0
		self.f1 = 0
		self.total_text = 0


	# Function splits data for k-fold cross validation
	def splitData(self):
		self.split_data = np.array_split(self.data, self.k_folds)

	
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


	# Test given data	
	def testModel(self, df):

		# Check each piece of text
		for index, row in df.iterrows():

			# Get sentence prediction
			prediction = self.predict(row['SentimentText'])

			# Record results
			self.total_text += 1
			if prediction == 1 and row['Sentiment'] == 1:
				self.TP += 1
			elif prediction == 0 and row['Sentiment'] == 0:
				self.TN += 1
			elif prediction == 0 and row['Sentiment'] == 1:
				self.FN += 1
			elif prediction == 1 and row['Sentiment'] == 0:
				self.FP += 1
		
		# Calculate final metrics
		try:
			self.accuracy = (self.TP+self.TN)/float(self.total_text)
		except ZeroDivisionError:
			self.accuracy = 0		
		try:		
			self.precision = self.TP/float((self.TP+self.FP))
		except ZeroDivisionError:
			self.precision = 0		
		try:
			self.recall = self.TP/float((self.TP+self.FN))
		except ZeroDivisionError:
			self.recall = 0		
		try:
			self.f1 = 2*((self.precision*self.recall)/float((self.precision+self.recall)))
		except ZeroDivisionError:
			self.f1 = 0		

	
	# Evaluate Algorithm with normal data
	def quickEval(self):

		# Split data and choose random split
		self.splitData()
		i = random.randint(0, self.k_folds-1)

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

		# Print results and reset values
		self.printEval()
		self.resetValues()


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
			self.resetValues()

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

	
