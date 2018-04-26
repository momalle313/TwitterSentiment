#!/usr/bin/python
import sys
import ast
import time
import string
import random
import numpy as np
import pandas as pd
from nltk import pos_tag, word_tokenize


### Michael O'Malley, Michael Burke
### Machine Learning
### Semster Project
### First Level Model Training


### Naive Bayes Text Analysis ###


class NaiveBayesText:


	# Initialize class values
	def __init__(self, datafile, n=500000, k=10):
		self.data = pd.read_csv('database/' + str(datafile), error_bad_lines = False, warn_bad_lines=False, nrows=n)
		self.tweet_num = n
		self.k_folds = k
		self.resetValues()


	# Reset all the values
	def resetValues(self):

		# Training data
		self.neg = {}
		self.pos = {}
		self.tot_neg = 0
		self.tot_pos = 0
		self.tot_word = 0
		self.pos_ratio = 0
		self.neg_ratio = 0

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


	# Train given DataFrame
	def trainData(self, df=None, reset=False):
		
		# Reset values if indicated
		if reset:
			self.resetValues()
			df = self.data

		# Count significant words (nouns/verbs, not articles/prepositions)
		for index, row in df.iterrows():
		
			# Split into words and tag
			words = word_tokenize(row['SentimentText'].translate(None,string.punctuation))
			tagged_words = pos_tag(words)

			# Go word by word
			for word, tag in tagged_words:

				# Add word to total count
				self.tot_word += 1

				# Check if sentiment is neg or pos, then add to dict
				if row['Sentiment'] == 0:
					self.tot_neg += 1
					if word in self.neg:
						self.neg[word] += 1
					else:
						self.neg[word] = 1
				else:
					self.tot_pos += 1
					if word in self.pos:
						self.pos[word] += 1
					else:
						self.pos[word] = 1

		# Fix zero-frequency problem
		self.pos = {key:value+1 for key, value in self.pos.iteritems()}
		self.neg = {key:value+1 for key, value in self.neg.iteritems()}
		for key in self.pos.keys():
			if key not in self.neg:
				self.neg[key] = 1
		for key in self.neg.keys():
			if key not in self.pos:
				self.pos[key] = 1
	
		# Normalize totals so ratios don't get too small
		self.tot_neg = self.tot_neg/(self.tweet_num/float(10))
		self.tot_pos = self.tot_pos/(self.tweet_num/float(10))
		self.tot_word = self.tot_word/(self.tweet_num/float(10))

		# Calculate word ratios
		self.pos_ratio = self.tot_pos/self.tot_word
		self.neg_ratio = self.tot_neg/self.tot_word
		self.pos = {key:(self.pos[key]/self.tot_pos) for key in self.pos.keys()}
		self.neg = {key:(self.neg[key]/self.tot_neg) for key in self.neg.keys()}
	

	def predict(self, sentence):
		
		# Set scores to 1 
		pos_score = 1
		neg_score = 1

		# Throw out non strings
		if not isinstance(sentence, basestring):
			return -1

		# Strip punctuation and split sentence
		words = sentence.translate(None,string.punctuation).split()

		# Go through every word
		for word in words:
			
			# Multiply pos and neg ratios to scores
			if word in self.pos:
				pos_score *= self.pos[word]
			if word in self.neg:
				neg_score *= self.neg[word]
				
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


	# Test given data	
	def testData(self, df):

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

	
	# Evaluate Naive Bayes Algorithm with normal data
	def quickEval(self):

		# Split data and choose random split
		self.splitData()
		i = random.randint(0, self.k_folds-1)

		# Train data
		if i == 0:
			self.trainData(pd.concat(self.split_data[i+1:len(self.split_data)]))
		elif i == len(self.split_data)-1:
			self.trainData(pd.concat(self.split_data[0:i]))
		else:
			self.trainData(pd.concat(self.split_data[0:i]))
			self.trainData(pd.concat(self.split_data[i+1:len(self.split_data)]))

		# Test data
		self.testData(self.split_data[i])

		# Print results and reset values
		self.printEval()
		self.resetValues()


	# Evaluate Naive Bayes algorithm with k-fold cross-validation
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

			# Train data
			if i == 0:
				self.trainData(pd.concat(self.split_data[i+1:len(self.split_data)]))
			elif i == len(self.split_data)-1:
				self.trainData(pd.concat(self.split_data[0:i]))
			else:
				self.trainData(pd.concat(self.split_data[0:i]))
				self.trainData(pd.concat(self.split_data[i+1:len(self.split_data)]))

			# Test data
			self.testData(self.split_data[i])
		
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

	
	# Save training values to file
	def saveModel(self, filename=None):
		
		# Open or create file to write to
		if filename == None:
			filename = 'n' + str(self.tweet_num) + 'k' + str(self.k_folds) + '.txt'
		f = open('models/' + filename,"w+")

		# Write values to file
		f.write(str(self.neg) + '\n')
		f.write(str(self.pos) + '\n')
		f.write(str(self.tot_neg) + '\n')
		f.write(str(self.tot_pos) + '\n')
		f.write(str(self.tot_word) + '\n')
		f.write(str(self.pos_ratio) + '\n')
		f.write(str(self.neg_ratio) + '\n')

		# Close file
		f.close()


	# Read training values from file
	def readModel(self, filename=None):

		# Open file and read in values
		values = []
		if filename == None:
			filename = 'n' + str(self.tweet_num) + 'k' + str(self.k_folds) + '.txt'
		with open('models/' + filename,"r") as f:
			for line in f:
				values.append(line)

		# Load values from file
		self.neg = ast.literal_eval(values[0])
		self.pos = ast.literal_eval(values[1])
		self.tot_neg = float(values[2])
		self.tot_pos = float(values[3])
		self.tot_word = float(values[4])
		self.pos_ratio = float(values[5])
		self.neg_ratio = float(values[6])

		# Close file
		f.close()


### Testing ###


if __name__=="__main__":

	start_time = time.time()
	n = 500000
	k = 10

	NB = NaiveBayesText('Sentiment Analysis Dataset.csv', n, k)

	NB.trainData(reset=True)
	NB.saveModel()

	NB.resetValues()
	NB.quickEval()

	print("Runtime: %s seconds" % (time.time() - start_time))

	
