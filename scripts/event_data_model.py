#!/usr/bin/python
import sys
import time
import string
import numpy as np
import pandas as pd
from tweet_scorer import TweetScorer


### Michael O'Malley, Michael Burke
### Machine Learning
### Semster Project
### Event Data Model


# Usage function
def usage(program):
    print 'Usage: {} keyword'.format(program)


### Main Execution ###


if __name__=="__main__":

	start_time = time.time()

	# Assert proper usage
	if len(sys.argv) != 2:
	        usage(sys.argv[0])
        	sys.exit(1)

	start_time = time.time()
	
	# Obtain scored twitter data and word dictionaries
	model = TweetScorer(str(str(sys.argv[1]) + '_Tweets.txt'), 'n100000k10.txt')
	data = model.getData()
	pos_dict, neg_dict = model.getDicts()

	# Count words unique to the event
	pos = {}
	neg = {}
	for index, row in data.iterrows():
		
		# Split into words
		words = row['SentimentText'].translate(None,string.punctuation).decode('ascii','ignore').encode('ascii','ignore').split()

		# Go word by word
		for word in words:

			# If word is positive and unique to event, record it
			if row['Sentiment'] == 1:
				if word not in pos_dict:
					if word in pos:
						pos[word] += 1
					else:
						pos[word] = 1

			# If word is negative and unique to event, record it
			else:
				if word not in neg_dict:
					if word in neg:
						neg[word] += 1
					else:
						neg[word] = 1
	
	new = {}
	for key in pos.keys():
		if key in neg:
			new[key] = pos[key] - neg[key]

	new_list = sorted(new.items(), key=lambda x: x[1], reverse=True)		
	print new_list[0:20]
	new_list = sorted(new.items(), key=lambda x: x[1], reverse=False)		
	print new_list[0:20]
	#pos_list = sorted(pos.items(), key=lambda x: x[1], reverse=True)		
	#print pos_list[0:20]

	print("Runtime: %s seconds" % (time.time() - start_time))

