#!/usr/bin/python
import os
import sys
import time
import pandas as pd
from naive_bayes_text import NaiveBayesText
from tweet_scorer import TweetScorer


### Michael O'Malley, Michael Burke
### Machine Learning
### Semster Project
### First Level Model Training


# Usage function
def usage(program):
    print 'Usage: {} tweetfile'.format(program)


### Main Execution ###


if __name__=="__main__":

	# Assert proper usage
	if len(sys.argv) != 2:
	        usage(sys.argv[0])
        	sys.exit(1)

	start_time = time.time()
	
	# Read in training data
	model = TweetScorer(str(sys.argv[1]), 'n100000k10.txt')
	data = model.getData()
	print data.head(20)
