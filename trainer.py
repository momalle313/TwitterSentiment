#!/usr/bin/python
import sys
import pandas as pd
from textblob import TextBlob
from textblob import Blobber
from textblob.sentiments import NaiveBayesAnalyzer
from textblob.classifiers import NaiveBayesClassifier
from textblob.classifiers import DecisionTreeClassifier

import time

### Michael O'Malley, Michael Burke
### Machine Learning
### Semster Project
### First Level Model Training

# Usage function
def usage(program):
    print 'Usage: {} tweetfile'.format(program)


### Main Execution ###


if __name__=="__main__":

	start_time = time.time()	# For time calculation

	# Assert proper usage
	if len(sys.argv) != 2:
	        usage(sys.argv[0])
        	sys.exit(1)
	
	# Read in training data
	data = pd.read_csv('database/' + sys.arv[1])
