#!/usr/bin/python
import os
import pydot
import sys


### Michael O'Malley, Michael Burke
### Machine Learning
### Semster Project
### Test All Scripts


### Main Execution ###


if __name__ == '__main__':
    
	print "Naive Bayes Test:\n"
	os.system("./naive_bayes_model.py 1000")
	print '\n'

	print "Text Blob Test:\n"
	os.system("./textblob_model.py 1000")
	print '\n'
	
	print "Tweet Scorer Test:\n"
	os.system("./tweet_scorer.py")
	print '\n'

	print "Event Data Test:\n"
	os.system("./event_data_model.py trump")
