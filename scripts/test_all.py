#!/usr/bin/python3
import os
import sys
sys.dont_write_bytecode = True


### Michael O'Malley, Michael Burke
### Machine Learning
### Semster Project
### Test All Scripts


### Main Execution ###


if __name__ == '__main__':
    
	print("Naive Bayes Test:\n")
	os.system("./naive_bayes_model.py 1000")
	print('\n')

	print("Text Blob Test:\n")
	os.system("./textblob_model.py 1000")
	print('\n')

	print("Vader Test:\n")
	os.system("./vader_model.py 1000")
	print('\n')
	
	print("Tweet Scorer Test:\n")
	os.system("./tweet_scorer.py trump")
	print('\n')

	print("Primary Model Test:\n")
	os.system("./primary_model_eval.py 2")
	print('\n')

	print("Event Data Naive Bayes Test:\n")
	os.system("./event_data_naive_bayes.py trump")
	print('\n')

	print("Event Data Neural Net Test:\n")
	os.system("./event_data_neural_net.py trump")
	print('\n')

	print("Tweet Scorer V2 Test:\n")
	os.system("./tweet_scorer_v2.py trump")
	print('\n')

	print("Secondary Model Test:\n")
	os.system("./secondary_model_eval.py 2")
	print('\n')

	print("Cluster Tweets Test:\n")
	os.system("./clustering_eval.py trump")

