# Twitter_Sentiment_Analysis
Machine Learning Final Project

Team:

Michael O'Malley
Michael Burke

Goal:

We aim to create a sentiment analysis model that can accurately determine the sentiment of a tweet that specifically addresses a trending keyword. This can be used to determine the general sentiment of a hashtag or an event that is trending on twitter.

Filesystem:

A brief explanation of our filesystem - 

	database: contains all scraped and cleaned twitter data, as well as the Twitter Sentiment Corpus we used to train our base model

	deliverables: contains our term paper

	images: contains all evaluation images and measures of performance

	models: contains pretrained model information in text files. This was use to speed up the evaluation process and to see the effects of different model parameters.

	scripts: contains all of the scripts necessary to build, train, test, and evaluate our models.

Scripts:

Every script in our project is a class implementation, allowing for maximum reuse of code and functionality. Here is a brief explanation of each script -

	tweet_collection.py - collects tweets using the Tweepy library according to a given keyword. Only stops on keyboard interrupt.
	
	base_model.py - basic functionality for sentiment models. All models inherit from this.

	naive_bayes_model.py - our implementation of naive bayes sentiment analysis 
	
	textblob_model.py - a textblob sentiment model to compare our own to

	vader_model.py - a vader sentiment model to compare our own to

	tweet_scorer.py - uses a chosen model to score all the tweets in a formatted tweet file

	primary_model_eval.py - evaluates all primary models (naivebayes, textblob, and vader) by compiling their evaluation metrics and graphing them against each other.
	
	event_data_naive_bayes.py - adds onto our implementation of naive bayes by locating language unique to an event and adding it to the scoring algorithm. Also takes into account user.
	
	event_data_neural_net.py - a different approach to classification, finds all the unique language in an event file and weighs each word according to its positive or negative scores.

	tweet_scorer_v2.py - scores tweets specifically for secondary layer

	secondary_model_eval.py - evaluates models the same way the primary one does

	clustering_eval.py - grades tweets using a specified model. Then clusters tweets according to text similarity, and graphs the clusters in a bar graph that shows what percentage of clusters are positive and negative.
	
	test_all.py - runs every script to test if they're working
	
Running:

We use python3 to run all our scripts. The necessary libraries are included at the beginning of each script. Most scripts have a help function, but if they don't, then an example of how to run the script can be found in the test_all file.
	


	
