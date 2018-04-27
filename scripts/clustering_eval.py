#!/usr/bin/python
import sys
import json
import pandas as pd

### Michael O'Malley, Michael Burke
### Machine Learning
### Semster Project
### Clustering Evaluation


### Cluster Tweets Model ###


class ClusterTweets:
	
	
	# Initialize class values
	def __init__(self):
		pass



# Function to compute Jaccard distance between sets
def jaccard(A, B):
	intersect = len(A.intersection(B))
	union = len(A.union(B))
	return 1 - (intersect/float(union))

# Function to choose clusters
def clusterTweets(centroids, tweets):

	# Initialize cluster dict
	clusters = {}
	for centroid in centroids:
		clusters[centroid] = []

	# Assign tweets to clusters
	for key, value in tweets.iteritems():
		min = float(1.0)
		cluster = 0
		
		# Check the distance of each tweet to each centroid,
		# assign it to the centroid with the smallest distance
		for centroid in centroids:
			temp = jaccard(set(value.split()), set(tweets[centroid].split()))
			if temp <= min:
				min = temp
				cluster = centroid
				
		clusters[cluster].append(key)

	return clusters
			
# Function to find new centroid
def newCentroids(centroids, clusters, tweets):

	change = False
	place = 0

	# Check each cluster for a new centroid
	for key, value in clusters.iteritems():
	
		# Set the min to the total jaccard distances for the current key
		min = 0
		for num in value:
			min += jaccard(set(tweets[key].split()), set(tweets[num].split()))

		# Check the total jaccard distances for every id against the others in the cluster
		for num in value:
			total = 0
			for i in range(0, len(value)):
				total += jaccard(set(tweets[num].split()), set(tweets[value[i]].split()))

			# If total distance is less than the total distance of 
			# the current centroid, the current number is the new centroid
			if total < min:
				centroids.insert(place, num)
				change = True
				min = total

		# Change the centroid being focused on
		place += 1

	return centroids, change


### Main Execution ###


if __name__ == "__main__":


	# Read in tweets
	tweets = pd.read_csv('../database/trump_Tweets.txt', sep='\t', names=['UserID','SentimentText','Time','Location'], error_bad_lines=False, warn_bad_lines=False, nrows=100)

	print tweets

	sys.exit()
	# Read in Jsons
	tweet_jsons = []
	for line in open('Tweets.json', 'r'):
		tweet_jsons.append(json.loads(line))

	tweets = {}
	for json in tweet_jsons:
		tweets[int(json["id_str"])] = json["text"]

	# Read in Initial seeds
	centroids = []
	for line in open('InitialSeeds.txt', 'r'):
		centroid = line.replace(',\n', '')
		centroids.append(int(centroid))

	# Create clusters
	clusters = cluster_tweets(centroids, tweets)

	# Find new centroids and recalculate
	change = True
	while change:
		centroids, change = newCentroids(centroids, clusters, tweets)
		clusters = clusterTweets(centroids, tweets)

	# Output clusters into results file
	output = open('ClusteringResults.txt', 'w')
	for key, value in clusters.iteritems():
		output.write(str(key) + ': ')
		output.write(str(value) + '\n')

	output.close()

