#!/usr/bin/python3
import sys
sys.dont_write_bytecode = True
import json
import time
import string
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mp
import matplotlib.lines as ml
from tweet_scorer import TweetScorer
from tweet_scorer_v2 import TweetScorerV2


### Michael O'Malley, Michael Burke
### Machine Learning
### Semster Project
### Clustering Evaluation


### Cluster Tweets Model ###


class ClusterTweets:
	
	
	# Initialize class values
	def __init__(self, keyword, modelFile='n100000k10.txt', n=500, k=10, max=100):
		
		# Read in fata
		self.keyword = keyword
		self.data = pd.read_csv('../database/' + keyword + '_Tweets.txt', sep='\t', names=['UserID','SentimentText','Time','Location', 'TweetID'], error_bad_lines=False, warn_bad_lines=False, nrows=n)

		# Format data
		self.data = self.data.dropna()
		self.data = TweetScorerV2(keyword, n=n, model=2).getData()

		# Prepare variables
		self.n = len(self.data)
		self.k = k
		self.centroids = []
		self.max_recluster = max


	# Function to compute Jaccard distance between sets
	def jaccard(self, A, B):
		intersect = len(A.intersection(B))
		union = len(A.union(B))
		return 1 - (intersect/float(union))


	# Function to choose clusters
	def clusterTweets(self):

		# Initialize cluster dict
		clusters = {}
		for centroid in self.centroids:
			clusters[centroid] = []

		# Assign tweets to clusters
		loc = self.data.columns.get_loc('SentimentText')
		for i in range(0, len(self.data.index)):
			min = float(1.0)
			cluster = 0

			# Check the distance of each tweet to each centroid
			for centroid in self.centroids:
				temp = self.jaccard(set(self.data.iat[i, loc]), set(self.data['SentimentText'].iloc[centroid]))

				# Assign cluster according to closest centroid
				if temp <= min:
					min = temp
					cluster = centroid
				
			# Add data to proper cluster
			clusters[cluster].append(i)

		# Return clustering results
		return clusters


	# Calculate random centroids
	def createCentroids(self):
		n = int(len(self.data.index)/float(self.k))
		for i in range(0, self.k):
			self.centroids.append(random.randint(0, len(self.data.index)-1))


	# Function to find new centroid
	def newCentroids(self, clusters):
		
		# Set change value to false, start at first centroid
		change = False
		place = 0

		# Check each cluster for a new centroid
		for key, value in clusters.items():
			
			# Set the min to the total jaccard distances for the current key
			min = 0
			for num in value:
				min += self.jaccard(set(self.data['SentimentText'].iloc[key]), set(self.data['SentimentText'].iloc[num]))

			# Check the total jaccard distances for every id against others
			for num in value:
				total = 0
				for i in range(0, len(value)):
					total += self.jaccard(set(self.data['SentimentText'].iloc[num]), set(self.data['SentimentText'].iloc[value[i]]))

				# If total distance is less than the total distance of 
				# the current centroid, the current number is the new centroid
				if total < min:
					self.centroids[place] = num
					change = True
					min = total

			# Change the centroid being focused on
			place += 1

		return change


	# Returns dict of clusters
	def makeClusters(self):

		# Create clusters
		self.createCentroids()
		clusters = self.clusterTweets()

		# Find new centroids and recalculate
		i = 0
		change = True
		while change:
			change = self.newCentroids(clusters)
			clusters = self.clusterTweets()
			i+=1
			if i >= self.max_recluster:
				break

		# Return clusters
		return clusters


	# Renumber keys of clusters
	def renumber(self, clusters):

		# Add new key to each value
		new_clusters = {}
		i = 1
		for key in clusters.keys():
			new_clusters[i] = clusters[key]
			i+=1

		# Return new values
		return new_clusters


	# Records clusters to database
	def recordClusters(self, clusters):

		# Renumber cluster
		clusters = self.renumber(clusters)

		# Add blank new column to database
		size = len(self.data.index)
		cluster = [None] * size
		self.data['Cluster'] = cluster

		# Make cluster and add them to the coulumn
		loc = self.data.columns.get_loc('Cluster')
		for i in range(0, len(self.data.index)):	
			for key, value in clusters.items():
				if i in value:
					self.data.iat[i, loc] = int(key)
	
		# Return numbered cluster
		return clusters


	# Compile data into lists
	def compileData(self, clusters):

		# Lists to return
		X = []
		y_neg = []
		y_pos = []
		diff = []

		# Analyze each key in the clusters dict
		for key in clusters.keys():
			X.append(key)
			neg = self.data['Sentiment'].loc[(self.data['Sentiment'] == 0) & (self.data['Cluster'] == key)]
			pos = self.data['Sentiment'].loc[(self.data['Sentiment'] == 1) & (self.data['Cluster'] == key)]
			n = len(neg)
			p = len(pos)
			y_neg.append(n)
			y_pos.append(p)
			try:
				diff.append(abs((n/float(n+p))-(p/float(n+p))))
			except ZeroDivisionError:
				diff.append(1.0)

		# Return compiled data
		return X, y_neg, y_pos, diff


	# Graph the results
	def graphClusters(self, clusters, show=True):

		# Compile X and y variables
		X, y_neg, y_pos, diff = self.compileData(clusters)
		
		# Make labels
		labels = []
		for i in range(0, len(X)):
			try:
				n = (y_neg[i]/float(y_neg[i]+y_pos[i]))*100
				p = (y_pos[i]/float(y_neg[i]+y_pos[i]))*100
			except ZeroDivisionError:
				n = 0
				p = 0
			labels.append('Neg:' + str(int(n)) + '%\nPos:' + str(int(p)) + '%')

		# Build stacked bar graph
		plt.figure(figsize=(12,8))
		p1 = plt.bar(X, y_neg, color='red')
		p2 = plt.bar(X, y_pos, bottom=y_neg)

		plt.ylabel('Number of Tweets')
		plt.xlabel('Clusters')
		plt.xticks(X, labels)
		plt.title(str(sys.argv[1])+' ENN Cluster Evaluation: n'+str(self.n)+'k'+str(self.k))
		
		avg = sum(diff)/float(len(diff))
		leg = [mp.Patch(color='red', label='Negative Sentiment'), mp.Patch(color='blue', label='Positive Sentiment'), ml.Line2D([], [], color='black', label='Avg Difference: %.2f' % avg)]
		plt.legend(handles=leg)	

		plt.savefig('../images/clustering/'+str(sys.argv[1])+'ENNClusterEvaluation:n'+str(self.n)+'k'+str(self.k)+'.png')
		if show:
			plt.show()
			plt.close()

		# Return avg difference
		return avg


### Main Execution ###


if __name__ == "__main__":

	start_time = time.time()

	CT = ClusterTweets(str(sys.argv[1]), n=1000, k=10)
	clusters = CT.makeClusters()
	clusters = CT.recordClusters(clusters)
		
	CT.graphClusters(clusters)

#	nrows = []
#	avgs = []
#	for j in range(100, 1000, 100):
#		avg_diff = []
#		for i in range(0,10):

#			CT = ClusterTweets(str(sys.argv[1]), n=j, k=10)
#			clusters = CT.makeClusters()
#			clusters = CT.recordClusters(clusters)
#			a, b, c, diff_temp = CT.compileData(clusters)
#			avg_diff.append(sum(diff_temp)/float(len(diff_temp)))

#		nrows.append(j)
#		avgs.append(sum(avg_diff)/float(len(avg_diff)))

	print("Runtime: %s seconds" % (time.time() - start_time))
	
#	plt.plot(nrows, avgs)

#	plt.ylabel('Average difference between\nPositive and Negative Tweets')
#	plt.xlabel('Number of Rows selected from Data')
#	plt.title(str(sys.argv[1])+' Cluster Evaluation')
		
#	plt.savefig('../images/clustering/'+str(sys.argv[1])+'ClusterEvaluation.png')
#	plt.show()
#	plt.close()
	
	


