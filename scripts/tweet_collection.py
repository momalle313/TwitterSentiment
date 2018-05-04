#!/usr/bin/python3
import sys
sys.dont_write_bytecode = True
import tweepy


### Michael O'Malley, Michael Burke
### Machine Learning
### Semster Project
### Twitter Scraper


# Define usage for user
def usage():
    print("Usage: " + str(sys.argv[0]) + " keyword")


# Stream Listener for Location problem


class StreamListener(tweepy.StreamListener):
		

	# Open output file to append to
	output = open('../database/' + sys.argv[1] + '_Tweets.txt', 'a+')
	num = 0


	# When a status is found, record the necessary information
	def on_status(self, status):
		self.num += 1
		print("Hit: " + str(self.num))
		self.output.write(str(status.user.id_str) + '\t')
	
		# Check if retweeted
		if hasattr(status, 'retweeted_status'):
			try:
				self.output.write(status.retweeted_status.extended_tweet['full_text'].encode('ascii', 'ignore').replace('\n','')+'\t')
			except AttributeError:
				self.output.write(status.retweeted_status.text.encode('ascii', 'ignore').replace('\n','')+'\t')
		else:
			try:
				self.output.write(status.extended_tweet['full_text'].encode('ascii', 'ignore').replace('\n','')+'\t')
			except AttributeError:
				self.output.write(status.text.encode('ascii', 'ignore').replace('\n','')+'\t')
		
		self.output.write(str(status.created_at)+'\t')
		self.output.write(str(status.coordinates) + '\t')
		self.output.write(str(status.id) + '\n')


	# Stop if an error occurs
	def on_error(self, status_code):
        	print("Error Code: " + str(status_code))
        	return False


### Main Execution ###


if __name__ == "__main__":

	# Check proper usage
	if len(sys.argv) != 2:
		usage()
		sys.exit(1)

	# Grab keywords
	keywords = []
	keywords.append(sys.argv[1])
	keywords.append('#' + sys.argv[1])

	# Consumer keys and access token #

	# O'Malley's Key
	consumer_key = 'KNmgLXm5unaUBXCRigqXAy4kf'
	consumer_secret = 'mjGC6ceKYfaQgYoiD3O1wRpoXr2JrEu3xrLQGZlUjgC2HzHEhZ'
        access_token = '960373416892170240-DVCoL4NtsQfM0CtMxp8nCnuagWYVLjr'
        access_token_secret = 'ZoCNlg9H5gT7VzuvtUORXVva5pBqHM2Pt0KkEzLpyanfO'

	# Burke's Key
	#consumer_key = 'kamqWW85IkVhJusJqd2ESevJF'
	#consumer_secret = '9EEMxbeT3tl1tb6n0yWbn1WJ7Z8rpFuWlf0AxnxvMwwK8u9d1j'
        #access_token = '868515705485963264-5rAylTUvV19R4jK69ZrgtAMrHZYZZNM'	
	#access_token_secret = 'pD6FyXnDyaEGrPJNgGFFHqQ3h7UgH4SGdarHiLa87ZIAX'

	# OAuth process
	auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_token, access_token_secret)
	 
	# Interface creation
	api = tweepy.API(auth)

	# Produce Stream results
	myListener = StreamListener()
	stream = tweepy.Stream(auth=api.auth, listener=myListener, tweet_mode='extended')
	stream.filter(track=keywords)	



