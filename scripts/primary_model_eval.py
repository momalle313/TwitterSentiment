#!/usr/bin/python
import sys
import time
import matplotlib.pyplot as plt
from naive_bayes_model import NaiveBayesText
from textblob_model import TextBlobModel


### Michael O'Malley, Michael Burke
### Machine Learning
### Semster Project
### Primary Model Evaluation


### Primary Evaluation Class ###


class PrimaryEval:


	# Initialize class variables
	def __init__(self, datafile):
		self.data = datafile
		self.n = []
		self.nb_acc = []
		self.nb_pre = []
		self.nb_rec = []
		self.nb_f1 = [] 
		self.tb_acc = []
		self.tb_pre = []
		self.tb_rec = []
		self.tb_f1 = []


	# Clear data
	def clearData(self):
		self.n = []
		self.nb_acc = []
		self.nb_pre = []
		self.nb_rec = []
		self.nb_f1 = [] 
		self.tb_acc = []
		self.tb_pre = []
		self.tb_rec = []
		self.tb_f1 = []


	# Compile eval data from models
	def compile(self, i, n, k):

		# Only compile specified amount of data
		for j in range(0,i):
		
			# Compile X data
			self.n.append(n)

			# Compile NaiveBayes y data
			NB = NaiveBayesText(self.data, n, k)
			NB.fullEval(False)
			acc, pre, rec, f1 = NB.returnEval()
			self.nb_acc.append(acc)
			self.nb_pre.append(pre)
			self.nb_rec.append(rec)
			self.nb_f1.append(f1)

			# Compile TextBlob y data
			TB = TextBlobModel(self.data, n, k)
			TB.fullEval(False)
			acc, pre, rec, f1 = TB.returnEval()
			self.tb_acc.append(acc)
			self.tb_pre.append(pre)
			self.tb_rec.append(rec)
			self.tb_f1.append(f1)
	
			# Iterate for next analysis
			n*=2


	# Graph data
	def graph(self, i, n, k, save=True):

		# Compile the data
		self.clearData()
		self.compile(i,n,k)

		# Plotting
		fig, axs = plt.subplots(2,2)
		fig.set_figheight(8)
		fig.set_figwidth(10)

		# Accuracy
		axs[0,0].plot(self.n, self.nb_acc, label='NaiveBayes', linestyle='-', color='r')
		axs[0,0].plot(self.n, self.tb_acc, label='TextBlob', linestyle='-', color='b')
		axs[0,0].set_title("Accuracy")
		axs[0,0].set_xlabel("Rows of data")
		axs[0,0].set_ylabel("Accuracy")
		axs[0,0].legend()

		# Precision
		axs[0,1].plot(self.n, self.nb_pre, label='NaiveBayes', linestyle='-', color='r')
		axs[0,1].plot(self.n, self.tb_pre, label='TextBlob', linestyle='-', color='b')
		axs[0,1].set_title("Precision")
		axs[0,1].set_xlabel("Rows of data")
		axs[0,1].set_ylabel("Precision")
		axs[0,1].legend()
	    
		# Recall
		axs[1,0].plot(self.n, self.nb_rec, label='NaiveBayes', linestyle='-', color='r')
		axs[1,0].plot(self.n, self.tb_rec, label='TextBlob', linestyle='-', color='b')
		axs[1,0].set_title("Recall")
		axs[1,0].set_xlabel("Rows of data")
		axs[1,0].set_ylabel("Recall")
		axs[1,0].legend()

		# F1
		axs[1,1].plot(self.n, self.nb_f1, label='NaiveBayes', linestyle='-', color='r')
		axs[1,1].plot(self.n, self.tb_f1, label='TextBlob', linestyle='-', color='b')
		axs[1,1].set_title("F1 Score")
		axs[1,1].set_xlabel("Rows of data")
		axs[1,1].set_ylabel("F1 Score")
		axs[1,1].legend()

		# Show graphs
		plt.suptitle('NaiveBayes vs TextBlob')
		plt.tight_layout(w_pad=1.5, h_pad=1.5)
		if save:
			fig.savefig('../images/NaiveBayes_vs_TextBlob.png', dpi=fig.dpi)
		plt.show()
		plt.close()


	# Print Results
	def printResults(self):
		print self.nb_acc
		print self.nb_pre
		print self.nb_rec
		print self.nb_f1 
		print self.tb_acc
		print self.tb_pre
		print self.tb_rec
		print self.tb_f1


### Main Execution ###


if __name__ == "__main__":

	start_time = time.time()

	PE = PrimaryEval('Sentiment Analysis Dataset.csv')
	PE.compile(int(sys.argv[1]), 1000, 10)
	PE.printResults()

	print("Runtime: %s seconds" % (time.time() - start_time))

	    
	
	
