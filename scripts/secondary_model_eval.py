#!/usr/bin/python3
import sys
sys.dont_write_bytecode = True
import time
import matplotlib.pyplot as plt
from naive_bayes_model import NaiveBayesText
from textblob_model import TextBlobModel
from vader_model import VaderModel
from event_data_naive_bayes import EventDataNaiveBayes
from event_data_neural_net import EventDataNeuralNet


### Michael O'Malley, Michael Burke
### Machine Learning
### Semster Project
### Secondary Model Evaluation


### Secondary Evaluation Class ###


class SecondaryEval:


	# Initialize class variables
	def __init__(self, datafile):
		self.data = datafile
		self.n = []
		self.enb_acc = []
		self.enb_pre = []
		self.enb_rec = []
		self.enb_f1 = [] 
		self.enn_acc = []
		self.enn_pre = []
		self.enn_rec = []
		self.enn_f1 = [] 
		self.nb_acc = []
		self.nb_pre = []
		self.nb_rec = []
		self.nb_f1 = [] 

		
	# Clear data
	def clearData(self):
		self.n = []
		self.enb_acc = []
		self.enb_pre = []
		self.enb_rec = []
		self.enb_f1 = [] 
		self.enn_acc = []
		self.enn_pre = []
		self.enn_rec = []
		self.enn_f1 = [] 
		self.nb_acc = []
		self.nb_pre = []
		self.nb_rec = []
		self.nb_f1 = [] 


	# Compile eval data from models
	def compile(self, i, n, k):

		# Only compile specified amount of data
		for j in range(0,i):
		
			# Compile X data
			self.n.append(n)
			
			# Compile Event Naive Bayes y data
			ENB = EventDataNaiveBayes(self.data, n, k, False, 2)
			ENB.fullEval(False)
			acc, pre, rec, f1 = ENB.returnEval()
			self.enb_acc.append(acc)
			self.enb_pre.append(pre)
			self.enb_rec.append(rec)
			self.enb_f1.append(f1)
			
			# Compile Event Neural Net y data
			ENN = EventDataNeuralNet(self.data, n, k, False, 2, 50)
			ENN.fullEval(False)
			acc, pre, rec, f1 = ENN.returnEval()
			self.enn_acc.append(acc)
			self.enn_pre.append(pre)
			self.enn_rec.append(rec)
			self.enn_f1.append(f1)
			
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
		axs[0,0].plot(self.n, self.enb_acc, label='EventNaiveBayes', linestyle='-', color='r')
		axs[0,0].plot(self.n, self.enn_acc, label='EventNerualNet', linestyle='-', color='b')
		axs[0,0].set_title("Accuracy")
		axs[0,0].set_xlabel("Rows of data")
		axs[0,0].set_ylabel("Accuracy")
		axs[0,0].legend()

		# Precision
		axs[0,1].plot(self.n, self.enb_pre, label='EventNaiveBayes', linestyle='-', color='r')
		axs[0,1].plot(self.n, self.enn_pre, label='EventNerualNet', linestyle='-', color='b')
		axs[0,1].set_title("Precision")
		axs[0,1].set_xlabel("Rows of data")
		axs[0,1].set_ylabel("Precision")
		axs[0,1].legend()
	    
		# Recall
		axs[1,0].plot(self.n, self.enb_rec, label='EventNaiveBayes', linestyle='-', color='r')
		axs[1,0].plot(self.n, self.enn_rec, label='EventNerualNet', linestyle='-', color='b')
		axs[1,0].set_title("Recall")
		axs[1,0].set_xlabel("Rows of data")
		axs[1,0].set_ylabel("Recall")
		axs[1,0].legend()

		# F1
		axs[1,1].plot(self.n, self.enb_f1, label='EventNaiveBayes', linestyle='-', color='r')
		axs[1,1].plot(self.n, self.enn_f1, label='EventNerualNet', linestyle='-', color='b')
		axs[1,1].set_title("F1 Score")
		axs[1,1].set_xlabel("Rows of data")
		axs[1,1].set_ylabel("F1 Score")
		axs[1,1].legend()

		# Show graphs
		plt.suptitle('EventNaiveBayes vs EventNeuralNet')
		plt.tight_layout(w_pad=1.5, h_pad=1.5)
		if save:
			fig.savefig('../images/EventNaiveBayes_vs_EventNeuralNet:NaiveBayesGraded.png', dpi=fig.dpi)
		plt.show()
		plt.close()


	# print Results
	def printResults(self):
		print(self.enb_acc)
		print(self.enb_pre)
		print(self.enb_rec)
		print(self.enb_f1)
		print(self.enn_acc)
		print(self.enn_pre)
		print(self.enn_rec)
		print(self.enn_f1)	


### Main Execution ###


if __name__ == "__main__":

	start_time = time.time()

	SE = SecondaryEval('trump')
	SE.graph(int(sys.argv[1]), 1000, 10)
	SE.printResults()

	print("Runtime: %s seconds" % (time.time() - start_time))

	    
	
	
