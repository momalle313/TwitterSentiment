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


### Main Execution ###


if __name__ == "__main__":

	start_time = time.time()

	# Initialize evaluation lists
	n = []
	nb_acc = []
	nb_pre = []
	nb_rec = []
	nb_f1 = [] 
	tb_acc = []
	tb_pre = []
	tb_rec = []
	tb_f1 = []

	# Compile eval data from both models
	k = 10
	i = 1000
	for j in range(0,10):
		
		# Compile X data
		n.append(i)

		# Compile NaiveBayes y data
		NB = NaiveBayesText('Sentiment Analysis Dataset.csv', i, k)
		NB.fullEval(False)
		acc, pre, rec, f1 = NB.returnEval()
		nb_acc.append(acc)
		nb_pre.append(pre)
		nb_rec.append(rec)
		nb_f1.append(f1)

		# Compile TextBlob y data
		TB = TextBlobModel('Sentiment Analysis Dataset.csv', i, k)
		TB.fullEval(False)
		acc, pre, rec, f1 = TB.returnEval()
		tb_acc.append(acc)
		tb_pre.append(pre)
		tb_rec.append(rec)
		tb_f1.append(f1)
	
		# Iterate for next analysis
		i*=2
	print("Runtime: %s seconds" % (time.time() - start_time))
    

	### Graph Results ###

	    
	# Plotting
	fig, axs = plt.subplots(2,2)
	fig.set_figheight(8)
	fig.set_figwidth(10)

	# Accuracy
	axs[0,0].plot(n, nb_acc, label='NaiveBayes', linestyle='-', color='r')
	axs[0,0].plot(n, tb_acc, label='TextBlob', linestyle='-', color='b')
	axs[0,0].set_title("Accuracy")
	axs[0,0].set_xlabel("Rows of data")
	axs[0,0].set_ylabel("Accuracy")
	axs[0,0].legend()

	# Precision
	axs[0,1].plot(n, nb_pre, label='NaiveBayes', linestyle='-', color='r')
	axs[0,1].plot(n, tb_pre, label='TextBlob', linestyle='-', color='b')
	axs[0,1].set_title("Precision")
	axs[0,1].set_xlabel("Rows of data")
	axs[0,1].set_ylabel("Precision")
	axs[0,1].legend()
    
	# Recall
	axs[1,0].plot(n, nb_rec, label='NaiveBayes', linestyle='-', color='r')
	axs[1,0].plot(n, tb_rec, label='TextBlob', linestyle='-', color='b')
	axs[1,0].set_title("Recall")
	axs[1,0].set_xlabel("Rows of data")
	axs[1,0].set_ylabel("Recall")
	axs[1,0].legend()

	# F1
	axs[1,1].plot(n, nb_f1, label='NaiveBayes', linestyle='-', color='r')
	axs[1,1].plot(n, tb_f1, label='TextBlob', linestyle='-', color='b')
	axs[1,1].set_title("F1 Score")
	axs[1,1].set_xlabel("Rows of data")
	axs[1,1].set_ylabel("F1 Score")
	axs[1,1].legend()

	# Show graphs
	plt.suptitle('NaiveBayes vs TextBlob')
	plt.tight_layout(w_pad=1.5, h_pad=1.5)
	fig.savefig('../images/NaiveBayes_vs_TextBlob.png', dpi=fig.dpi)
	plt.show()
	plt.close()

	
