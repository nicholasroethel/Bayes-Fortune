import numpy as np
import pandas as pd
import math

class trainedData(object):
	"""docstring for trainedData"""
	def __init__(self, vocab,clasifiers,prior,conprob):
		super(trainedData, self).__init__()
		self.vocab = vocab
		self.clasifiers = clasifiers
		self.prior = prior
		self.conprob =conprob


def CountTokensOfTerm(text,t):
	a = len(text[text.str.contains(t)])
	return a
	

def train(data):
	vocab = [x.split() for x in data["quotes"]]
	vocab = [val for sublist in vocab for val in sublist]
	vocab = list(set(vocab))#remove duplicate words
	clasifiers = data["clasifiers"].unique()
	numberOfDocs = len(data.index)
	prior = []
	conprob = []
	for c in clasifiers:
		textCountGivenVocab = []
		DocumentsCountGivenClassifier = len(data["quotes"].loc[data["clasifiers"] == c].index)
		prior.append(math.log10(DocumentsCountGivenClassifier/numberOfDocs))
		textGivenClassifier = data["quotes"].loc[data["clasifiers"] == c]
		for t in vocab:
			b = CountTokensOfTerm(textGivenClassifier,t)
			textCountGivenVocab.append(b)
		totalVocabFrequency = sum([x+1 for x in textCountGivenVocab]) #MIGHT HAVE ISSUE DON"T KNOW YET
		for (t,n) in zip(vocab,textCountGivenVocab):
			conprob.append([t,math.log10((n+1)/totalVocabFrequency)])
	return trainedData(vocab,clasifiers,prior,dict(conprob))
	
def MultinomialBayes(trainedData,data):
	score = []
	print(trainedData.clasifiers)
	# words = 
	# cl = 0
	# for c in trainedData.clasifiers:
	# 	score.add(trainedData.prior[cl])
	# 	for t in words:
	# 		a = trainedData.vocab.index(t)
	# 		score[cl] += math.log10(trainedData)
	# 	cl += 1
	# 	print(score)



def main():
	quotes = pd.read_csv('traindata.txt', header = None)
	labels = pd.read_csv('trainlabels.txt', header = None)
	frames = [quotes,labels]
	data = pd.concat(frames,axis=1)
	data.columns =["quotes","clasifiers"]
	trainedData = train(data)
	MultinomialBayes(trainedData,quotes)
	print(trainedData.conprob)
	
if __name__ == '__main__':
	main()