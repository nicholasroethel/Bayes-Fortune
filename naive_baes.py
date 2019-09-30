#Multinomial Naive Bayes algorithm that predictes fortune cookie outcomes

import numpy as np
import pandas as pd
import csv
import math

class trainedData(object):
	def __init__(self, vocab,clasifiers,prior,conprob):
		super(trainedData, self).__init__()
		self.vocab = vocab #the vocab for the MNB
		self.clasifiers = clasifiers #the classifers for the MNB (in this case just 0 and 1)
		self.prior = prior #the prior probabilities
		self.conprob = conprob #the weighted pronabilities for each vocab word.  (e.x: [['world', -3.5052856741441323, 1], ['world', -2.922465945298413, 0]])



def getAccuracy(list1, list2): #gets the similary between two equal sized lists
	count = 0
	same = 0
	length = len(list1)
	while(count<length):
		if list1[count] == list2[count]:
			same+=1
		count +=1
	return same/length*100


def CountTokensOfTerm(text,t): #counts the tokens
	a = len(text[text.str.contains(t)])
	return a
	

def train(data): #trains a MNB
	vocab = [x.split() for x in data["quotes"]]
	vocab = [val for sublist in vocab for val in sublist]
	vocab = list(set(vocab)) #a vocab without duplicates

	clasifiers = data["clasifiers"].unique()
	numberOfDocs = len(data.index)
	prior = [] 
	conprob = []

	for c in clasifiers: #for each classifer
		textCountGivenVocab = []

		DocumentsCountGivenClassifier = len(data["quotes"].loc[data["clasifiers"] == c].index)
		prior.append(math.log10(DocumentsCountGivenClassifier/numberOfDocs))

		textGivenClassifier = data["quotes"].loc[data["clasifiers"] == c]
		for t in vocab:
			b = CountTokensOfTerm(textGivenClassifier,t)
			textCountGivenVocab.append(b)
		totalVocabFrequency = sum([x+1 for x in textCountGivenVocab]) 

		for (t,n) in zip(vocab,textCountGivenVocab):
			conprob.append([t,math.log10((n+1)/totalVocabFrequency),c])
	return trainedData(vocab,clasifiers,prior,conprob)
	
def MultinomialBayes(trainedData,data):
	sentences = [x.split() for x in data[0]] #gets all the sentances
	predictions = []
	for sentence in sentences: #for each sentance
		score = []
		count = 0
		for c in trainedData.clasifiers: #for each classifier
			score.append(trainedData.prior[count]) #add the scores
			for word in sentence: #for each word in the sentance
				matches = (x for x in trainedData.conprob if x[0] == word)
				matches = list(matches)
				flatMatches = [item for sublist in matches for item in sublist]
				if (word in trainedData.vocab) and count == 0: #if the word matches and you're currently looking at the false probability
					score[0] += flatMatches[4]
					
				elif (word in trainedData.vocab) and count == 1: #if the word matches and you're currently looking at the true probability
					score[1] += flatMatches[1]
				else: #otherwise no match and skip
					pass
			count += 1
		if(score[0]>score[1]): #return the option with the higher probablilty
			predictions.append(0)
		else:
			predictions.append(1)
	return predictions




def main():

	#reads in data
	quotes = pd.read_csv('traindata.txt', header = None)
	labels = pd.read_csv('trainlabels.txt', header = None)
	testQuotes = pd.read_csv('testdata.txt', header = None)
	testLabelFile = pd.read_csv('testlabels.txt', header = None)

	#puts the target attributes into lists
	labelsList = labels.values.tolist()
	trainLabels = [item for sublist in labelsList for item in sublist]
	labelsList = testLabelFile.values.tolist()
	testLabels = [item for sublist in labelsList for item in sublist]

	#creates the data fram for the training data
	frames = [quotes,labels]
	data = pd.concat(frames,axis=1)
	data.columns =["quotes","clasifiers"]


	trainedData = train(data) #train the MNB with the training data
	predictions = [] #list to store the predictions
	predictions = (MultinomialBayes(trainedData,quotes)) #get the predictions using the MNB
	accuracy = getAccuracy(predictions, trainLabels) #calculate the accuracy
	
	print("The accuracy using the training data as the test data is:") #prints the score
	print(round(accuracy),"%")

	predictions = (MultinomialBayes(trainedData,testQuotes)) #get the predictions using the MNB
	accuracy = getAccuracy(predictions, trainLabels) #prints the score
	print("The accuracy using the proper test data is:")
	print(round(accuracy),"%")
	
	
if __name__ == '__main__':
	main()