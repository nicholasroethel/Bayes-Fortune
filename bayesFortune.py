import numpy as np
import pandas as pd 

import sklearn
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer

def main():

	with open("./fortune_cookies/testdata.txt","r") as testDataFile: #gets the test data
		lines = testDataFile.readlines()
	testData = [x.strip() for x in lines] 

	with open("./fortune_cookies/testlabels.txt", "r") as testLabelFile: #gets the test labels
		lines = testLabelFile.readlines()
	testLabels = [x.strip() for x in lines]

	with open("./fortune_cookies/traindata.txt","r") as trainDataFile: #gets the training data
		lines = trainDataFile.readlines()
	trainData = [x.strip() for x in lines] 

	with open("./fortune_cookies/trainlabels.txt", "r") as trainLabelFile: #gets the training labels
		lines = trainLabelFile.readlines()
	trainLabels = [x.strip() for x in lines] 
	
	
	vectorizer = CountVectorizer() #initial word counter
	trainDataCounts = vectorizer.fit_transform(trainData) #count word freqs for training data
	testDataCounts = vectorizer.transform(testData) #count word freqs for test data

	
	MultiNB = MultinomialNB() #initalize the Multinomial Naive Bayes classifier
	MultiNB.fit(trainDataCounts, trainLabels) #fit the MNB to the data

	real_prediction = MultiNB.predict(testDataCounts) #test it agaianst the test data
	pred1 = accuracy_score(testLabels,real_prediction)*100

	sanity_prediction = MultiNB.predict(trainDataCounts) #test it against the training data
	pred2 = accuracy_score(trainLabels,sanity_prediction)*100


	print("The accuracy using the proper test data is:") #print results
	print(round(pred1),"%")


	print("The accuracy using the training data as the test data is:")
	print(round(pred2),"%")


if __name__ == "__main__":
	main()