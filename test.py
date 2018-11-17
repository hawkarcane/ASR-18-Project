import fnmatch
import pandas as pd
import numpy as np
from sklearn import mixture,metrics
import matplotlib.pyplot as plt
import pickle
import os

def read_models(models_path):
	models = {}
	for root, dirnames, filenames in os.walk(models_path):
		for filename in fnmatch.filter(filenames, "*.pkl"):
			x = pickle.load(open(os.path.join(root,filename)))
			models[filename[1:-4]] = x
	return models

def read_features(test_path):
	df = pd.read_hdf(test_path)
	features = np.array(df["features"].tolist())
	labels = np.array(df["labels"].tolist())
	return features,labels

def predicted_phonemes(test_path,models_path):
	models = read_models(models_path)
	features,labels = read_features(test_path)
	predicted = {}
	phonemes = models.keys()

	for phoneme in phonemes:
		predicted[phoneme] = models[phoneme].score_samples(features)

	predicted_values = []
	for i in range(len(features)):
		mx = predicted[phonemes[0]][i]
		output = phonemes[0]
		for phoneme in phonemes:
			if mx < predicted[phoneme][i]:
				output = phoneme
				mx = predicted[phoneme][i]
		predicted_values.append(output)
	return predicted_values

def save_predicted(save_to_file,predicted_phonemes):
	exList = open(save_to_file, 'wb')
	pickle.dump(predicted_phonemes,exList)
	exList.close()
	return "success"

def read_predicted(save_to_file):
	exList = open(save_to_file, 'r')
	predicted_phonemes = pickle.load(exList)
	exList.close()
	return predicted_phonemes

def calculate_accuracy(test_path,predicted_values):
	features,labels = read_features(test_path)
	return metrics.accuracy_score(labels.tolist(),predicted_values)

def mixture_size(type_,energy_coefficient=True):
	if type_ == "mfcc" and not energy_coefficient:
		return [2,4,8,16,32,64,128,256]
	else:
		return [64]

if __name__ == '__main__':
	types = ["mfcc","mfcc_delta","mfcc_delta_delta"]
	
	energy_coefficient=True
	for type_ in types:
		for i in mixture_size(type_,energy_coefficient):
			if energy_coefficient == True:
				models_path = "./models/with/"+type_+"/"+str(i).zfill(3)
			else:
				models_path = "./models/without/"+type_+"/"+str(i).zfill(3)
			test_path = "./test/"+type_+"/test.hdf"
			predicted_values = predicted_phonemes(test_path,models_path)
			print "The accuracy that was calculated for "+ test_path + " " + models_path + " is  = " + str(calculate_accuracy(test_path,predicted_values))

	energy_coefficient=False
	for type_ in types:
		for i in mixture_size(type_,energy_coefficient):
			if energy_coefficient == True:
				models_path = "./models/with/"+type_+"/"+str(i).zfill(3)
			else:
				models_path = "./models/without/"+type_+"/"+str(i).zfill(3)
			test_path = "./test/"+type_+"/test.hdf"
			predicted_values = predicted_phonemes(test_path,models_path)
			print "The accuracy that was calculated for "+test_path + " " + models_path + " is  = " + str(calculate_accuracy(test_path,predicted_values)

