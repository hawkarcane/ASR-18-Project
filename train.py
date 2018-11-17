import pandas as pd
import numpy as np
from sklearn import mixture
import matplotlib.pyplot as plt
import pickle
import os

def read_features(test_path):
	df = pd.read_hdf(test_path)
	features = np.array(df["features"].tolist())
	labels = np.array(df["labels"].tolist())
	return features,labels

def get_params(type_):
	if type_== "mfcc":
		corresponding_coff = 0
	elif type_== "mfcc_delta":
		corresponding_coff = 14
	else:
		corresponding_coff = 27
	return corresponding_coff

def mixture_size(type_,energy_coefficient=True):

	if type_ == "mfcc" and not energy_coefficient:
		return [2,4,8,16,32,64,128,256]
	else:
 		return [64]

def train(types,energy_coefficient):
	for type_ in types:
		df = pd.read_hdf("./features/"+type_+"/timit.hdf")
		features = np.array(df["features"].tolist())
		labels = np.array(df["labels"].tolist())
		print np.unique(labels)
		temp = {}
		corresponding_coff = get_params(type_)
		for i in range(len(labels)):
			if energy_coefficient:
				pass
			else:
				features[i][corresponding_coff] = 0
			if labels[i] in temp.keys():
				temp[labels[i]].append(features[i])
			else:
				temp[labels[i]] = [features[i]]
		n_components = mixture_size(type_,energy_coefficient)
		for nth_component in n_components:
			if energy_coefficient:
				models_path = "./models/with/"+type_+"/"+str(nth_component).zfill(3)
			else:
				models_path = "./models/without/"+type_+"/"+str(nth_component).zfill(3)
			os.system("mkdir -p "+models_path)
			for i in temp.keys():
				f = open(models_path+"/_"+i+".pkl","wb")
				g = mixture.GaussianMixture(n_components = nth_component)
				gmm = g.fit(temp[i])
				pickle.dump(gmm, f)
				f.close()
			if energy_coefficient:
				print str(nth_component) + " with type " + type_  + " with energy_coefficient "+" complete"
			else:
				print str(nth_component) + " with type " + type_  + " without energy_coefficient "+" complete"

def create_struct(types):
	os.system("mkdir -p ./models")
	os.system("mkdir -p ./models/with")
	os.system("mkdir -p ./models/without")
	for type_ in types:
		os.system("mkdir -p ./models/with/"+type_)
		os.system("mkdir -p ./models/without/"+type_)

if __name__ == '__main__':
	types = ["mfcc","mfcc_delta","mfcc_delta_delta"]
	create_struct(types)
	train(types,energy_coefficient = True)
	train(types,energy_coefficient = False)
