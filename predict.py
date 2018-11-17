import errno
import os
from os import path
import sys
import tarfile
import fnmatch
import pandas as pd
import subprocess
import argparse
from sklearn import mixture,metrics
from mapping import phone_maps
import python_speech_features as psf
import scipy.io.wavfile as wav
import pickle
import numpy as np
timit_phone_map = phone_maps(mapping_file="kaldi_60_48_39.map")

def read_transcript(phn_file):
    with open(phn_file, "r") as file:
        trans = file.readlines()
    durations = [ele.strip().split(" ")[:-1] for ele in trans]
    durations_int = []
    for duration in durations:
        durations_int.append([int(duration[0]), int(duration[1])])
    trans = [ele.strip().split(" ")[-1] for ele in trans]
    trans = [timit_phone_map.map_symbol_reduced(symbol=phoneme) for phoneme in trans]
    return trans, durations_int

def compute_mfcc(wav_file, n_delta=0):
    mfcc_feat = psf.mfcc(wav_file)
    if(n_delta == 0):
        return(mfcc_feat)
    elif(n_delta == 1):
        return(np.hstack((mfcc_feat, psf.delta(mfcc_feat,1))))
    elif(n_delta == 2):
        return(np.hstack((mfcc_feat, psf.delta(mfcc_feat,1), psf.delta(mfcc_feat, 2))))
    else:
        return 0

def read_features_from_file(wavfile,phn_file,n_delta):
	sph_file = wavfile
	wav_file = wavfile[:-4] + "_rif.wav"
	print("converting {} to {}".format(sph_file, wav_file))
	subprocess.check_call(["sox", sph_file, wav_file])
	print("Preprocessing Complete")
	mfcc_features = []
	trans, durations = read_transcript(phn_file)
	(sample_rate,wav_file) = wav.read(wav_file)
	mfcc_feats = compute_mfcc(wav_file[durations[0][0]:durations[0][1]], n_delta=n_delta)
	labels = []

	for i in range(len(mfcc_feats)):
		labels.append(trans[0])
	for index, chunk in enumerate(durations[1:]):
		mfcc_feat = compute_mfcc(wav_file[chunk[0]:chunk[1]], n_delta=n_delta)
		mfcc_feats = np.vstack((mfcc_feats, mfcc_feat))
		for i in range(len(mfcc_feat)):
			labels.append(trans[index])
	mfcc_features.extend(mfcc_feats)
	return mfcc_features,labels

def read_models(models_path):
	models = {}
	for root, dirnames, filenames in os.walk(models_path):
		for filename in fnmatch.filter(filenames, "*.pkl"):
			x = pickle.load(open(os.path.join(root,filename)))
			models[filename[1:-4]] = x
	return models
def calculate_accuracy(wavfile,phnfile,predicted_values):
	features,labels = read_features_from_file(wavfile,phnfile,2)
	return metrics.accuracy_score(labels,predicted_values)

def predicted_phonemes(wavfile,phnfile,models_path):
	models = read_models(models_path)
	features,labels = read_features_from_file(wavfile,phnfile,2)
	predicted = {}
	phonemes = models.keys()

	for phoneme in phonemes:
		predicted[phoneme] = models[phoneme].score_samples(features)

	predicted_values = []
	for i in range(len(features)):
		mx = predicted[phoneme][i]
		output = phonemes[0]
		for phoneme in phonemes:
			if mx < predicted[phoneme][i]:
				output = phoneme
				mx = predicted[phoneme][i]
		predicted_values.append(output)
	return predicted_values

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('--wav', type=str, default="./SA1.WAV",
						help='TIMIT root directory')
	parser.add_argument('--phn', type=str, default="./SA1.PHN",
						help='Number of delta features to compute')
	args = parser.parse_args()
	predicted_values = predicted_phonemes(args.wav,args.phn,"./models/with/mfcc_delta_delta/064/")
	print calculate_accuracy(args.wav,args.phn,predicted_values)