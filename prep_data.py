import json
import os
import config
import librosa
import utils
import numpy as np
import matplotlib.pyplot as plt
import essentia
from essentia.standard import Envelope
import h5py

def main():

	with open(os.path.join(config.wav_dir,'sounds_temporal_deacrease_duration.json')) as jfile:
		data = json.load(jfile)

	files_to_use = [[x['id'], x['ac_analysis']] for x in data if x['duration']<1 and not x['id'] in config.do_not_use]

	lengths = []
	count = 0
	do_not_use = []

	with h5py.File(config.feats_dir+'feats.hdf5', mode='w') as hdf5_file:
		hdf5_file.create_dataset("waveform", [len(files_to_use), config.fs], np.float32)
		hdf5_file.create_dataset("envelope", [len(files_to_use), config.fs], np.float32)
		hdf5_file.create_dataset("mask", [len(files_to_use), config.fs], np.float32)
		hdf5_file.create_dataset("features", [len(files_to_use),len(config.feats_to_use)], np.float32)

	count = 0

	for lf in files_to_use:

		audio,fs = librosa.load(os.path.join(config.wav_dir+'sounds/',str(lf[0])+'.wav'), sr = config.fs)

		length = len(audio)

		if length<=config.fs:

			env = Envelope(releaseTime=50, attackTime=5)

			envelope = env(essentia.array(audio))

			audio = np.pad(audio, [0, config.fs - length], mode = 'constant')

			envelope = np.pad(envelope, [0, config.fs - length], mode = 'constant')

			mask = np.zeros(config.fs)

			mask[:length] = 1

			features = [lf[1][x] for x in config.feats_to_use]

			with h5py.File(config.feats_dir+'feats.hdf5', mode='a') as hdf5_file:
				hdf5_file["waveform"][count,:] = audio
				hdf5_file["envelope"][count,:] = envelope
				hdf5_file["mask"][count,:] = mask
				hdf5_file["features"][count,:] = features
		count+=1
		utils.progress(count,len(files_to_use))

if __name__ == '__main__':
    main()
