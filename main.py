import models
import tensorflow as tf
import argparse
import os, sys
import config
import utils
import numpy as np
import mir_eval

def train(_):
    if config.mode =='GAN':
        model = models.PercSynthGAN()
    elif config.mode =='L1':
        model = models.PercSynth()
    elif config.mode == 'Content':
        model = models.PercSynthContent()
    elif config.mode == 'Encode':
        model = models.PercSynthEncode()
    model.train()

def eval_hdf5_file():
    if config.mode =='GAN':
        model = models.PercSynthGAN()
    elif config.mode =='L1':
        model = models.PercSynth()
    elif config.mode == 'Content':
        model = models.PercSynthContent()
    elif config.mode == 'Encode':
        model = models.PercSynthEncode()
    model.test_model()

if __name__ == '__main__':
    if len(sys.argv)<2 or sys.argv[1] == '-help' or sys.argv[1] == '--help' or sys.argv[1] == '--h' or sys.argv[1] == '-h':
        print("%s --help or -h or --h or -help to see this menu" % sys.argv[0])
        print("%s --train or -t or --t or -train to train the model" % sys.argv[0])
        print("%s -e or --e or -eval or --eval  <filename> to evaluate an hdf5 file" % sys.argv[0])
        print("%s -v or --v or -val or --val <filename> to calculate metrics for entire dataset and save to given filename" % sys.argv[0])
        print("%s -w or --w or -wavfile or --wavfile <filename> <save_path> to evaluate wavefile and save CSV" % sys.argv[0])
        print("%s -wf or --wf or -wavfolder or --wavolder <foldername> <save_path> to evaluate all wavefiles in the folder and save CSV" % sys.argv[0])
    else:
        if sys.argv[1] == '-train' or sys.argv[1] == '--train' or sys.argv[1] == '--t' or sys.argv[1] == '-t':
            print("Training")
            tf.app.run(main=train)

        elif sys.argv[1] == '-e' or sys.argv[1] == '--e' or sys.argv[1] == '--eval' or sys.argv[1] == '-eval':
            eval_hdf5_file()
