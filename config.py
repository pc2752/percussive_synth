import numpy as np
import tensorflow as tf


wav_dir = '../datasets/percussive_synth/'

mode = 'L1'

feats_dir = './feats/'

if mode =='GAN':
	log_dir = './log_GAN/'
elif mode =='L1':
	log_dir = './log/'
elif mode == 'Content':
	log_dir = './log_content/'



data_log = './log/data_log.log'


dir_npy = './data_npy/'
stat_dir = './stats/'
h5py_file_train = './data_h5py/train.hdf5'
h5py_file_val = './data_h5py/val.hdf5'
val_dir = './val_dir_synth/'

in_mode = 'mix'
norm_mode_out = "max_min"
norm_mode_in = "max_min"

voc_ext = '_voc_stft.npy'
feats_ext = '_synth_feats.npy'

f0_weight = 10
max_models_to_keep = 10
f0_threshold = 1


filter_len = 5
encoder_layers = 15
filters = 32


fs = 16000
num_f0 = 256
max_phr_len = 16000
input_features = 7
output_features = 1

kernel_size = 2
num_filters = 32

augment_filters_every = 3

wavenet_layers = 12

train_split = 0.9

feats_to_use = ['ac_brightness', 'ac_hardness', 'ac_depth', 'ac_roughness', 'ac_boominess', 'ac_warmth', 'ac_sharpness']

do_not_use = [328490, 328491, 328482, 328483]



augment = True
aug_prob = 0.5

noise_threshold = 0.4 #0.7 for the unnormalized features
pred_mode = 'all'

# Hyperparameters
num_epochs = 2500

batches_per_epoch_train = 100
batches_per_epoch_val = 10

batch_size = 16
samples_per_file = 4


init_lr = 0.0002


comp_mode = 'mfsc'
hoptime = 5.80498866

noise = 0.05

print_every = 1
save_every = 50
validate_every =1 


dtype = tf.float32
