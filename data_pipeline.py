import numpy as np
import os
import time
import h5py
import random
import matplotlib.pyplot as plt
import collections
import config
import utils
import soundfile as sf

from scipy.ndimage import filters


# def one_hotize(inp, max_index=config.num_phos):


#     output = np.eye(max_index)[inp.astype(int)]

#     return output



def gen_train_val():
    casas_list = [x for x in os.listdir(config.voice_dir) if x.endswith('.hdf5') and x.startswith('casas') and not x in config.do_not_use and not x.startswith('casasros')]

    trn_list = casas_list[:int(len(casas_list)*0.9)]

    val_list = casas_list[int(len(casas_list)*0.9):]

    utils.list_to_file(val_list,config.log_dir+'val_files.txt')

    utils.list_to_file(trn_list,config.log_dir+'train_files.txt')





def data_gen(mode = 'Train'):

    with h5py.File(config.feats_dir+'kick_feats.hdf5', mode='r') as hdf5_file:
        audios = hdf5_file["waveform"][()]
        envelope = hdf5_file["envelope"][()]
        mask = hdf5_file["mask"][()]
        features = hdf5_file["features"][()]

    files_to_use = [x for x in os.listdir(config.wav_dir) if x.endswith('.wav') and not x.startswith('.')]
    # import pdb;pdb.set_trace()    
    

    audios = np.delete(audios, config.remove_indecis, axis=0)
    envelope = np.delete(envelope, config.remove_indecis, axis=0)
    mask = np.delete(mask, config.remove_indecis, axis=0)
    features = np.delete(features, config.remove_indecis, axis=0)
    files_to_use = np.delete(files_to_use, config.remove_indecis, axis=0)


    max_feats = features.max(axis = 0)

    train_split = int(len(audios)*config.train_split)

    # import pdb;pdb.set_trace()




    if mode == "Train":
        batches_per_epoch = config.batches_per_epoch_train
        in_indecis = np.arange(train_split)
    else:
        in_indecis = np.arange(train_split, len(audios))
        batches_per_epoch = config.batches_per_epoch_val

        # random.shuffle(in_indecis)

    for i, idx_batch in enumerate(range(batches_per_epoch)):

        i_start = i * config.batch_size
        i_end = min([(i + 1) * config.batch_size, len(in_indecis)])
        indecis = in_indecis[i_start:i_end]
        indecis.sort()

        out_audios = audios[indecis]
        out_envelopes = envelope[indecis]
        out_masks = mask[indecis]
        out_features = features[indecis]/max_feats
        out_envelopes = np.array([x/(x+1e-12).max() for x in out_envelopes])
        out_files = files_to_use[indecis]


        yield np.expand_dims(out_audios, -1), np.expand_dims(out_envelopes, -1), out_features, np.expand_dims(out_masks, -1)



def get_stats():
    with h5py.File(config.feats_dir+'feats.hdf5', mode='r') as hdf5_file:
        audios = hdf5_file["waveform"][()]
        envelope = hdf5_file["envelope"][()]
        mask = hdf5_file["mask"][()]
        features = hdf5_file["features"][()]
    import pdb;pdb.set_trace()



def main():
    # gen_train_val()
    # get_stats()
    gen = data_gen('val')
    while True :
        start_time = time.time()
        out_audios, out_envelopes, out_features, out_masks, out_files = next(gen)
        print(time.time()-start_time)

    #     plt.subplot(411)
    #     plt.imshow(np.log(1+inputs.reshape(-1,513).T),aspect='auto',origin='lower')
    #     plt.subplot(412)
    #     plt.imshow(targets.reshape(-1,66)[:,:64].T,aspect='auto',origin='lower')
    #     plt.subplot(413)
    #     plt.plot(targets.reshape(-1,66)[:,-2])
    #     plt.subplot(414)
    #     plt.plot(targets.reshape(-1,66)[:,-1])

    #     plt.show()
    #     # vg = val_generator()
    #     # gen = get_batches()


        import pdb;pdb.set_trace()


if __name__ == '__main__':
    main()
