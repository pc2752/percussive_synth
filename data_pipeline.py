import numpy as np
import os
import time
import h5py

import matplotlib.pyplot as plt
import collections
import config
import utils
import soundfile as sf

from scipy.ndimage import filters


def one_hotize(inp, max_index=config.num_phos):


    output = np.eye(max_index)[inp.astype(int)]

    return output



def gen_train_val():
    casas_list = [x for x in os.listdir(config.voice_dir) if x.endswith('.hdf5') and x.startswith('casas') and not x in config.do_not_use and not x.startswith('casasros')]

    trn_list = casas_list[:int(len(casas_list)*0.9)]

    val_list = casas_list[int(len(casas_list)*0.9):]

    utils.list_to_file(val_list,config.log_dir+'val_files.txt')

    utils.list_to_file(trn_list,config.log_dir+'train_files.txt')





def data_gen(mode = 'Train', sec_mode = 0):

    casas_list = [x for x in os.listdir(config.voice_dir) if x.endswith('.hdf5') and x.startswith('casas') and not x in config.do_not_use and not x.startswith('casasros')]

    trn_list = casas_list[:int(len(casas_list)*0.9)]

    val_list = casas_list[int(len(casas_list)*0.9):]

    stat_file = h5py.File(config.stat_dir+'stats.hdf5', mode='r')

    max_feat = np.array(stat_file["feats_maximus"])
    min_feat = np.array(stat_file["feats_minimus"])

    stat_file.close()


    max_files_to_process = int(config.batch_size/config.samples_per_file)

    if mode == "Train":
        num_batches = config.batches_per_epoch_train
        file_list = trn_list

    else: 
        num_batches = config.batches_per_epoch_val
        file_list = val_list

    for k in range(num_batches):
        conds = []

        voc_out = []


        for i in range(max_files_to_process):


            voc_index = np.random.randint(0,len(file_list))
            voc_to_open = file_list[voc_index]


            voc_file = h5py.File(config.voice_dir+voc_to_open, "r")

            singer_name = voc_to_open.split('_')[1]
            singer_index = config.singers.index(singer_name)

            feats = np.array(voc_file['feats'])

            if np.isnan(feats).any():
                print("nan found")
                import pdb;pdb.set_trace()

            f0 = feats[:,-2]

            med = np.median(f0[f0 > 0])

            f0[f0==0] = med

            f0_nor = (f0 - min_feat[-2])/(max_feat[-2]-min_feat[-2])

            pho_target = voc_file["phonemes"]


            for j in range(config.samples_per_file):
                    voc_idx = np.random.randint(0,len(feats)-config.max_phr_len)

                    condi = pho_target[voc_idx:voc_idx+config.max_phr_len, :]

                    prev = one_hotize(condi[:,0])
                    cur = one_hotize(condi[:,1])
                    nex = one_hotize(condi[:,2])
                    sings = one_hotize(np.tile(singer_index, config.max_phr_len), config.num_singers)

                    condi = np.concatenate((prev, cur, nex, sings, condi[:,3:], np.expand_dims(f0_nor[voc_idx:voc_idx+config.max_phr_len], -1)), axis = -1)

                    conds.append(condi)

                    voc_out.append(feats[voc_idx:voc_idx+config.max_phr_len,:])


        conds = np.array(conds)

        voc_out = (np.array(voc_out) - min_feat)/(max_feat - min_feat)

        voc_out = voc_out[:,:,:-2]



        yield conds, voc_out


def get_stats():
    voc_list = [x for x in os.listdir(config.voice_dir) if x.endswith('.hdf5') and x.startswith('casas') and not x.startswith('casasros')]

    back_list = [x for x in os.listdir(config.backing_dir) if x.endswith('.hdf5') and not x.startswith('._') and not x.startswith('mir') and not x.startswith('med')]

    do_no_use = []

    max_feat = np.zeros(66)
    min_feat = np.ones(66)*1000

    max_voc = np.zeros(513)
    min_voc = np.ones(513)*1000

    max_mix = np.zeros(513)
    min_mix = np.ones(513)*1000    

    for voc_to_open in voc_list:

        voc_file = h5py.File(config.voice_dir+voc_to_open, "r")

        if not 'feats' in voc_file:
            do_no_use.append(voc_to_open)

    #     voc_stft = voc_file['voc_stft']

    #     feats = np.array(voc_file['feats'])

    #     f0 = feats[:,-2]

    #     med = np.median(f0[f0 > 0])

    #     f0[f0==0] = med

    #     feats[:,-2] = f0

    #     if np.isnan(feats).any():
    #         do_no_use.append(voc_to_open)

    #     maxi_voc_feat = np.array(feats).max(axis=0)

    #     for i in range(len(maxi_voc_feat)):
    #         if maxi_voc_feat[i]>max_feat[i]:
    #             max_feat[i] = maxi_voc_feat[i]

    #     mini_voc_feat = np.array(feats).min(axis=0)

    #     for i in range(len(mini_voc_feat)):
    #         if mini_voc_feat[i]<min_feat[i]:
    #             min_feat[i] = mini_voc_feat[i]   


    # hdf5_file = h5py.File(config.stat_dir+'stats.hdf5', mode='w')

    # hdf5_file.create_dataset("feats_maximus", [66], np.float32) 
    # hdf5_file.create_dataset("feats_minimus", [66], np.float32)   


    # hdf5_file["feats_maximus"][:] = max_feat
    # hdf5_file["feats_minimus"][:] = min_feat
    # hdf5_file.close()


    import pdb;pdb.set_trace()



def get_stats_phonems():

    phon=collections.Counter([])

    voc_list = [x for x in os.listdir(config.voice_dir) if x.endswith('.hdf5') and x.startswith('nus') and not x.startswith('nus_KENN') and not x == 'nus_MCUR_read_17.hdf5']

    for voc_to_open in voc_list:

        voc_file = h5py.File(config.voice_dir+voc_to_open, "r")
        pho_target = np.array(voc_file["phonemes"])
        phon += collections.Counter(pho_target)
    phonemas_weights = np.zeros(41)
    for pho in phon:
        phonemas_weights[pho] = phon[pho]

    phonemas_above_threshold = [config.phonemas[x[0]] for x in np.argwhere(phonemas_weights>70000)]

    pho_order = phonemas_weights.argsort()

    # phonemas_weights = 1.0/phonemas_weights
    # phonemas_weights = phonemas_weights/sum(phonemas_weights)
    import pdb;pdb.set_trace()


def main():
    # gen_train_val()
    get_stats()
    gen = data_gen_full('val', sec_mode = 0)
    while True :
        start_time = time.time()
        conds, voc_out = next(gen)
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
