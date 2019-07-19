import tensorflow as tf
import modules_tf as modules
import config
from data_pipeline import data_gen
import time, os
import utils
import h5py
import numpy as np
import mir_eval
import pandas as pd
from random import randint
import librosa
# import sig_process

import soundfile as sf

import matplotlib.pyplot as plt
from scipy.ndimage import filters

def cgm_crossentropy(y_true, y_pred):
    # assume y_pred is of size (batch_size,nframes,3*n), with y_pred(:,:,0:n)=mean, y_pred(:,:,n:2*n)=variance, y_pred(:,:,2*n:3*n)=value
    # assume y_true is of size (batch_size,nframes,1), with y_true(:,:,0)=value
    M = 4 # number of mixture components

    mean = y_pred[:,:,0:1]
    sigma = tf.exp(y_pred[:,:,1:2])/255.*2. # before it was std in [0,255] range, now in [-1,1] range 
    skew = y_pred[:,:,2:3] # [-1,1]
    gamma = y_pred[:,:,3:4] # [-1,1]

    two_pi = 2. * np.pi
    minsigma = 1/255.*2.
    norm = tf.sqrt(two_pi)*minsigma
    ku = 1.6;
    kw = 1./1.75*gamma;
    ks = 1.1;

    sigma2 = sigma*tf.exp(tf.abs(skew)*ks)/2.7183 # T.exp(1.)
    sigma3 = sigma2*tf.exp(tf.abs(skew)*ks)/2.7183 # T.exp(1.)
    sigma4 = sigma3*tf.exp(tf.abs(skew)*ks)/2.7183 # T.exp(1.)

    mean2 = mean + sigma*ku*skew;
    mean3 = mean2 + sigma2*ku*skew;
    mean4 = mean3 + sigma3*ku*skew;
    
    w = tf.ones_like(skew)
    w2 = w*kw*tf.square(skew)
    w3 = w2*kw*tf.square(skew)
    w4 = w3*kw*tf.square(skew)
    wc = tf.concat([w, w2, w3, w4],axis=-1)
    sw = tf.reduce_sum(wc,axis=-1,keep_dims=True) / norm

    zc = []
    zc.append( tf.log(w/sw + 1e-7) + (-tf.square(y_true[:,:,0:1]- mean[:,:,0:1])/2./tf.square( sigma[:,:,0:1])) - tf.log(tf.sqrt(two_pi)* sigma[:,:,0:1]))
    zc.append(tf.log(w2/sw + 1e-7) + (-tf.square(y_true[:,:,0:1]-mean2[:,:,0:1])/2./tf.square(sigma2[:,:,0:1])) - tf.log(tf.sqrt(two_pi)*sigma2[:,:,0:1]))
    zc.append(tf.log(w3/sw + 1e-7) + (-tf.square(y_true[:,:,0:1]-mean3[:,:,0:1])/2./tf.square(sigma3[:,:,0:1])) - tf.log(tf.sqrt(two_pi)*sigma3[:,:,0:1]))
    zc.append(tf.log(w4/sw + 1e-7) + (-tf.square(y_true[:,:,0:1]-mean4[:,:,0:1])/2./tf.square(sigma4[:,:,0:1])) - tf.log(tf.sqrt(two_pi)*sigma4[:,:,0:1]))
    z = tf.concat(zc,axis=-1)
    z_max = tf.reduce_max(z,axis=-1,keepdims=True)
    # z_max of size (:,:,1)
    for m in range(M):
        if m==0:
            zc = tf.exp(z[:,:,m:m+1]-z_max)
        else:
            zc += tf.exp(z[:,:,m:m+1]-z_max)
    loss = - (tf.log(zc)+z_max) # loss = sum( - y_true * log(y_pred)) --> -log(y_pred(true_index))

    n = 64
    if n>1:
        w = 1. - .5*np.arange(n)/(n-1)
        w = np.array(w).reshape((1, 1, len(w)))
        loss = tf.reshape(loss,(config.batch_size, config.max_phr_len, w.shape[-1])) * w
    #elif inet==0:
    #    # testing weight depending on y_true for X_harm
    #    w = 0.25 + 0.75*(1.+y_true[:,:,0:1])/2.
    #    w = w.reshape((batch_size_, ny_, n))
    #else:
    #    w = np.ones((n,))
    #    w = np.array(w, dtype=theano.config.floatX).reshape((1, 1, len(w)))

    return tf.reduce_mean(loss)


# def one_hotize(inp, max_index=config.num_phos):


#     output = np.eye(max_index)[inp.astype(int)]

#     return output

class Model(object):
    def __init__(self):
        self.get_placeholders()
        self.model()


    def test_file_all(self, file_name, sess):
        """
        Function to extract multi pitch from file. Currently supports only HDF5 files.
        """
        scores = self.extract_f0_file(file_name, sess)
        return scores

    def validate_file(self, file_name, sess):
        """
        Function to extract multi pitch from file, for validation. Currently supports only HDF5 files.
        """
        scores = self.extract_f0_file(file_name, sess)
        pre = scores['Precision']
        acc = scores['Accuracy']
        rec = scores['Recall']
        return pre, acc, rec




    def load_model(self, sess, log_dir):
        """
        Load model parameters, for synthesis or re-starting training. 
        """
        self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep= config.max_models_to_keep)


        sess.run(self.init_op)

        ckpt = tf.train.get_checkpoint_state(log_dir)

        if ckpt and ckpt.model_checkpoint_path:
            print("Using the model in %s"%ckpt.model_checkpoint_path)
            self.saver.restore(sess, ckpt.model_checkpoint_path)


    def save_model(self, sess, epoch, log_dir):
        """
        Save the model.
        """
        checkpoint_file = os.path.join(log_dir, 'model.ckpt')
        self.saver.save(sess, checkpoint_file, global_step=epoch)

    def print_summary(self, print_dict, epoch, duration):
        """
        Print training summary to console, every N epochs.
        Summary will depend on model_mode.
        """

        print('epoch %d took (%.3f sec)' % (epoch + 1, duration))
        for key, value in print_dict.items():
            print('{} : {}'.format(key, value))
            

class PercSynth(Model):

    def get_optimizers(self):
        """
        Returns the optimizers for the model, based on the loss functions and the mode. 
        """

        self.final_optimizer = tf.train.AdamOptimizer(learning_rate = config.init_lr)

        self.global_step = tf.Variable(0, name='global_step', trainable=False)



        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.final_train_function = self.final_optimizer.minimize(self.final_loss, global_step = self.global_step)


    def loss_function(self):
        """
        returns the loss function for the model, based on the mode. 
        """



        # self.final_loss = cgm_crossentropy(tf.reshape(self.output_placeholder, [config.batch_size, -1, 1]), tf.reshape(self.output, [config.batch_size, -1, 4]))
        # self.final_loss = tf.losses.mean_squared_error(self.output,self.output_placeholder )
        self.final_loss = tf.reduce_sum(tf.abs(self.output_placeholder- self.output) * self.input_placeholder)
        # tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels= self.output_placeholder, logits = self.output)) 
        # tf.reduce_sum(tf.abs(self.input_placeholder- self.output))

    def get_summary(self, sess, log_dir):
        """
        Gets the summaries and summary writers for the losses.
        """

        self.final_summary = tf.summary.scalar('final_loss', self.final_loss)


        self.train_summary_writer = tf.summary.FileWriter(log_dir+'train/', sess.graph)
        self.val_summary_writer = tf.summary.FileWriter(log_dir+'val/', sess.graph)
        self.summary = tf.summary.merge_all()

    def get_placeholders(self):
        """
        Returns the placeholders for the model. 
        Depending on the mode, can return placeholders for either just the generator or both the generator and discriminator.
        """

        self.input_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size, config.max_phr_len, 1),
                                           name='input_placeholder')

        self.cond_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size, config.input_features),
                                           name='cond_placeholder')

        self.output_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size, config.max_phr_len, 1),
                                           name='output_placeholder')       

        self.mask_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size, config.max_phr_len, 1),
                                           name='mask_placeholder')       

        self.is_train = tf.placeholder(tf.bool, name="is_train")


    def train(self):
        """
        Function to train the model, and save Tensorboard summary, for N epochs. 
        """
        sess = tf.Session()


        self.loss_function()
        self.get_optimizers()
        self.load_model(sess, config.log_dir)
        self.get_summary(sess, config.log_dir)
        start_epoch = int(sess.run(tf.train.get_global_step()) / (config.batches_per_epoch_train))


        print("Start from: %d" % start_epoch)


        for epoch in range(start_epoch, config.num_epochs):
            data_generator = data_gen()
            val_generator = data_gen(mode = 'Val')
            start_time = time.time()


            batch_num = 0
            epoch_final_loss = 0

            val_final_loss = 0




            with tf.variable_scope('Training'):
                for out_audios, out_envelopes, out_features, out_masks in data_generator:

                    final_loss, summary_str = self.train_model(out_audios, out_envelopes, out_features, out_masks, sess)


                    epoch_final_loss+=final_loss


                    self.train_summary_writer.add_summary(summary_str, epoch)
                    self.train_summary_writer.flush()

                    utils.progress(batch_num,config.batches_per_epoch_train, suffix = 'training done')

                    batch_num+=1

                epoch_final_loss = epoch_final_loss/batch_num


                print_dict = {"Final Loss": epoch_final_loss}


            if (epoch + 1) % config.validate_every == 0:
                batch_num = 0
                with tf.variable_scope('Validation'):
                    for out_audios, out_envelopes, out_features, out_masks in val_generator:


                        final_loss, summary_str= self.validate_model(out_audios, out_envelopes, out_features, out_masks, sess)
                        val_final_loss+=final_loss


                        self.val_summary_writer.add_summary(summary_str, epoch)
                        self.val_summary_writer.flush()
                        batch_num+=1

                        utils.progress(batch_num, config.batches_per_epoch_val, suffix='validation done')

                    val_final_loss = val_final_loss/batch_num


                    print_dict["Val Final Loss"] =  val_final_loss


            end_time = time.time()
            if (epoch + 1) % config.print_every == 0:
                self.print_summary(print_dict, epoch, end_time-start_time)
            if (epoch + 1) % config.save_every == 0 or (epoch + 1) == config.num_epochs:
                self.save_model(sess, epoch+1, config.log_dir)

    def train_model(self, out_audios, out_envelopes, out_features, out_masks, sess):
        """
        Function to train the model for each epoch
        """
        feed_dict = {self.input_placeholder: out_envelopes,self.output_placeholder: out_audios, self.cond_placeholder: out_features, self.mask_placeholder:out_masks, self.is_train: True}

        _,final_loss= sess.run([self.final_train_function, self.final_loss], feed_dict=feed_dict)

        summary_str = sess.run(self.summary, feed_dict=feed_dict)

        return final_loss, summary_str

    def validate_model(self, out_audios, out_envelopes, out_features, out_masks, sess):
        """
        Function to train the model for each epoch
        """
        feed_dict = {self.input_placeholder: out_envelopes,self.output_placeholder: out_audios, self.cond_placeholder: out_features, self.mask_placeholder:out_masks, self.is_train: False}

        final_loss= sess.run(self.final_loss, feed_dict=feed_dict)

        summary_str = sess.run(self.summary, feed_dict=feed_dict)

        return final_loss, summary_str

    def test_model(self):
        sess = tf.Session()
        self.load_model(sess, log_dir = config.log_dir)
        val_generator = data_gen()
        out_audios, out_envelopes, out_features, out_masks = next(val_generator)
        feed_dict = {self.input_placeholder: out_envelopes,self.output_placeholder: out_audios, self.cond_placeholder: out_features, self.mask_placeholder:out_masks, self.is_train: False}
        output = sess.run(self.output, feed_dict=feed_dict)
        plt.subplot(311)
        plt.plot(output[0])
        plt.subplot(312)
        plt.plot(out_audios[0])
        plt.subplot(313)
        plt.plot(out_envelopes[0])
        plt.show()
        import pdb;pdb.set_trace()





    def read_hdf5_file(self, file_name):
        """
        Function to read and process input file, given name and the synth_mode.
        Returns features for the file based on mode (0 for hdf5 file, 1 for wav file).
        Currently, only the HDF5 version is implemented.
        """
        # if file_name.endswith('.hdf5'):
        stat_file = h5py.File(config.stat_dir+'stats.hdf5', mode='r')

        max_feat = np.array(stat_file["feats_maximus"])
        min_feat = np.array(stat_file["feats_minimus"])
        stat_file.close()

        with h5py.File(config.voice_dir + file_name) as feat_file:

            pho_target = np.array(feat_file["phonemes"])
            feats = np.array(feat_file['feats'])

        singer_name = file_name.split('_')[1]
        if singer_name in config.singers:
            singer_index = config.singers.index(singer_name)
        else:
            singer_index = 0
            # np.random.randint(len(config.singers))
        f0 = feats[:,-2]

        med = np.median(f0[f0 > 0])

        f0[f0==0] = med

        f0_nor = (f0 - min_feat[-2])/(max_feat[-2]-min_feat[-2])

        prev = one_hotize(pho_target[:,0])
        cur = one_hotize(pho_target[:,1])
        nex = one_hotize(pho_target[:,2])

        sings = one_hotize(np.tile(singer_index, pho_target.shape[0]), config.num_singers)

        # import pdb;pdb.set_trace()

        condi = np.concatenate((prev, cur, nex, sings, pho_target[:,3:], np.expand_dims(f0_nor, -1)), axis = -1)

        return condi, feats, singer_index

    def test_file_hdf5(self, file_name):
        """
        Function to extract multi pitch from file. Currently supports only HDF5 files.
        """
        sess = tf.Session()
        self.load_model(sess, log_dir = config.log_dir)
        condi, feats, singer_id = self.read_hdf5_file(file_name)

        out_feats = self.process_file(condi, sess)

        self.plot_features(feats, out_feats)

        # import pdb;pdb.set_trace()

        out_featss = np.concatenate((out_feats[:feats.shape[0]], feats[:,-2:]), axis = -1)

        utils.feats_to_audio(out_featss,file_name[:-4]+'_'+str(config.singers[singer_id])+'.wav') 

    def plot_features(self, feats, out_feats):

        plt.figure(1)
        
        ax1 = plt.subplot(211)

        plt.imshow(feats[:,:-2].T,aspect='auto',origin='lower')

        ax1.set_title("Ground Truth STFT", fontsize=10)

        ax3 =plt.subplot(212, sharex = ax1, sharey = ax1)

        ax3.set_title("Output STFT", fontsize=10)

        plt.imshow(out_feats.T,aspect='auto',origin='lower')


        plt.show()


    def process_file(self, condi, sess):

        stat_file = h5py.File(config.stat_dir+'stats.hdf5', mode='r')

        max_feat = np.array(stat_file["feats_maximus"])
        min_feat = np.array(stat_file["feats_minimus"])
        stat_file.close()

        conds = np.zeros((config.batch_size, config.max_phr_len, config.input_features))
        outs = []
        inps = np.zeros((config.batch_size, config.max_phr_len, config.output_features))

        count = 0

        for con in condi:
            conds = np.roll(conds, -1, 1)
            conds[:,-1, :] = con
            feed_dict = {self.input_placeholder: inps ,self.cond_placeholder: conds, self.is_train: False}
            frame_op = sess.run(self.output, feed_dict=feed_dict)
            outs.append(frame_op[0,-1,:])
            inps = np.roll(inps, -1, 1)
            inps[:,-1,:] = frame_op[:,-1,:]
            count+=1
            utils.progress(count,len(condi), suffix = 'Done')
        outs = np.array(outs)
        outs = outs*(max_feat[:-2] - min_feat[:-2]) + min_feat[:-2]
        return np.array(outs)



    def model(self):
        """
        The main model function, takes and returns tensors.
        Defined in modules.

        """

        with tf.variable_scope('NPSS') as scope:
            self.output = modules.full_network(self.cond_placeholder,self.input_placeholder,  self.is_train)



def test():
    # model = DeepSal()
    # # model.test_file('nino_4424.hdf5')
    # model.test_wav_folder('./helena_test_set/', './results/')

    model = MultiSynth()
    model.train()

if __name__ == '__main__':
    test()





