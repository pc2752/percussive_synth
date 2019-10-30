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
import json
# import sig_process
from scipy import signal

import soundfile as sf

import matplotlib.pyplot as plt
from scipy.ndimage import filters
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

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
            

class PercSynthGAN(Model):

    def get_optimizers(self):
        """
        Returns the optimizers for the model, based on the loss functions and the mode. 
        """

        self.final_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope = 'Final_Model')
        self.d_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope = 'Discriminator')

        self.final_optimizer = tf.train.RMSPropOptimizer(learning_rate=5e-5)
        self.dis_optimizer = tf.train.RMSPropOptimizer(learning_rate=5e-5)


        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.global_step_dis = tf.Variable(0, name='dis_global_step', trainable=False)



        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.final_train_function = self.final_optimizer.minimize(self.final_loss, global_step = self.global_step, var_list = self.final_params)
            self.dis_train_function = self.dis_optimizer.minimize(self.D_loss, global_step = self.global_step_dis, var_list = self.d_params)
            self.clip_discriminator_var_op = [var.assign(tf.clip_by_value(var, -0.01, 0.01)) for var in self.d_params]

    def loss_function(self):
        """
        returns the loss function for the model, based on the mode. 
        """



        # self.final_loss = cgm_crossentropy(tf.reshape(self.output_placeholder, [config.batch_size, -1, 1]), tf.reshape(self.output, [config.batch_size, -1, 4]))
        # self.final_loss = tf.losses.mean_squared_error(self.output,self.output_placeholder )
        # self.final_loss = tf.reduce_sum(tf.abs(self.output_placeholder- self.output) * self.input_placeholder)
        self.final_loss = tf.reduce_sum(tf.abs(self.output_placeholder- self.output) * self.input_placeholder)/(config.batch_size*config.max_phr_len) + tf.reduce_mean(self.D_fake+1e-12)

        self.D_loss = tf.reduce_mean(self.D_real +1e-12) - tf.reduce_mean(self.D_fake+1e-12)

    def get_summary(self, sess, log_dir):
        """
        Gets the summaries and summary writers for the losses.
        """

        self.final_summary = tf.summary.scalar('final_loss', self.final_loss)

        self.dis_summary = tf.summary.scalar('dis_loss', self.D_loss)




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
            epoch_dis_loss = 0

            val_final_loss = 0
            val_dis_loss = 0




            with tf.variable_scope('Training'):
                for out_audios, out_envelopes, out_features, out_masks in data_generator:

                    final_loss, dis_loss, summary_str = self.train_model(out_audios, out_envelopes, out_features, out_masks, epoch, sess)



                    epoch_final_loss+=final_loss
                    epoch_dis_loss+=dis_loss


                    self.train_summary_writer.add_summary(summary_str, epoch)
                    self.train_summary_writer.flush()

                    utils.progress(batch_num,config.batches_per_epoch_train, suffix = 'training done')

                    batch_num+=1

                epoch_final_loss = epoch_final_loss/batch_num
                epoch_dis_loss = epoch_dis_loss/batch_num


                print_dict = {"Final Loss": epoch_final_loss}
                print_dict["Dis Loss"] =  epoch_dis_loss


            if (epoch + 1) % config.validate_every == 0:
                batch_num = 0
                with tf.variable_scope('Validation'):
                    for out_audios, out_envelopes, out_features, out_masks in val_generator:


                        final_loss, dis_loss, summary_str= self.validate_model(out_audios, out_envelopes, out_features, out_masks, sess)
                        val_final_loss+=final_loss
                        val_dis_loss+=dis_loss


                        self.val_summary_writer.add_summary(summary_str, epoch)
                        self.val_summary_writer.flush()
                        batch_num+=1

                        utils.progress(batch_num, config.batches_per_epoch_val, suffix='validation done')

                    val_final_loss = val_final_loss/batch_num
                    val_dis_loss = val_dis_loss/batch_num


                    print_dict["Val Final Loss"] =  val_final_loss
                    print_dict["Val Dis Loss"] =  val_dis_loss


            end_time = time.time()
            if (epoch + 1) % config.print_every == 0:
                self.print_summary(print_dict, epoch, end_time-start_time)
            if (epoch + 1) % config.save_every == 0 or (epoch + 1) == config.num_epochs:
                self.save_model(sess, epoch+1, config.log_dir)

    def train_model(self, out_audios, out_envelopes, out_features, out_masks, epoch,  sess):
        """
        Function to train the model for each epoch
        """
        if epoch<25 or epoch%100 == 0:
            n_critic = 25
        else:
            n_critic = 5
        feed_dict = {self.input_placeholder: out_envelopes,self.output_placeholder: out_audios, self.cond_placeholder: out_features, self.mask_placeholder:out_masks, self.is_train: True}

        for critic_itr in range(n_critic):
            sess.run(self.dis_train_function, feed_dict = feed_dict)
            sess.run(self.clip_discriminator_var_op, feed_dict = feed_dict)

        _,final_loss, dis_loss = sess.run([self.final_train_function, self.final_loss, self.D_loss], feed_dict=feed_dict)

        summary_str = sess.run(self.summary, feed_dict=feed_dict)

        return final_loss, dis_loss, summary_str

    def validate_model(self, out_audios, out_envelopes, out_features, out_masks, sess):
        """
        Function to train the model for each epoch
        """
        feed_dict = {self.input_placeholder: out_envelopes,self.output_placeholder: out_audios, self.cond_placeholder: out_features, self.mask_placeholder:out_masks, self.is_train: False}

        final_loss, dis_loss= sess.run([self.final_loss, self.D_loss], feed_dict=feed_dict)

        summary_str = sess.run(self.summary, feed_dict=feed_dict)

        return final_loss, dis_loss, summary_str

    def test_model(self):
        sess = tf.Session()
        self.load_model(sess, log_dir = config.log_dir)
        val_generator = data_gen(mode = 'val')
        out_audios, out_envelopes, out_features, out_masks = next(val_generator)

        feed_dict = {self.input_placeholder: out_envelopes,self.output_placeholder: out_audios, self.cond_placeholder: out_features, self.mask_placeholder:out_masks, self.is_train: False}
        output = sess.run(self.output, feed_dict=feed_dict)
        # feed_dict = {self.input_placeholder: out_envelopes,self.output_placeholder: out_audios, self.cond_placeholder: np.ones(out_features.shape)*0.1  , self.mask_placeholder:out_masks, self.is_train: False}
        # output_1 = sess.run(self.output, feed_dict=feed_dict)
        # mod_feats = np.ones(out_features.shape)*0.1 
        # mod_feats[:,0] = 0.9
        # feed_dict = {self.input_placeholder: out_envelopes,self.output_placeholder: out_audios, self.cond_placeholder: mod_feats, self.mask_placeholder:out_masks, self.is_train: False}
        # output_2 = sess.run(self.output, feed_dict=feed_dict)

        output = output * out_envelopes

        # output_1 = output_1 * out_envelopes

        # output_2 = output_2 * out_envelopes

        for i in range(config.batch_size):
            print( [str(out_features[i][x])+":"+config.feats_to_use[x] for x in range(len(config.feats_to_use))])
            ax1 = plt.subplot(311)
            ax1.set_title("Output Waveform", fontsize=10)
            plt.plot(np.clip(output[i][:14000], -1.0,1.0))
            # ax2 = plt.subplot(512)
            # ax2.set_title("Output Waveform 0.1", fontsize=10)
            # plt.plot(np.clip(output_1[i][:14000], -1.0,1.0))
            # ax2 = plt.subplot(513)
            # ax2.set_title("Output Waveform 0.9", fontsize=10)
            # plt.plot(np.clip(output_2[i][:14000], -1.0,1.0))
            ax2 = plt.subplot(312)
            ax2.set_title("Ground Truth Waveform", fontsize=10)
            plt.plot(out_audios[i])
            ax3 = plt.subplot(313)
            ax3.set_title("Input Envelope", fontsize=10)
            plt.plot(out_envelopes[i])
            # synth = utils.query_yes_no("Save Plot?")

            # if synth:
            # plt.savefig('./fig_{}'.format(i))

            plt.show()
            # synth = utils.query_yes_no("Synthesize output? ")


            # if synth:

            sf.write('./op_{}.wav'.format(i), np.clip(output[i][:14000], -1.0,1.0), config.fs)
            # sf.write('./op0.1_{}.wav'.format(i), np.clip(output_1[i][:14000], -1.0,1.0), config.fs)
            # sf.write('./op0.2_{}.wav'.format(i), np.clip(output_2[i][:14000], -1.0,1.0), config.fs)
            # synth = utils.query_yes_no("Synthesize  Ground Truth? ")

            # if synth:

            sf.write('./gt_{}.wav'.format(i), out_audios[i], config.fs)
        # ax4 = plt.subplot(414)
        # ax4.set_title("Input Envelope", fontsize=10)
        # plt.plot(out_masks[0])
            
            # import pdb;pdb.set_trace()





  



    def model(self):
        """
        The main model function, takes and returns tensors.
        Defined in modules.

        """

        with tf.variable_scope('Final_Model') as scope:
            self.output = modules.full_network(self.cond_placeholder,self.input_placeholder,  self.is_train)

        with tf.variable_scope('Discriminator') as scope: 
            self.D_real = modules.discriminator(self.output_placeholder* self.input_placeholder, self.cond_placeholder,  self.is_train)
            scope.reuse_variables()
            self.D_fake = modules.discriminator(self.output* self.input_placeholder, self.cond_placeholder,  self.is_train)

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
        self.output_stft = tf.abs(tf.contrib.signal.stft(tf.squeeze(self.output), 1024, 512))
        self.output_stft_placeholder = tf.abs(tf.contrib.signal.stft(tf.squeeze(self.output_placeholder), 1024, 512))
        self.final_loss = tf.reduce_sum(tf.abs(self.output_placeholder- self.output) * self.input_placeholder) + 0.5*tf.reduce_sum(tf.abs(self.output_stft_placeholder - self.output_stft)) 
        self.show_loss = (tf.abs(self.output_placeholder- self.output) * self.input_placeholder)
        # tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels= self.output_placeholder, logits = self.output)) 
        # tf.reduce_sum(tf.abs(self.input_placeholder- self.output))

    def get_summary(self, sess, log_dir):
        """
        Gets the summaries and summary writers for the losses.
        """

        self.final_summary = tf.summary.scalar('final_loss', self.final_loss)

        self.summary_audio = tf.summary.audio('audios', self.output, config.fs)


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

    def export_model(self):
        sess = tf.Session()
        self.load_model(sess, log_dir = config.log_dir)
        builder = tf.saved_model.builder.SavedModelBuilder('./export_model/')
        signature = predict_signature_def(inputs={'input_placeholder': self.input_placeholder, 'cond_placeholder': self.cond_placeholder, "is_train": self.is_train},
                                      outputs={'output': self.output})
        print(signature)
        builder.add_meta_graph_and_variables(sess,
                                         ["predict"], signature_def_map={'predict': signature},
                                         strip_default_attrs=True)
        builder.save() 


    # def test_model(self):
    #     sess = tf.Session()
    #     self.load_model(sess, log_dir = config.log_dir)
    #     self.loss_function()
    #     val_generator = data_gen(mode = 'val')
    #     batch_num = 0
    #     out_dict = {}
    #     for out_audios, out_envelopes, out_features, out_masks, out_files in val_generator:
        
    #         feed_dict = {self.input_placeholder: out_envelopes,self.output_placeholder: out_audios, self.cond_placeholder: out_features, self.mask_placeholder:out_masks, self.is_train: False}
    #         output, losses = sess.run([self.output, self.show_loss], feed_dict=feed_dict)
    #         output = output* out_envelopes

    #         for i in range(config.batch_size):
    #             sf.write('./outputs_kicks_full/op_{}_{}.wav'.format(batch_num,i), np.clip(output[i], -1.0,1.0), config.fs)
    #             sf.write('./outputs_kicks_full/gt_{}_{}.wav'.format(batch_num,i), out_audios[i], config.fs)
    #             out_dict['op_{}_{}_.wav'.format(batch_num,i)] = out_files[i]

    #         # for j, feat in enumerate(config.feats_to_use):
    #         #     feats_low = np.copy(out_features)
    #         #     feats_low[:,j] = 0.2
    #         #     feed_dict = {self.input_placeholder: out_envelopes,self.output_placeholder: out_audios, self.cond_placeholder: feats_low, self.mask_placeholder:out_masks, self.is_train: False}
    #         #     output_low = sess.run(self.output, feed_dict=feed_dict)
    #         #     output_low = output_low* out_envelopes
    #         #     for i in range(config.batch_size):
    #         #         sf.write('./outputs_full/op_{}_{}_{}_low.wav'.format(batch_num, i, feat), np.clip(output_low[i], -1.0,1.0), config.fs)

    #         #     feats_mid = np.copy(out_features)
    #         #     feats_mid[:,j] = 0.5
    #         #     feed_dict = {self.input_placeholder: out_envelopes,self.output_placeholder: out_audios, self.cond_placeholder: feats_mid, self.mask_placeholder:out_masks, self.is_train: False}
    #         #     output_mid = sess.run(self.output, feed_dict=feed_dict)
    #         #     output_mid = output_mid* out_envelopes
    #         #     for i in range(config.batch_size):
    #         #         sf.write('./outputs_full/op_{}_{}_{}_mid.wav'.format(batch_num,i, feat), np.clip(output_mid[i], -1.0,1.0), config.fs)


    #         #     feats_high = np.copy(out_features)
    #         #     feats_high[:,j] = 0.8
    #         #     feed_dict = {self.input_placeholder: out_envelopes,self.output_placeholder: out_audios, self.cond_placeholder: feats_high, self.mask_placeholder:out_masks, self.is_train: False}
    #         #     output_high = sess.run(self.output, feed_dict=feed_dict)
    #         #     output_high = output_high* out_envelopes
    #         #     for i in range(config.batch_size):
    #         #         sf.write('./outputs_full/op_{}_{}_{}_high.wav'.format(batch_num,i, feat), np.clip(output_high[i], -1.0,1.0), config.fs)
                    
    #         batch_num+=1
    #         utils.progress(batch_num, config.batches_per_epoch_val, suffix='evaluation done')
    #     with open('mapping_kicks_full.json', 'w') as fp:
    #         json.dump(out_dict, fp)


    def load_sess(self, log_dir=config.log_dir):
        sess = tf.Session()
        self.load_model(sess, log_dir = log_dir)
        return sess

    def get_output(self, envelope, features, sess):

        envelope = np.repeat(envelope.reshape(1,-1), config.batch_size, 0).reshape(config.batch_size, -1, 1)
        features = np.repeat(features.reshape(1,-1), config.batch_size, 0)
        feed_dict = {self.input_placeholder: envelope,self.cond_placeholder: features,  self.is_train: False}
        output_full = sess.run(self.output, feed_dict=feed_dict)
        output = output_full[0]
        return output

    def test_model(self):
        sess = tf.Session()
        self.load_model(sess, log_dir = config.log_dir)
        self.loss_function()

        # with h5py.File(config.feats_dir+'feats.hdf5', mode='r') as hdf5_file:
        #     audios = hdf5_file["waveform"][()]
        #     envelope = hdf5_file["envelope"][()]
        #     mask = hdf5_file["mask"][()]
        #     features = hdf5_file["features"][()]

        # num_batches = len(audios)/config.batch_size

        # out_losses = []
        # out_sounds = []

        # for i in range(int(num_batches)):
        #     feed_dict = {self.input_placeholder: np.expand_dims(envelope[i*config.batch_size:(i+1)*config.batch_size], -1), self.output_placeholder: np.expand_dims(audios[i*config.batch_size:(i+1)*config.batch_size], -1), self.cond_placeholder: features[i*config.batch_size:(i+1)*config.batch_size], self.mask_placeholder: np.expand_dims(mask[i*config.batch_size:(i+1)*config.batch_size], -1), self.is_train: False}
        #     output, losses = sess.run([self.output, self.show_loss], feed_dict=feed_dict)
        #     if i == 0:

        #         out_losses.append(losses.sum(axis = -1).sum(axis=-1))
        #         out_losses = np.squeeze(np.array(out_losses))
        #     else:
        #         # import pdb;pdb.set_trace()
        #         out_losses = np.concatenate((out_losses, losses.sum(axis = -1).sum(axis=-1)), axis = 0)
        #     utils.progress(i, int(num_batches), suffix = 'Processed')

        # import pdb;pdb.set_trace()



        val_generator = data_gen(mode = 'val')
        count_batch = 0
        for out_audios, out_envelopes, out_features, out_masks in val_generator: 

            if count_batch in [0,1,2,3,4,5]:

                bobo = np.copy(out_features)

                feed_dict = {self.input_placeholder: out_envelopes,self.output_placeholder: out_audios, self.cond_placeholder: out_features, self.mask_placeholder:out_masks, self.is_train: False}
                output_full = sess.run(self.output, feed_dict=feed_dict)

                self.load_model(sess, log_dir = './log_kicks_high/')
                output_high = sess.run(self.output, feed_dict=feed_dict)

                self.load_model(sess, log_dir = './log_kicks/')
                output_wave = sess.run(self.output, feed_dict=feed_dict)

                feats = np.repeat(out_features,config.max_phr_len, -1).reshape([config.batch_size, config.max_phr_len,-1])

                # out_features[:,0] = 0.2



                # feed_dict = {self.input_placeholder: out_envelopes, self.cond_placeholder: out_features, self.mask_placeholder:out_masks, self.is_train: False}
                # output_low = sess.run(self.output, feed_dict=feed_dict)
                # output_low = output_low 

                # out_features[:,0] =  0.8

                # feed_dict = {self.input_placeholder: out_envelopes, self.cond_placeholder: out_features, self.mask_placeholder:out_masks, self.is_train: False}
                # output_bright = sess.run(self.output, feed_dict=feed_dict)        

                # output = output * out_envelopes

                # output_bright = output_bright * out_envelopes

                # output_low = output_low * out_envelopes

                for i in range(config.batch_size):
                    if "{}_{}".format(count_batch, i) in ["0_11", "0_15", "1_1", "1_5", "1_11", "1_13", "2_5", "2_11", "2_12", "3_1", "3_6", "3_9", "3_10", "4_7", "5_0"]:
                        # # print("Loss: {}".format(np.mean(losses[i])))
                        # # print([str(bobo[i][x])+":"+config.feats_to_use[x] for x in range(len(config.feats_to_use))])
                        # plt.subplots_adjust(hspace = 0.3, wspace=0.05)
                        # ax1 = plt.subplot(233)
                        # ax1.set_title("Input Envelope", fontsize=15)
                        # ax1.set_xticklabels([])
                        # ax1.set_yticklabels([])
                        # plt.plot(out_envelopes[i][:6000])
                        # ax1 = plt.subplot(232)
                        # ax1.set_title("Input Timbral Features", fontsize=15)
                        # ax1.set_xticklabels([])
                        # ax1.set_yticklabels([])
                        # ax1.barh(np.arange(len(config.feats_to_use)), out_features[i], align='center')
                        # # ax1.set_xticklabels([])
                        # # ax1.set_yticklabels(['']+config.feats_to_use)
                        # # ax1.set_yticklabels([])
                        # # plt.imshow(feats[i][:6000], origin='lower', aspect='auto')
                        

                        # ax2 = plt.subplot(231)
                        # ax2.set_title("Ground Truth Waveform", fontsize=15)
                        # ax2.set_xticklabels([])
                        # ax2.set_yticklabels([])
                        # plt.plot(out_audios[i][:6000])
                        # ax3 = plt.subplot(234)
                        # ax3.set_title("Output Waveform WAVE", fontsize=15)
                        # ax3.set_xticklabels([])
                        # ax3.set_yticklabels([])
                        # plt.plot(np.clip(output_wave[i][:6000], -1.0,1.0))
                        # ax4 = plt.subplot(235)
                        # ax4.set_title("Output Waveform FULL", fontsize=15)
                        # ax4.set_xticklabels([])
                        # ax4.set_yticklabels([])
                        # plt.plot(np.clip(output_full[i][:6000], -1.0,1.0))
                        # ax5 = plt.subplot(236)
                        # ax5.set_title("Output Waveform HIGH", fontsize=15)
                        # ax5.set_xticklabels([])
                        # ax5.set_yticklabels([])
                        # plt.plot(np.clip(output_high[i][:6000], -1.0,1.0))
                        # plt.show()
                        # # synth = utils.query_yes_no("Synthesize output? ")


                        # # if synth:

                        # # sf.write('./op_{}.wav'.format(i), np.clip(output[i][:14000], -1.0,1.0), config.fs)

                        # # sf.write('./op_{}_bright.wav'.format(i), np.clip(output_bright[i][:14000], -1.0,1.0), config.fs)
                        # # sf.write('./op_{}_low.wav'.format(i), np.clip(output_low[i][:14000], -1.0,1.0), config.fs)
                        # # sf.write('./op0.1_{}.wav'.format(i), np.clip(output_1[i][:14000], -1.0,1.0), config.fs)
                        # # sf.write('./op0.2_{}.wav'.format(i), np.clip(output_2[i][:14000], -1.0,1.0), config.fs)
                        # # synth = utils.query_yes_no("Synthesize  Ground Truth? ")

                        # # if synth:

                        sf.write('./demo_kicks/gt_{}_{}.wav'.format(count_batch, i), out_audios[i], config.fs)
                        sf.write('./demo_kicks/op_WAVE_{}_{}.wav'.format(count_batch, i), output_wave[i], config.fs)
                        sf.write('./demo_kicks/op_FULL_{}_{}.wav'.format(count_batch, i), output_full[i], config.fs)
                        sf.write('./demo_kicks/op_HIGH_{}_{}.wav'.format(count_batch, i), output_high[i], config.fs)
                    # ax4 = plt.subplot(414)
                    # ax4.set_title("Input Envelope", fontsize=10)
                    # plt.plot(out_masks[0])
                        
                        # import pdb;pdb.set_trace()
            count_batch+=1



    def model(self):
        """
        The main model function, takes and returns tensors.
        Defined in modules.
        """

        with tf.variable_scope('NPSS') as scope:
           # scope.reuse_variables()
            self.output = modules.full_network(self.cond_placeholder,self.input_placeholder,  self.is_train)

class PercSynthEncode(Model):

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
        self.loss_function()

        # with h5py.File(config.feats_dir+'feats.hdf5', mode='r') as hdf5_file:
        #     audios = hdf5_file["waveform"][()]
        #     envelope = hdf5_file["envelope"][()]
        #     mask = hdf5_file["mask"][()]
        #     features = hdf5_file["features"][()]

        # num_batches = len(audios)/config.batch_size

        # out_losses = []
        # out_sounds = []

        # for i in range(int(num_batches)):
        #     feed_dict = {self.input_placeholder: np.expand_dims(envelope[i*config.batch_size:(i+1)*config.batch_size], -1), self.output_placeholder: np.expand_dims(audios[i*config.batch_size:(i+1)*config.batch_size], -1), self.cond_placeholder: features[i*config.batch_size:(i+1)*config.batch_size], self.mask_placeholder: np.expand_dims(mask[i*config.batch_size:(i+1)*config.batch_size], -1), self.is_train: False}
        #     output, losses = sess.run([self.output, self.show_loss], feed_dict=feed_dict)
        #     if i == 0:

        #         out_losses.append(losses.sum(axis = -1).sum(axis=-1))
        #         out_losses = np.squeeze(np.array(out_losses))
        #     else:
        #         # import pdb;pdb.set_trace()
        #         out_losses = np.concatenate((out_losses, losses.sum(axis = -1).sum(axis=-1)), axis = 0)
        #     utils.progress(i, int(num_batches), suffix = 'Processed')

        # import pdb;pdb.set_trace()



        val_generator = data_gen(mode = 'val')
        out_audios, out_envelopes, out_features, out_masks = next(val_generator)

        bobo = out_features


        feed_dict = {self.input_placeholder: out_envelopes,self.output_placeholder: out_audios, self.cond_placeholder: out_features, self.mask_placeholder:out_masks, self.is_train: False}
        output = sess.run(self.output, feed_dict=feed_dict)

        out_features[:,0] = 0.2



        feed_dict = {self.input_placeholder: out_envelopes,self.output_placeholder: out_audios, self.cond_placeholder: out_features, self.mask_placeholder:out_masks, self.is_train: False}
        output_low = sess.run(self.output, feed_dict=feed_dict)

        out_features[:,0] =  0.8

        feed_dict = {self.input_placeholder: out_envelopes,self.output_placeholder: out_audios, self.cond_placeholder: out_features, self.mask_placeholder:out_masks, self.is_train: False}
        output_bright = sess.run(self.output, feed_dict=feed_dict)        

        # output = output * out_envelopes

        # output_bright = output_bright * out_envelopes

        # output_low = output_low * out_envelopes

        for i in range(config.batch_size):
            # print("Loss: {}".format(np.mean(losses[i])))
            print([str(bobo[i][x])+":"+config.feats_to_use[x] for x in range(len(config.feats_to_use))])
            ax1 = plt.subplot(511)
            ax1.set_title("Output Waveform", fontsize=10)
            plt.plot(np.clip(output[i][:14000], -1.0,1.0))
            ax1 = plt.subplot(512)
            ax1.set_title("Output Waveform Bright", fontsize=10)
            plt.plot(np.clip(output_bright[i][:14000], -1.0,1.0))
            ax1 = plt.subplot(513)
            ax1.set_title("Output Waveform Not Bright", fontsize=10)
            plt.plot(np.clip(output_low[i][:14000], -1.0,1.0))
            ax2 = plt.subplot(514)
            ax2.set_title("Ground Truth Waveform", fontsize=10)
            plt.plot(out_audios[i])
            ax3 = plt.subplot(515)
            ax3.set_title("Input Envelope", fontsize=10)
            plt.plot(out_envelopes[i])

            plt.show()
            # synth = utils.query_yes_no("Synthesize output? ")


            # if synth:

            sf.write('./op_{}.wav'.format(i), np.clip(output[i][:14000], -1.0,1.0), config.fs)

            sf.write('./op_{}_bright.wav'.format(i), np.clip(output_bright[i][:14000], -1.0,1.0), config.fs)
            sf.write('./op_{}_low.wav'.format(i), np.clip(output_low[i][:14000], -1.0,1.0), config.fs)
            # sf.write('./op0.1_{}.wav'.format(i), np.clip(output_1[i][:14000], -1.0,1.0), config.fs)
            # sf.write('./op0.2_{}.wav'.format(i), np.clip(output_2[i][:14000], -1.0,1.0), config.fs)
            # synth = utils.query_yes_no("Synthesize  Ground Truth? ")

            # if synth:

            sf.write('./gt_{}.wav'.format(i), out_audios[i], config.fs)
        # ax4 = plt.subplot(414)
        # ax4.set_title("Input Envelope", fontsize=10)
        # plt.plot(out_masks[0])
            
            import pdb;pdb.set_trace()





    def model(self):
        """
        The main model function, takes and returns tensors.
        Defined in modules.
        """

        with tf.variable_scope('Encoder') as scope:
            self.encoding = modules.content_encode(self.output_placeholder,  self.is_train)
        with tf.variable_scope('Decoder') as scope:
            self.output = modules.full_network_encode(self.encoding, self.cond_placeholder,self.input_placeholder,  self.is_train)
            # self.encoding_output = modules.content_encoder(self.output_placeholder,  self.is_train)


class PercSynthContent(Model):

    def get_optimizers(self):
        """
        Returns the optimizers for the model, based on the loss functions and the mode. 
        """

        self.final_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope = 'Final_Model')
        self.d_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope = 'Discriminator')

        self.final_optimizer = tf.train.AdamOptimizer(learning_rate = config.init_lr)
        self.dis_optimizer = tf.train.AdamOptimizer(learning_rate = config.init_lr)


        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.global_step_dis = tf.Variable(0, name='dis_global_step', trainable=False)



        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.final_train_function = self.final_optimizer.minimize(self.final_loss, global_step = self.global_step, var_list = self.final_params)
            self.dis_train_function = self.dis_optimizer.minimize(self.D_loss, global_step = self.global_step_dis, var_list = self.d_params)
            self.clip_discriminator_var_op = [var.assign(tf.clip_by_value(var, -0.01, 0.01)) for var in self.d_params]

    def loss_function(self):
        """
        returns the loss function for the model, based on the mode. 
        """



        # self.final_loss = cgm_crossentropy(tf.reshape(self.output_placeholder, [config.batch_size, -1, 1]), tf.reshape(self.output, [config.batch_size, -1, 4]))
        # self.final_loss = tf.losses.mean_squared_error(self.output,self.output_placeholder )
        # self.final_loss = tf.reduce_sum(tf.abs(self.output_placeholder- self.output) * self.input_placeholder)
        self.final_loss = tf.reduce_sum(tf.abs(self.output_placeholder- self.output) * self.input_placeholder)/(config.batch_size*config.max_phr_len)  + tf.reduce_mean(tf.abs(self.fake_emb - self.real_emb))
# + tf.reduce_mean(tf.abs(self.fake_cont - self.cond_placeholder))
        self.D_loss = tf.reduce_mean(tf.abs(self.real_cont - self.cond_placeholder))



    def get_summary(self, sess, log_dir):
        """
        Gets the summaries and summary writers for the losses.
        """

        self.final_summary = tf.summary.scalar('final_loss', self.final_loss)

        self.dis_summary = tf.summary.scalar('dis_loss', self.D_loss)




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
            epoch_dis_loss = 0

            val_final_loss = 0
            val_dis_loss = 0




            with tf.variable_scope('Training'):
                for out_audios, out_envelopes, out_features, out_masks in data_generator:

                    final_loss, dis_loss, summary_str = self.train_model(out_audios, out_envelopes, out_features, out_masks, epoch, sess)



                    epoch_final_loss+=final_loss
                    epoch_dis_loss+=dis_loss


                    self.train_summary_writer.add_summary(summary_str, epoch)
                    self.train_summary_writer.flush()

                    utils.progress(batch_num,config.batches_per_epoch_train, suffix = 'training done')

                    batch_num+=1

                epoch_final_loss = epoch_final_loss/batch_num
                epoch_dis_loss = epoch_dis_loss/batch_num


                print_dict = {"Final Loss": epoch_final_loss}
                print_dict["Dis Loss"] =  epoch_dis_loss


            if (epoch + 1) % config.validate_every == 0:
                batch_num = 0
                with tf.variable_scope('Validation'):
                    for out_audios, out_envelopes, out_features, out_masks in val_generator:


                        final_loss, dis_loss, summary_str= self.validate_model(out_audios, out_envelopes, out_features, out_masks, sess)
                        val_final_loss+=final_loss
                        val_dis_loss+=dis_loss


                        self.val_summary_writer.add_summary(summary_str, epoch)
                        self.val_summary_writer.flush()
                        batch_num+=1

                        utils.progress(batch_num, config.batches_per_epoch_val, suffix='validation done')

                    val_final_loss = val_final_loss/batch_num
                    val_dis_loss = val_dis_loss/batch_num


                    print_dict["Val Final Loss"] =  val_final_loss
                    print_dict["Val Dis Loss"] =  val_dis_loss


            end_time = time.time()
            if (epoch + 1) % config.print_every == 0:
                self.print_summary(print_dict, epoch, end_time-start_time)
            if (epoch + 1) % config.save_every == 0 or (epoch + 1) == config.num_epochs:
                self.save_model(sess, epoch+1, config.log_dir)

    def train_model(self, out_audios, out_envelopes, out_features, out_masks, epoch,  sess):
        """
        Function to train the model for each epoch
        """
        if epoch<25 or epoch%100 == 0:
            n_critic = 25
        else:
            n_critic = 5
        feed_dict = {self.input_placeholder: out_envelopes,self.output_placeholder: out_audios, self.cond_placeholder: out_features, self.mask_placeholder:out_masks, self.is_train: True}

        for critic_itr in range(n_critic):
            sess.run(self.dis_train_function, feed_dict = feed_dict)
            # sess.run(self.clip_discriminator_var_op, feed_dict = feed_dict)

        _,final_loss, dis_loss = sess.run([self.final_train_function, self.final_loss, self.D_loss], feed_dict=feed_dict)

        summary_str = sess.run(self.summary, feed_dict=feed_dict)

        return final_loss, dis_loss, summary_str

    def validate_model(self, out_audios, out_envelopes, out_features, out_masks, sess):
        """
        Function to train the model for each epoch
        """
        feed_dict = {self.input_placeholder: out_envelopes,self.output_placeholder: out_audios, self.cond_placeholder: out_features, self.mask_placeholder:out_masks, self.is_train: False}

        final_loss, dis_loss= sess.run([self.final_loss, self.D_loss], feed_dict=feed_dict)

        summary_str = sess.run(self.summary, feed_dict=feed_dict)

        return final_loss, dis_loss, summary_str

    def test_model(self):
        sess = tf.Session()
        self.load_model(sess, log_dir = config.log_dir)
        val_generator = data_gen(mode = 'val')
        out_audios, out_envelopes, out_features, out_masks = next(val_generator)

        feed_dict = {self.input_placeholder: out_envelopes,self.output_placeholder: out_audios, self.cond_placeholder: out_features, self.mask_placeholder:out_masks, self.is_train: False}
        output = sess.run(self.output, feed_dict=feed_dict)
        # feed_dict = {self.input_placeholder: out_envelopes,self.output_placeholder: out_audios, self.cond_placeholder: np.ones(out_features.shape)*0.1  , self.mask_placeholder:out_masks, self.is_train: False}
        # output_1 = sess.run(self.output, feed_dict=feed_dict)
        # mod_feats = np.ones(out_features.shape)*0.1 
        # mod_feats[:,0] = 0.9
        # feed_dict = {self.input_placeholder: out_envelopes,self.output_placeholder: out_audios, self.cond_placeholder: mod_feats, self.mask_placeholder:out_masks, self.is_train: False}
        # output_2 = sess.run(self.output, feed_dict=feed_dict)

        output = output * out_envelopes

        # output_1 = output_1 * out_envelopes

        # output_2 = output_2 * out_envelopes

        for i in range(config.batch_size):
            print( [str(out_features[i][x])+":"+config.feats_to_use[x] for x in range(len(config.feats_to_use))])

            ax1 = plt.subplot(311)
            ax1.set_title("Output Waveform", fontsize=10)
            plt.plot(np.clip(output[i][:14000], -1.0,1.0))
            # ax2 = plt.subplot(512)
            # ax2.set_title("Output Waveform 0.1", fontsize=10)
            # plt.plot(np.clip(output_1[i][:14000], -1.0,1.0))
            # ax2 = plt.subplot(513)
            # ax2.set_title("Output Waveform 0.9", fontsize=10)
            # plt.plot(np.clip(output_2[i][:14000], -1.0,1.0))
            ax2 = plt.subplot(312)
            ax2.set_title("Ground Truth Waveform", fontsize=10)
            plt.plot(out_audios[i])
            ax3 = plt.subplot(313)
            ax3.set_title("Input Envelope", fontsize=10)
            plt.plot(out_envelopes[i])
            # synth = utils.query_yes_no("Save Plot?")

            # if synth:
            # plt.savefig('./fig_{}'.format(i))

            plt.show()
            # synth = utils.query_yes_no("Synthesize output? ")


            # if synth:

            sf.write('./op_{}.wav'.format(i), np.clip(output[i][:14000], -1.0,1.0), config.fs)
            # sf.write('./op0.1_{}.wav'.format(i), np.clip(output_1[i][:14000], -1.0,1.0), config.fs)
            # sf.write('./op0.2_{}.wav'.format(i), np.clip(output_2[i][:14000], -1.0,1.0), config.fs)
            # synth = utils.query_yes_no("Synthesize  Ground Truth? ")

            # if synth:

            sf.write('./gt_{}.wav'.format(i), out_audios[i], config.fs)





  



    def model(self):
        """
        The main model function, takes and returns tensors.
        Defined in modules.

        """

        with tf.variable_scope('Final_Model') as scope:
            self.output = modules.full_network(self.cond_placeholder,self.input_placeholder,  self.is_train)

        with tf.variable_scope('Discriminator') as scope: 
            self.real_emb, self.real_cont = modules.content(self.output_placeholder* self.input_placeholder, self.is_train)
            scope.reuse_variables()
            self.fake_emb, self.fake_cont  = modules.content(self.output* self.input_placeholder,  self.is_train)
def test():
    # model = DeepSal()
    # # model.test_file('nino_4424.hdf5')
    # model.test_wav_folder('./helena_test_set/', './results/')

    model = PercSynth()

    output = model.get_output(np.random.rand(16000), np.random.rand(7))
    import pdb;pdb.set_trace()

if __name__ == '__main__':
    test()





