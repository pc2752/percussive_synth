from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import tensorflow as tf
import numpy as np
from tensorflow.python import debug as tf_debug
from tensorflow.contrib.rnn import GRUCell
from tensorflow.contrib import rnn
import config


tf.logging.set_verbosity(tf.logging.INFO)



def wavenet_block(inputs, conditioning, is_train, dilation_rate = 2, kernel_size = config.kernel_size, name = "name"):

    pad = (kernel_size - 1) * dilation_rate

    conditioning = tf.layers.batch_normalization(tf.layers.conv1d(conditioning, config.num_filters, 1, dilation_rate = 1, padding = 'valid', name = name+"_cond"), training = is_train)

    con_pad_forward = tf.pad(inputs, [[0,0],[dilation_rate,0],[0,0]],"CONSTANT")

    con_sig_forward = tf.layers.batch_normalization(tf.layers.conv1d(con_pad_forward, config.num_filters, kernel_size, dilation_rate = dilation_rate, padding = 'valid', name = name+"_1"), training = is_train)

    sig = tf.sigmoid(con_sig_forward + conditioning)

    con_tanh_forward = tf.layers.batch_normalization(tf.layers.conv1d(con_pad_forward, config.num_filters, kernel_size, dilation_rate = dilation_rate, padding = 'valid', name = name+"_2"), training = is_train)

    tanh = tf.tanh(con_tanh_forward + conditioning)

    outputs = tf.multiply(sig,tanh)

    residual = outputs + inputs

    skip = tf.layers.conv1d(outputs,config.skip_filters,1, name = name+"_skip")

    residual = tf.layers.conv1d(residual,config.num_filters,1, name = name+"_residual")
    
    return skip, residual

def wave_archi(inputs, conditioning, is_train):


    receptive_field = 2**config.wavenet_layers

    inputs = tf.pad(inputs, [[0,0],[config.first_conv -1 ,0],[0,0]],"CONSTANT")

    residual = tf.layers.batch_normalization(tf.layers.conv1d(inputs, config.num_filters, config.first_conv, name = "first_conv"), training = is_train)

    skips = []

    output = tf.layers.conv1d(residual,config.skip_filters,1, name = "first_skip")

    for i in range(config.wavenet_layers):
        skip, residual = wavenet_block(residual,conditioning, is_train, dilation_rate = config.dilation_rates[i], name = "npss_block_"+str(i+1))
        skips.append(skip)
    for skip in skips:
        output+=skip

    conditioning = tf.layers.batch_normalization(tf.layers.conv1d(conditioning, config.skip_filters, 1, dilation_rate = 1, padding = 'valid', name = "cond"), training = is_train)

    output = output + conditioning

    output = tf.nn.tanh(output)

    output = tf.layers.conv1d(output,config.output_features,1, name = "Output" )

    return output


def encoder_conv_block(inputs, layer_num, is_train, num_filters = config.filters):

    output = tf.layers.batch_normalization(tf.nn.relu(tf.layers.conv2d(inputs, num_filters * 2**int(layer_num/config.augment_filters_every), (config.filter_len,1)
        , strides=(2,1),  padding = 'same', name = "G_"+str(layer_num))), training = is_train)
    # print(output.shape)
    # print(num_filters * 2**int(layer_num/config.augment_filters_every))
    return output

def decoder_conv_block(inputs, layer, layer_num, is_train, num_filters = config.filters):

    deconv = tf.image.resize_images(inputs, size=(layer.shape[1],1), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    deconv = tf.layers.batch_normalization( tf.nn.relu(tf.layers.conv2d(deconv, layer.shape[-1]
        , (config.filter_len,1), strides=(1,1),  padding = 'same', name =  "D_"+str(layer_num))), training = is_train)

    print(deconv.shape)
    print(layer.shape)

    deconv =  tf.concat([deconv, layer], axis = -1)

    return deconv


def encoder_decoder_archi(inputs, is_train):
    """
    Input is assumed to be a 4-D Tensor, with [batch_size, phrase_len, 1, features]
    """

    encoder_layers = []

    encoded = inputs

    encoder_layers.append(encoded)

    for i in range(config.encoder_layers):
        encoded = encoder_conv_block(encoded, i, is_train)
        encoder_layers.append(encoded)
    
    encoder_layers.reverse()

    decoded = encoder_layers[0]

    for i in range(config.encoder_layers):
        decoded = decoder_conv_block(decoded, encoder_layers[i+1], i, is_train)

    return decoded


def full_network(condsi, env, is_train):

    conds = tf.tile(tf.reshape(condsi,[config.batch_size,1,-1]),[1,config.max_phr_len,1])


    inputs = tf.concat([conds, env], axis = -1)

    inputs = tf.reshape(inputs, [config.batch_size, config.max_phr_len , 1, -1])

    # inputs = tf.layers.batch_normalization(tf.layers.dense(inputs, config.filters
    #     , name = "S_in"), training = is_train)

    output = encoder_decoder_archi(inputs, is_train)

    output = tf.layers.batch_normalization(tf.layers.dense(output, config.output_features, name = "Fu_F"), training = is_train)

    output = tf.reshape(output, [config.batch_size, config.max_phr_len, -1])

    return output

def main():    
    vec = tf.placeholder("float", [config.batch_size, config.max_phr_len, 1])
    tec = np.random.rand(config.batch_size, config.max_phr_len,1) #  batch_size, time_steps, features
    
    conds = tf.placeholder("float", [config.batch_size, 7])
    condi = np.random.rand(config.batch_size, 7) 

    is_train = tf.placeholder(tf.bool, name="is_train")
    # seqlen = tf.placeholder("float", [config.batch_size, 256])

    with tf.variable_scope('full_Model') as scope:
        out_put = full_network( conds,vec, is_train)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    op= sess.run(out_put, feed_dict={conds:condi, vec: tec, is_train: True})
    # writer = tf.summary.FileWriter('.')
    # writer.add_graph(tf.get_default_graph())
    # writer.add_summary(summary, global_step=1)
    import pdb;pdb.set_trace()


if __name__ == '__main__':
  main()