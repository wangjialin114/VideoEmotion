# coding: utf-8
"""
Created on Sat May 20 17:07:44 2017
used for renaming the bilibili dataset
@author: WangJalin
## This script is used to 
######################### extract the CNN feature(such as fc7, and other low level feature)
######################### save the CNN feature to the hdf file
"""

import tensorflow as tf
import tqdm
import numpy as np
import h5py
import matplotlib.image as image
from time import time

class vgg16(object):
    """VGG16 CNN
    """
    def __init__(self, batch_size=90):
        """init method
        
        Args:
            batch_size: the number of the examples to extracte feature once
        Returns:
            None
        """
        self.batch_size = batch_size
        # load the pretrained net weight
        self.vgg_pre = np.load("vgg16_weights.npz")
        # rebuild the net
        self.build_vgg16()

    def conv_layer(self, input_op, name):
        """construct the conv layer in the VGG16 net
        
        Args:
            input_op: the input from that last layer
            name: the variable scope
        Returns:
            relu: the layer output
        """
        
        with tf.variable_scope(name):
            conv_w = tf.constant(self.vgg_pre[name+"_W"])
            conv_bias = tf.constant(self.vgg_pre[name+"_b"])
            conv_out = tf.nn.bias_add(tf.nn.conv2d(input=input_op, filter=conv_w, strides=[1, 1, 1, 1], padding="SAME"), conv_bias)

            relu = tf.nn.relu(conv_out)
            return relu

    def fc_layer(self, input_op, name):
        """construct the fully connected layer in the VGG16 net
        
        Args:
            input_op: the input from that last layer
            name: the variable scope
        Returns:
            relu: the layer output    
        """
        
        input_op = tf.reshape(input_op, shape=[self.batch_size, -1])
        input_shape= input_op.get_shape().as_list()
        with tf.variable_scope(name):
            fc_w = tf.constant(self.vgg_pre[name+"_W"])
            fc_b = tf.constant(self.vgg_pre[name+"_b"])
            fc_out = tf.nn.bias_add(tf.matmul(input_op, fc_w), fc_b)

        return fc_out

    def build_vgg16(self):
        """build the VGG16 net"""
        
        self.x = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, 224, 224, 3])
        # 1 conv layer + max pool
        conv1_1 = self.conv_layer(self.x, "conv1_1")
        conv1_2 = self.conv_layer(conv1_1, "conv1_2")
        max_pool1 = tf.nn.max_pool(value=conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool1")
        # 2 conv layer + max pool
        conv2_1 = self.conv_layer(max_pool1, "conv2_1")
        conv2_2 = self.conv_layer(conv2_1, "conv2_2")
        max_pool2 = tf.nn.max_pool(value=conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool2")
        # 3 conv layer + max pool
        conv3_1 = self.conv_layer(max_pool2, "conv3_1")
        conv3_2 = self.conv_layer(conv3_1, "conv3_2")
        conv3_3 = self.conv_layer(conv3_2, "conv3_3")
        max_pool3 = tf.nn.max_pool(value=conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool3")
        # 4 conv layer + max pool
        conv4_1 = self.conv_layer(max_pool3, "conv4_1")
        conv4_2 = self.conv_layer(conv4_1, "conv4_2")
        conv4_3 = self.conv_layer(conv4_2, "conv4_3")
        max_pool4 = tf.nn.max_pool(value=conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool4")
        # 5 conv layer + max pool
        conv5_1 = self.conv_layer(max_pool4, "conv5_1")
        conv5_2 = self.conv_layer(conv5_1, "conv5_2")
        conv5_3 = self.conv_layer(conv5_2, "conv5_3")
        max_pool5 = tf.nn.max_pool(value=conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool5")
        # 6 fc layer
        fc_6 = self.fc_layer(input_op=max_pool5, name="fc6")
        # 7 fc layer
        self.fc_7 = self.fc_layer(input_op=fc_6, name="fc7")
        # 8 fc layer
        fc_8 = self.fc_layer(input_op=self.fc_7, name="fc8")
        # softmax layer
        self.prob = tf.nn.softmax(fc_8, name="prob")

        print("vgg 16 net build finished")

        self.sess = tf.Session()

    def generate_batch(self, x, i):
        """ generate the batch for the input placeholder
        
        Args:
            x: the hdf file that includes the image data
            i: the iteration number
            frame_num: how
        Returns:
            x_batch: the input for the VGG16 net
        """
        
        frame_num = x.shape[1]
        video_num = x.shape[0]
        x_batch = np.zeros([self.batch_size, 224, 224, 3])
        if (i+1)*self.batch_size//frame_num  > x.shape[0]:  # may exceed the size of x
            x_batch[0:(video_num-i*self.batch_size//frame_num)*frame_num] = np.reshape(x[i*self.batch_size//frame_num:], [-1, 224, 224, 3])
        else:
            x_batch = np.reshape(x[i*self.batch_size//frame_num:(i+1)*self.batch_size//frame_num], [-1, 224, 224, 3])
        return x_batch

    def extract_fc7(self, x, fc7_size=4096):
        """evaluating the fc7 feature
        
        Args:
            x: the input hdf file
        """
        fc7_f = h5py.File("frame_fc7.h5", "w")
        video_num = x.shape[0]
        frame_num = x.shape[1]
        fc7_h5 = fc7_f.create_dataset(name="fc_7", shape=(video_num, frame_num, fc7_size), chunks=True)
        ## should be exactly divisible
        batch_num = int(np.ceil(x.shape[0]/(self.batch_size/frame_num)))
        for i in tqdm.tqdm(range(batch_num)):
            x_batch =  self.generate_batch(x, i)
            fc_7 = self.sess.run([self.fc_7], feed_dict={self.x:x_batch})
            if i == batch_num - 1:
                fc7_h5[i*self.batch_size//frame_num:,:,:] = np.reshape(fc_7, [self.batch_size//frame_num, frame_num, fc7_size])[0:(video_num-i*self.batch_size//frame_num)]
            else:
                fc7_h5[i*self.batch_size//frame_num:(i+1)*self.batch_size//frame_num,:,:] = np.reshape(fc_7, [self.batch_size//frame_num, frame_num, fc7_size])
        print("extract fc7 feature finished!")
        fc7_f.close()
        print("save the fc7 feature to the hdf")

if __name__ == "__main__":
    t_start = time()
    vgg = vgg16()
    f_h5 = h5py.File("frame.h5", "r")
    h5_x = f_h5["frames"]
    #h5_y = f_h5.require_dataset(name="video_label", shape=(video_num*crop_num,1)) 
    #h5_x = f_h5.require_dataset(name="video_frame", shape=(video_num, frame_num, 224*224*3),dtype='float32', chunks=True)
    vgg.extract_fc7(h5_x)
    f_h5.close()
    t_end = time()
    t_cost = t_end - t_start
    print("cost %f seconds" % (t_cost))
