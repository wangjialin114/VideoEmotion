# coding: utf-8
"""
Created on Sat May 20 17:07:44 2017
used for renaming the bilibili dataset
@author: WangJalin
### Use CNN-LSTM Model
For the deitals please read the documents
"""

import tensorflow as tf
import numpy as np
import h5py
import matplotlib.image as image
import tqdm
from time import time

class video_lstm(object):
    """CNN+LSTM to classify the video
    
    Since the CNN part are fixed, we extract the image CNN feature and save them to the file to speed up the training.
    This part only includes the LSTM part of the model.
    """
    
    def __init__(self, learning_rate=1e-3, batch_size=32, frame_num=10, hidden_size=100, fc_size=4096, log="log", 
                 max_epochs=10, lr_decay_rate=0.96, output_size=11):
        """init the parameters of the model, build the network
        
        Args:
            learning_rate: learning rate when uodating the parameters
            batch_size: number of the trainging examples in one batch
            frame_num: image number per one video
            hidden_size: the dimmension of the hidden state in the LSTM
            fc_size: the dimmension of the CNN feature
            log: the directory name that stores the model related information
            max_epochs: the max number times that one training example used to train
            lr_decay_rate: the decay rate of learning rate when the traning step grows
            output_size: the number of the target category
        """
        
        self.lr = learning_rate
        self.batch_size = batch_size
        self.frame_num = frame_num
        self.hidden_size = hidden_size
        self.max_epochs = max_epochs
        self.fc_size = fc_size
        self.output_size = output_size
        self.lr_decay_rate = lr_decay_rate
        self.frame_per_cluster = 3  # number of extracted frames per cluster
        self.log = log
        # build the net
        self.build_lstm()

    def compute_lstm_vector(self, fc_7):
        """compute the lstm network last hidden state
        
        Args:
            fc_7: the CNN feature, also the input of the LSTM part
        Returns:
            final_state: the final hidden state of the LSTM part
        """
        
        frame_list = [tf.squeeze(input) for input in tf.split(fc_7, self.frame_num, axis=1)]
        # here utilize the layernorm LSTM cell to speed the convergence of the training
        cell = tf.contrib.rnn.LayerNormBasicLSTMCell(self.hidden_size, forget_bias=0.1)
        outputs, output_state = tf.contrib.rnn.static_rnn(cell, inputs=frame_list ,dtype=tf.float32)
        # take the last state of all the states
        final_state = output_state[-1]

        return final_state

    def build_lstm(self):
        """build the lstm network
        """
        
        # VGG fc_7 feature as x input
        self.x = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.frame_num, self.fc_size])
        self.y = tf.placeholder(tf.float32, shape=[self.batch_size])
        # lstm
        with tf.variable_scope("lstm") as scope:
            final_state = self.compute_lstm_vector(self.x)
            # fully conneccted layer
            lstm_fc_w = tf.get_variable("fc_w", shape=[self.hidden_size, self.fc_size], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            lstm_fc_b = tf.get_variable("fc_b", shape=[self.fc_size], initializer=tf.constant_initializer(0.0))
            lstm_fc_out = tf.matmul(final_state, lstm_fc_w) + lstm_fc_b
            # logistic layer
            lstm_soft_w = tf.get_variable("soft_w",shape=[self.fc_size, self.output_size], initializer=tf.contrib.layers.xavier_initializer())
            lstm_soft_b = tf.get_variable("soft_b", shape=[self.output_size], initializer=tf.constant_initializer(0.0))
            # Projection layer
            y_ = tf.nn.softmax(tf.matmul(lstm_fc_out, lstm_soft_w)+lstm_soft_b)
        # the label is the sequencial number of the columns whose value is biggest along the rows
        # the total number of the columns denotes the feature dimmension
        # the total number of the rows denotes the number of the examples
        label = tf.arg_max(dimension=1, input=y_)
        # compute the accuracy
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(self.y,tf.int32),tf.cast(label, tf.int32)), tf.float32))
        # compute the loss
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(self.y,tf.int32), logits=y_))
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.learning_rate = tf.train.exponential_decay(self.lr,self.global_step,500,
                                                        self.lr_decay_rate,staircase=True)
        # add the traning operation
        self.train_op = tf.train.AdagradOptimizer(self.lr).minimize(self.loss)
        # to save the model information every fixed number training epochs
        self.latest_checkpoint = tf.train.latest_checkpoint(self.log)
        # add the saver
        self.saver = tf.train.Saver()
        
        
    def generate_next_batch(self, x, y, iter_num):
        """traverse the dataset
        
        Args:
            x: the hdf file, the amount of the data that cannot be stored in the memory[video_num, frame_num, fc_size]
            y: the hdf file
            iter_num: denotes the data position
        Returns:
            x_batch: a array whose shape should be [batch_size, frame_num, fc_size]
            y_batch: a array whose shape should be [batch_size, 1]
        """
        
        k = np.array(range(iter_num*self.batch_size,(iter_num+1)*self.batch_size))
        x_batch = np.zeros([self.batch_size, self.frame_num, self.fc_size])
        for i in range(k.shape[0]):
            video_seq = k[i]
            for j in range(self.frame_num):
                selected_frame_seq = 0  ## mainly for overfitting, so proof that our model is right
                x_batch[i, j, :] = x[video_seq, j*self.frame_per_cluster+selected_frame_seq, :]
            
        y_batch = y[k]

        return x_batch, y_batch
    def generate_batch(self, x, y):
        """prepare the data batch that will be fed into the network
        
        cause the hdf file can only be taken in a strict ascending order. 
        thus we take a continual examples for convenience
        
        Args:
            x: the hdf file, the amount of the data that cannot be stored in the memory[video_num, frame_num, fc_size]
            y: the hdf file
        Returns:
            x_batch: a array whose shape should be [batch_size, frame_num, fc_size]
            y_batch: a array whose shape should be [batch_size, 1]
        """
        
        k = np.random.randint(low=0, high=y.shape[0]-self.batch_size, size=[self.batch_size])
        x_batch = np.zeros([self.batch_size, self.frame_num, self.fc_size])
        for i in range(k.shape[0]):
            video_seq = k[i]
            for j in range(self.frame_num):
                selected_frame_seq = np.random.randint(self.frame_per_cluster)
                x_batch[i, j, :] = x[video_seq, j*self.frame_per_cluster+selected_frame_seq, :]
                
        y_batch = y[k]

        return x_batch, y_batch
    
    def read_class_file(self, video_info_filename, success_filename):
        """read the class of the video
        
        Args:
            video_info_filename: the path of the file that record the video information
            success_filename: the path of the file that record the video_id extracted successfully
        Returns:
            y: the array of the successfully extracted video emotion class
        """
        
        # 11 cols, the ist is the video_id, the last 2 cols are the category and the sub_category
        f = open(video_info_filename, encoding="utf-8")
        info = f.readlines()
        video_class = {}
        for item in info:
            video_info = item.split()
            video_class[video_info[0]] = video_info[-2]  # store the category
        f.close()
        
        f = open(success_filename, "r")
        video_id = f.readlines()
        y = np.array([video_class[x.strip()] for x in video_id])
        return y
        
    
    def eval_network(self, x_test, y_test):
        """eval the network using the test dataset
        
        Args:
            x_test: the hdf file of the testing dataset
            y_test: the hdf file of the testing dataset
        Returns:
            None
        """
        
        mean_loss = []  # store the loss for every iteration
        mean_accuracy = []  # store the accuracy for every iteration

        batch_iter_num = y_test.shape[0]//self.batch_size
        for i in range(batch_iter_num):
            x_batch, y_batch = self.generate_batch(x_test, y_test)
            loss, accuracy = self.sess.run([self.loss, self.accuracy], feed_dict={self.x: x_batch, self.y:y_batch})
            mean_loss.append(loss)
            mean_accuracy.append(accuracy)
        print("eval -- loss: %f, accuracy: %f" %(np.mean(mean_loss), np.mean(mean_accuracy)))

    def train_network(self, x_train, y_train):
        """training the network using the training dataset
        
        Args:
            x_train: the hdf file of the training dataset
            y_train: the hdf file of the training dataset
            x_test: the hdf file of the testing dataset
            y_test: the hdf file of the testing dataset
        Returns:
            None
        """
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        self.sess = tf.Session()
        self.sess.run(init_op)
        # check if there exists checkpoint, if true, load it
        ckpt = tf.train.get_checkpoint_state(self.log)
        print(ckpt)
        if ckpt and ckpt.model_checkpoint_path:
            print("Load the checkpoint: %s" % (ckpt.model_checkpoint_path))
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        # train
        #for epoch in tqdm.tqdm(range(self.max_epochs)):
        for epoch in range(self.max_epochs):
            mean_loss = []
            mean_accuracy = []
            batch_iter_num = y_train[0:100].shape[0]//self.batch_size
            for i in range(batch_iter_num):
                x_batch, y_batch = self.generate_next_batch(x_train, y_train, i)
                _, loss, accuracy, step = self.sess.run([self.train_op, self.loss, self.accuracy, self.global_step], feed_dict={self.x: x_batch, self.y:y_batch})
                mean_loss.append(loss)
                mean_accuracy.append(accuracy)
                #if i % 100 == 0 and i > 0:
                #    self.saver.save(self.sess, self.log+"/model.ckpt",global_step=i+1)
                    #print("save the model")
                    #print("step %d / %d: loss : %f, accuracy : %f" %(i, batch_iter_num, np.mean(mean_loss), np.mean(mean_accuracy)))
                #    mean_loss = []
                #    mean_accuracy = []
            ## evalating the network
            #self.eval_network(x_test, y_test)
            if epoch % 5 == 0:
                self.saver.save(self.sess, self.log+"/model.ckpt",global_step=step)
            print("epoch %d --- loss : %f, accuracy : %f" %(epoch, np.mean(mean_loss), np.mean(mean_accuracy)))
if __name__ == "__main__":
    fc7_f = h5py.File("frame_fc7.h5", "r")
    fc7_h5 = fc7_f["fc_7"]
    clf = video_lstm(learning_rate=1e-3, max_epochs=500)
    labels = clf.read_class_file("video_1_final.txt", "success_video_id.txt") 
    t_start = time()
    clf.train_network(fc7_h5, labels)
    t_end = time()
    t_cost = t_end - t_start
    print("cost %f seconds" % (t_cost))
    fc7_f.close()
