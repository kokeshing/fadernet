import tensorflow as tf
import numpy as np
import optparse
import os
import shutil
import time
import random
import sys
import pickle
import glob

IMG_SIZE = 256
IMG_CHANNEL = 3

class Fadernet(object):

    def __init__(self, batch_size):
        self.batch_size = batch_size

        # input
        with tf.variable_scope('inputs'):
            self.x = tf.placeholder(tf.float32, [self.batch_size, IMG_SIZE, IMG_SIZE, 3])
            self.input_attr = tf.placeholder(tf.float32, [self.batch_size, 1])


        # Encoder
        with tf.variable_scope('encoder'):
            self.co16    = tf.layers.conv2d(inputs=self.x, filters=16, kernel_size=(4, 4), strides=(2, 2), padding="same", activation=tf.nn.leaky_relu(features=float32, alpha=0.2), name="co16")
            self.co32    = tf.layers.conv2d(inputs=self.co16, filters=32, kernel_size=(4, 4), strides=(2, 2), padding="same", activation=tf.nn.leaky_relu(features=float32, alpha=0.2), name="co32")
            self.co64    = tf.layers.conv2d(inputs=self.co32, filters=64, kernel_size=(4, 4), strides=(2, 2), padding="same", activation=tf.nn.leaky_relu(features=float32, alpha=0.2), name="co64")
            self.co128   = tf.layers.conv2d(inputs=self.co64, filters=128, kernel_size=(4, 4), strides=(2, 2), padding="same", activation=tf.nn.leaky_relu(features=float32, alpha=0.2),  name="co128")
            self.co256   = tf.layers.conv2d(inputs=self.co128, filters=256, kernel_size=(4, 4), strides=(2, 2), padding="same", activation=tf.nn.leaky_relu(features=float32, alpha=0.2), name="co256")
            self.co512   = tf.layers.conv2d(inputs=self.co256, filters=512, kernel_size=(4, 4), strides=(2, 2), padding="same", activation=tf.nn.leaky_relu(features=float32, alpha=0.2), name="co512")
            self.enc_out = tf.layers.conv2d(inputs=self.co512, filters=512, kernel_size=(4, 4), strides=(2, 2), padding="same", activation=tf.nn.leaky_relu(features=float32, alpha=0.2), name="enc_out")


        # decoder
        with tf.variable_scope('decoder'):
            self.y = tf.ones([self.batch_size, 2, 2, 1]) * self.input_attr
            self.in_dec512_0  = tf.concat([self.enc_out, self.y], axis=3)
            self.out_dec512_0 = tf.layers.conv2d_transpose(inputs=self.in_dec512_0, filters=512, kernel_size=(4, 4), strides=(2, 2), padding="same", activation=tf.nn.leaky_relu(features=float32, alpha=0.2), name="dc512_0")

            self.y4 = tf.ones([self.batch_size, 4, 4, 1]) * self.input_attr
            self.in_dec512_1 =  tf.concat([self.out_dec512_0, self.y4], axis=3)
            self.out_dec512_1 = tf.layers.conv2d_transpose(inputs=self.in_dec512_1, filters=512, kernel_size=(4, 4), strides=(2, 2), padding="same", activation=tf.nn.leaky_relu(features=float32, alpha=0.2), name="dc512_1")

            self.y8 = tf.ones([self.batch_size, 8, 8, 1]) * self.input_attr
            self.in_dec256  =  tf.concat([self.out_dec512_1, self.y8], axis=3)
            self.out_dec256 = tf.layers.conv2d_transpose(inputs=self.in_dec256, filters=256, kernel_size=(4, 4), strides=(2, 2), padding="same", activation=tf.nn.leaky_relu(features=float32, alpha=0.2), name="dc256")

            self.y16 = tf.ones([self.batch_size, 16, 16, 1]) * self.input_attr
            self.in_dec128  =  tf.concat([self.out_dec256, self.y16], axis=3)
            self.out_dec128 = tf.layers.conv2d_transpose(inputs=self.in_dec128, filters=128, kernel_size=(4, 4), strides=(2, 2), padding="same", activation=tf.nn.leaky_relu(features=float32, alpha=0.2), name="dc128")

            self.y32 = tf.ones([self.batch_size, 32, 32, 1]) * self.input_attr
            self.in_dec64  =  tf.concat([self.out_dec128, self.y32], axis=3)
            self.out_dec64 = tf.layers.conv2d_transpose(inputs=self.in_dec64, filters=64, kernel_size=(4, 4), strides=(2, 2), padding="same", activation=tf.nn.leaky_relu(features=float32, alpha=0.2), name="dc64")

            self.y64 = tf.ones([self.batch_size, 64, 64, 1]) * self.input_attr
            self.in_dec32  =  tf.concat([self.out_dec64, self.y64], axis=3)
            self.out_dec32 = tf.layers.conv2d_transpose(inputs=self.in_dec32, filters=32, kernel_size=(4, 4), strides=(2, 2), padding="same", activation=tf.nn.leaky_relu(features=float32, alpha=0.2), name="dc32")

            self.y128 = tf.ones([self.batch_size, 128, 128, 1]) * self.input_attr
            self.in_dec16  =  tf.concat([self.out_dec32, self.y128], axis=3)
            self.out_dec16 = tf.layers.conv2d_transpose(inputs=self.in_dec16, filters=16, kernel_size=(4, 4), strides=(2, 2), padding="same", activation=tf.nn.leaky_relu(features=float32, alpha=0.2), name="dc16")


        # discriminator
        with tf.variable_scope('discriminator'):
            self.disc_conv = tf.layers.conv2d(inputs=self.enc_out, filters=512, activation=tf.nn.leaky_relu(features=float32, alpha=0.2), name="disc_co")
            self.disc_conv = tf.layers.dropout(inputs=self.disc_conv, rate=0.3)
            self.disc_flat = tf.reshape(self.disc_conv, [self.batch_size, 512])
            self.disc_fc1  = tf.layers.dense(inputs=self.disc_flat, units=512, name="disc_fc1")
            self.disc_fc2  = tf.layers.dense(inputs=self.disc_fc1, units=1, name="disc_fc2")
            self.disc_out  = tf.nn.sigmoid(disc_fc2)


    def autoEncoder_loss(self, input_img, output_img):
        return  tf.reduce_sum(tf.squared_difference(input_img, output_img), [0, 1, 2])


    def discriminator_loss(self, input_y, disc_out):
        return tf.losses.log_loss(labels=input_y, predictions=disc_out)


    def adversarial_loss(self, input_img, output_img, input_y, disc_out, lambda_e):
        loss_ae = tf.losses.mean_squared_error(labels=input_img, predictions=output_img)
        loss_ad = lambda_e * tf.losses.log_loss(labels=input_y, predictions=(1.0-disc_out))

        return loss_ae - loss_ad


    def normalize_input(self, imgs):
        return imgs/255.0-1.0


    def load_batch(self, batch_num, mode="train"):

        if(mode == "train"):
            temp = []
            for i in range(self.batch_size):
                temp.append(self.normalize_input(np.array(Image.open(self.imagePath[i + batch_sz*(batch_num)]),'f')))

            return temp

        elif (mode == "test"):
            temp = []
            for i in range(self.batch_size):
                temp.append(self.normalize_input(np.array(Image.open(self.imagePath[i + batch_sz*(batch_num)]),'f')))

            return temp

    def load_dataset(self, mode='train', dataset_dir, num_images):

        if(mode == "train"):

            self.train_attr = []

            imgs = glob.glob(dataset_dir + '/*.jpg')

            dictn = []

            count = 0
            with open(self.dataset_dir + "/list_attr_celeba.txt") as f:
                for lines in f:
                    temp = lines
                    temp = temp.split()
                    dictn.append(temp[1])


            for i in range(num_images):
                self.train_attr = dictn
            # print(self.train_attr[0:10])

        elif (mode == "test"):

            self.test_attr = []

            imgs = glob.glob(dataset_dir + '/*.jpg')

            dictn = []

            count = 0
            with open(dataset_dir + "/list_attr_test.txt") as f:
                for lines in f:
                    temp = lines
                    temp = temp.split()
                    dictn.append(temp[1])

            for i in range(num_images):
                self.test_attr = dictn


    def train(self, dataset_dir="./train_data", result_dir="./result", load_ckp=False, epoch_num=64):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        ckp_dir = os.path.join(result_dir + "/checkpoints")
        tensorboard_dir = os.path.join(result_dir + "/tensorboard")

        if not os.path.exists(dataset_dir):
            sys.exit()

        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        if not os.path.exists(ckp_dir):
            os.makedirs(ckp_dir)

        if not os.path.exists(tensorboard_dir):
            os.makedirs(tensorboard_dir)

        num_train_images = len(glob.glob(dataset_dir + "/*.jpg"))

        with tf.Session() as sess:

            sess.run(init)
            writer = tf.summary.FileWriter(tensorboard_dir)
            writer.add_graph(sess.graph)

            if load_ckp:
                chkpt_fname = tf.train.latest_checkpoint(ckp_dir)
                saver.restore(sess,chkpt_fname)

            per_epoch_steps = int(num_train_images/self.batch_size)

            t = time.time()

            for epoch in range(0, epoch_num):
                for itr in range(0, per_epoch_steps):

                    temp_lmd = 0.0001*(epoch * per_epoch_steps + itr) / (per_epoch_steps * epoch_num)

                    imgs = self.load_batch(itr)
                    attrs = self.train_attr[itr*self.batch_size:(itr+1)*(self.batch_size)]



                    writer.add_summary(img_loss_str,epoch*per_epoch_steps + itr)
                    writer.add_summary(enc_loss_str,epoch*per_epoch_steps + itr)
                    writer.add_summary(disc_loss_str,epoch*per_epoch_steps + itr)

                    if(itr == per_epoch_steps - 1):
                        print("epoch:", epoch)
                        print(time.time() - t)
                        print(temp_tot_loss, temp_img_loss, temp_enc_loss, temp_disc_loss)

                saver.save(sess,os.path.join(self.check_dir,"Fader"),global_step=epoch)