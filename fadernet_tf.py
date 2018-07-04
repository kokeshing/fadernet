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
from PIL import Image

IMG_SIZE = 256
IMG_CHANNEL = 3

class Fadernet(object):

    def __init__(self, batch_size):
        self.batch_size = batch_size

        # input
        with tf.variable_scope('inputs'):
            self.x = tf.placeholder(tf.float32, [self.batch_size, IMG_SIZE, IMG_SIZE, 3])
            self.input_attr = tf.placeholder(tf.int32, [self.batch_size, 1])
            self.input_attr_one_hot = tf.one_hot(indices=tf.reshape(self.input_attr, [self.batch_size]), depth=2)
            self.itr_num = tf.placeholder(tf.float32,[1])


        # Encoder
        with tf.variable_scope('encoder'):
            self.co16    = tf.layers.conv2d(inputs=self.x, filters=16, kernel_size=(4, 4), strides=(2, 2), padding="same", activation=tf.nn.leaky_relu, name="co16")
            self.co32    = tf.layers.conv2d(inputs=self.co16, filters=32, kernel_size=(4, 4), strides=(2, 2), padding="same", activation=tf.nn.leaky_relu, name="co32")
            self.co64    = tf.layers.conv2d(inputs=self.co32, filters=64, kernel_size=(4, 4), strides=(2, 2), padding="same", activation=tf.nn.leaky_relu, name="co64")
            self.co128   = tf.layers.conv2d(inputs=self.co64, filters=128, kernel_size=(4, 4), strides=(2, 2), padding="same", activation=tf.nn.leaky_relu,  name="co128")
            self.co256   = tf.layers.conv2d(inputs=self.co128, filters=256, kernel_size=(4, 4), strides=(2, 2), padding="same", activation=tf.nn.leaky_relu, name="co256")
            self.co512   = tf.layers.conv2d(inputs=self.co256, filters=512, kernel_size=(4, 4), strides=(2, 2), padding="same", activation=tf.nn.leaky_relu, name="co512")
            self.enc_out = tf.layers.conv2d(inputs=self.co512, filters=512, kernel_size=(4, 4), strides=(2, 2), padding="same", activation=tf.nn.leaky_relu, name="enc_out")


        # decoder
        with tf.variable_scope('decoder'):
            self.y2 = tf.transpose(tf.stack([self.input_attr_one_hot] * 4), [1, 0, 2])
            self.in_dec512_0  = tf.concat([self.enc_out, tf.reshape(self.y2, [-1, 2, 2, 2])], axis=3)
            self.out_dec512_0 = tf.layers.conv2d_transpose(inputs=self.in_dec512_0, filters=512, kernel_size=(4, 4), strides=(2, 2), padding="same", activation=tf.nn.leaky_relu, name="dc512_0")

            self.y4 = tf.concat([self.y2] * 4, axis=1)
            self.in_dec512_1 =  tf.concat([self.out_dec512_0, tf.reshape(self.y4, [-1, 4, 4, 2])], axis=3)
            self.out_dec512_1 = tf.layers.conv2d_transpose(inputs=self.in_dec512_1, filters=512, kernel_size=(4, 4), strides=(2, 2), padding="same", activation=tf.nn.leaky_relu, name="dc512_1")

            self.y8 = tf.concat([self.y4] * 4, axis=1)
            self.in_dec256  =  tf.concat([self.out_dec512_1, tf.reshape(self.y8, [-1, 8, 8, 2])], axis=3)
            self.out_dec256 = tf.layers.conv2d_transpose(inputs=self.in_dec256, filters=256, kernel_size=(4, 4), strides=(2, 2), padding="same", activation=tf.nn.leaky_relu, name="dc256")

            self.y16 = tf.concat([self.y8] * 4, axis=1)
            self.in_dec128  =  tf.concat([self.out_dec256, tf.reshape(self.y16, [-1, 16, 16, 2])], axis=3)
            self.out_dec128 = tf.layers.conv2d_transpose(inputs=self.in_dec128, filters=128, kernel_size=(4, 4), strides=(2, 2), padding="same", activation=tf.nn.leaky_relu, name="dc128")

            self.y32 = tf.concat([self.y16] * 4, axis=1)
            self.in_dec64  =  tf.concat([self.out_dec128, tf.reshape(self.y32, [-1, 32, 32, 2])], axis=3)
            self.out_dec64 = tf.layers.conv2d_transpose(inputs=self.in_dec64, filters=64, kernel_size=(4, 4), strides=(2, 2), padding="same", activation=tf.nn.leaky_relu, name="dc64")

            self.y64 = tf.concat([self.y32] * 4, axis=1)
            self.in_dec32  =  tf.concat([self.out_dec64, tf.reshape(self.y64, [-1, 64, 64, 2])], axis=3)
            self.out_dec32 = tf.layers.conv2d_transpose(inputs=self.in_dec32, filters=32, kernel_size=(4, 4), strides=(2, 2), padding="same", activation=tf.nn.leaky_relu, name="dc32")

            self.y128 = tf.concat([self.y64] * 4 , axis=1)
            self.in_dec16  =  tf.concat([self.out_dec32, tf.reshape(self.y128, [-1, 128, 128, 2])], axis=3)
            self.out_dec16 = tf.layers.conv2d_transpose(inputs=self.in_dec16, filters=16, kernel_size=(4, 4), strides=(2, 2), padding="same", activation=tf.nn.leaky_relu, name="dc16")


        # discriminator
        with tf.variable_scope('discriminator'):
            self.disc_conv = tf.layers.conv2d(inputs=self.enc_out, filters=512, kernel_size=(4, 4), strides=(2, 2), padding="same", activation=tf.nn.leaky_relu, name="disc_co")
            self.disc_conv = tf.layers.dropout(inputs=self.disc_conv, rate=0.3)
            self.disc_flat = tf.reshape(self.disc_conv, [self.batch_size, 512])
            self.disc_fc1  = tf.layers.dense(inputs=self.disc_flat, units=512, name="disc_fc1")
            self.disc_fc2  = tf.layers.dense(inputs=self.disc_fc1, units=1, name="disc_fc2")
            self.disc_out  = tf.nn.sigmoid(self.disc_fc2)

        self.model_var = tf.trainable_variables()


    def autoEncoder_loss(self, input_img, output_img):
        return  tf.reduce_sum(tf.squared_difference(input_img, output_img[:, :, :, :3]))


    def discriminator_loss(self, input_y, disc_out):
        return tf.losses.log_loss(labels=input_y, predictions=disc_out)


    def adversarial_loss(self, input_img, output_img, input_y, disc_out, itr):

        lambda_e = 0.0001 * itr / 500000.0

        loss_ae = tf.reduce_sum(tf.squared_difference(input_img, output_img[:, :, :, :3]))
        loss_ad = lambda_e * tf.losses.log_loss(labels=input_y, predictions=(1.0-disc_out))

        return loss_ae - loss_ad


    def normalize_input(self, imgs):
        return imgs/255.0-1.0


    def load_batch(self, batch_num, img_path, mode="train", epoch_num=None):

        if(mode == "train"):
            temp = []
            for i in range(self.batch_size):
                temp.append(self.normalize_input(np.array(Image.open(img_path[i + self.batch_size*(batch_num)]),'f')))

            return temp

        elif (mode == "test"):
            temp = []
            for i in range(self.batch_size):
                temp.append(self.normalize_input(np.array(Image.open(img_path[epoch_num]),'f')))

            return temp

    def load_dataset(self, dataset_dir, num_images, mode='train'):

        if(mode == "train"):

            self.train_attr = []

            imgs = glob.glob(dataset_dir + '/*.jpg')

            dictn = []

            count = 0
            with open(dataset_dir + "/list_attr_celeba.txt") as f:
                for lines in f:
                    temp = lines
                    temp = temp.split()
                    dictn.append([temp[1]])


            for i in range(num_images):
                self.train_attr = dictn

            return imgs

        elif (mode == "test"):
            imgs = glob.glob(dataset_dir + '/*.jpg')


            return imgs


    def train(self, dataset_dir="./train_data/celebA", result_dir="./result", load_ckp=False, epoch_num=64):
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
        img_path = self.load_dataset(dataset_dir, num_train_images)

        enc_var = [var for var in self.model_var if 'encoder' in var.name]
        dec_var = [var for var in self.model_var if 'decoder' in var.name]
        disc_var = [var for var in self.model_var if 'discriminator' in var.name]

        ae_loss_op = self.autoEncoder_loss(self.x, self.out_dec16)
        dc_loss_op = self.discriminator_loss(self.input_attr, self.disc_out)
        ad_loss_op = self.adversarial_loss(self.x, self.out_dec16, self.input_attr, self.disc_out, self.itr_num)

        ae_train_op = tf.train.AdamOptimizer(0.002, beta1=0.5).minimize(ae_loss_op, var_list=dec_var)
        ds_train_op = tf.train.AdamOptimizer(0.002, beta1=0.5).minimize(dc_loss_op, var_list=disc_var)
        ad_train_op  = tf.train.AdamOptimizer(0.002, beta1=0.5).minimize(ad_loss_op, var_list=enc_var)

        '''
        ae_loss_summ = tf.summary.scalar("ae_loss", ae_loss_op)
        dc_loss_summ = tf.summary.scalar("dc_loss", dc_loss_op)
        ad_loss_summ = tf.summary.scalar("ad_loss", ad_loss_op)
        '''

        init = tf.initialize_all_variables()
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init)

            '''
            summary_op = tf.summary.merge_all()
            '''
            writer = tf.summary.FileWriter(tensorboard_dir)
            writer.add_graph(sess.graph)

            if load_ckp:
                chkpt_fname = tf.train.latest_checkpoint(ckp_dir)
                saver.restore(sess,chkpt_fname)

            per_epoch_steps = int(num_train_images/self.batch_size)

            t = time.time()

            for epoch in range(0, epoch_num):
                for itr in range(0, per_epoch_steps):

                    imgs = self.load_batch(itr, img_path)
                    attrs = self.train_attr[itr*self.batch_size:(itr+1)*(self.batch_size)]

                    itr_num = epoch * per_epoch_steps + itr

                    # omit summary_op and result
                    _, ae_loss, _, ds_loss, _, ad_loss = sess.run([ae_train_op, ae_loss_op, ds_train_op, ds_train_op, ad_train_op, ad_loss_op],
                                                                 feed_dict={self.x:imgs, self.input_attr:attrs, self.itr_num:[itr_num]})

                    writer.add_summary(ae_loss, self.itr_num)
                    writer.add_summary(ds_loss, self.itr_num)
                    writer.add_summary(ad_loss, self.itr_num)

                    if(itr == per_epoch_steps - 1):
                        print("epoch:", epoch)
                        print(time.time() - t)
                        print(f"AutoEncoder_loss           : {ae_loss}")
                        print(f"Discriminator_loss         : {ds_loss}")
                        print(f"Adversarial_operation_loss : {ad_loss}")

                saver.save(sess,os.path.join(self.check_dir,"Fader"),global_step=epoch)

    def test(self, dataset_dir="./test_data/celebA",result_dir="./result", ckp_dir="./result/checkpoints"):
        img_path = self.load_dataset(dataset_dir=dataset_dir, mode="test")

        array = np.linspace(-1, 1, self.batch_size)
        test_attrs = [[v] for v in array]

        if not os.path.exists(result_dir + "/generated"):
            os.makedirs(result_dir + "/generated")
        if not os.path.exists(ckp_dir):
            os.makedirs(ckp_dir)

        saver = tf.train.Saver()
        with tf.Session() as sess:

            trained_model = tf.train.latest_checkpoint(ckp_dir)
            saver.restore(sess, trained_model)

            count = 0

            for epoch in range(0, self.num_test_images):
                imgs = self.load_batch(batch_num=0, mode="test", img_path=img_path, epoch_num=epoch)

                temp_output = sess.run([self.o_dec], feed_dict={self.input_imgs:imgs, self.input_attr:test_attrs})

                for outputs in temp_output:
                    for i, output in enumerate(outputs):
                        output = (output + 1) * 255
                        image = Image.fromarray(np.uint8(output))
                        image.save(result_dir + f'/{count}.bmp')
                        count += 1