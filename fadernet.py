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
        self.x = tf.placeholder(tf.float32, [self.batch_size, IMG_SIZE, IMG_SIZE, 3])
        self.input_attr = tf.placeholder(tf.float32, [self.batch_size, 1])


        # Encoder
        self.co16    = tf.layers.conv2d(inputs=self.x, filters=16, kernel_size=(4, 4), strides=(2, 2), padding="same", activation=tf.nn.leaky_relu(features=float32, alpha=0.2))
        self.co32    = tf.layers.conv2d(inputs=self.co16, filters=32, kernel_size=(4, 4), strides=(2, 2), padding="same", activation=tf.nn.leaky_relu(features=float32, alpha=0.2))
        self.co64    = tf.layers.conv2d(inputs=self.co32, filters=64, kernel_size=(4, 4), strides=(2, 2), padding="same", activation=tf.nn.leaky_relu(features=float32, alpha=0.2))
        self.co128   = tf.layers.conv2d(inputs=self.co64, filters=128, kernel_size=(4, 4), strides=(2, 2), padding="same", activation=tf.nn.leaky_relu(features=float32, alpha=0.2))
        self.co256   = tf.layers.conv2d(inputs=self.co128, filters=256, kernel_size=(4, 4), strides=(2, 2), padding="same", activation=tf.nn.leaky_relu(features=float32, alpha=0.2))
        self.co512   = tf.layers.conv2d(inputs=self.co256, filters=512, kernel_size=(4, 4), strides=(2, 2), padding="same", activation=tf.nn.leaky_relu(features=float32, alpha=0.2))
        self.enc_out = tf.layers.conv2d(inputs=self.co512, filters=512, kernel_size=(4, 4), strides=(2, 2), padding="same", activation=tf.nn.leaky_relu(features=float32, alpha=0.2))


        # decoder
        self.y = tf.ones([self.batch_size, 2, 2, 1]) * self.input_attr
        self.in_dec512_0  = tf.concat([self.enc_out, self.y], axis=3)
        self.out_dec512_0 = tf.layers.conv2d_transpose(inputs=self.in_dec512_0, filters=512, kernel_size=(4, 4), strides=(2, 2), padding="same", activation=tf.nn.leaky_relu(features=float32, alpha=0.2))

        self.y4 = tf.ones([self.batch_size, 4, 4, 1]) * self.input_attr
        self.in_dec512_1 =  tf.concat([self.out_dec512_0, self.y4], axis=3)
        self.out_dec512_1 = tf.layers.conv2d_transpose(inputs=self.in_dec512_1, filters=512, kernel_size=(4, 4), strides=(2, 2), padding="same", activation=tf.nn.leaky_relu(features=float32, alpha=0.2))

        self.y8 = tf.ones([self.batch_size, 8, 8, 1]) * self.input_attr
        self.in_dec256  =  tf.concat([self.out_dec512_1, self.y8], axis=3)
        self.out_dec256 = tf.layers.conv2d_transpose(inputs=self.in_dec256, filters=256, kernel_size=(4, 4), strides=(2, 2), padding="same", activation=tf.nn.leaky_relu(features=float32, alpha=0.2))

        self.y16 = tf.ones([self.batch_size, 16, 16, 1]) * self.input_attr
        self.in_dec128  =  tf.concat([self.out_dec256, self.y16], axis=3)
        self.out_dec128 = tf.layers.conv2d_transpose(inputs=self.in_dec128, filters=128, kernel_size=(4, 4), strides=(2, 2), padding="same", activation=tf.nn.leaky_relu(features=float32, alpha=0.2))

        self.y32 = tf.ones([self.batch_size, 32, 32, 1]) * self.input_attr
        self.in_dec64  =  tf.concat([self.out_dec128, self.y32], axis=3)
        self.out_dec64 = tf.layers.conv2d_transpose(inputs=self.in_dec64, filters=64, kernel_size=(4, 4), strides=(2, 2), padding="same", activation=tf.nn.leaky_relu(features=float32, alpha=0.2))

        self.y64 = tf.ones([self.batch_size, 64, 64, 1]) * self.input_attr
        self.in_dec32  =  tf.concat([self.out_dec64, self.y64], axis=3)
        self.out_dec32 = tf.layers.conv2d_transpose(inputs=self.in_dec32, filters=32, kernel_size=(4, 4), strides=(2, 2), padding="same", activation=tf.nn.leaky_relu(features=float32, alpha=0.2))

        self.y128 = tf.ones([self.batch_size, 128, 128, 1]) * self.input_attr
        self.in_dec16  =  tf.concat([self.out_dec32, self.y128], axis=3)
        self.out_dec16 = tf.layers.conv2d_transpose(inputs=self.in_dec16, filters=16, kernel_size=(4, 4), strides=(2, 2), padding="same", activation=tf.nn.leaky_relu(features=float32, alpha=0.2))


        # discriminator
        self.disc_conv = tf.layers.conv2d(inputs=self.enc_out, filters=512, activation=tf.nn.leaky_relu(features=float32, alpha=0.2))
        self.disc_conv = tf.layers.dropout(inputs=self.disc_conv, rate=0.3)
        self.disc_flat = tf.reshape(self.disc_conv, [self.batch_size, 512])
        self.disc_fc1  = tf.layers.dense(inputs=self.disc_flat, units=512)
        self.disc_fc2  = tf.layers.dense(inputs=self.disc_fc1, units=1)
        self.disc_out  = tf.nn.sigmoid(disc_fc2)

    def autoEncoder_loss(self, input_img, output_img):
        return tf.losses.mean_squared_error(labels=input_img, predictions=output_img)

    def discriminator_loss(self, input_y, disc_out):
        return tf.losses.sigmoid_cross_entropy(multi_class_labels=input_y, logits=disc_out)

    def adversarial_loss(self, input_img, output_img, input_y, disc_out, lambda_e):
        loss_ae = tf.losses.mean_squared_error(labels=input_img, predictions=output_img)
        loss_ad = lambda_e * tf.losses.sigmoid_cross_entropy(multi_class_labels=input_y, logits=(1.0-disc_out))

        return loss_ae - loss_ad