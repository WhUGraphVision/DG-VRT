#coding: utf-8
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Activation, merge, Input, Lambda, Reshape,Lambda
from keras import layers,regularizers
from keras.layers import Concatenate,Multiply,Average,Maximum
from keras.layers import Conv2D, Flatten, Dropout, Conv1D,UpSampling2D,Cropping2D,Cropping1D
from keras.layers.pooling import MaxPooling1D, GlobalAveragePooling1D, MaxPooling2D,GlobalAveragePooling2D,AveragePooling2D
from keras.layers import LSTM, GRU, TimeDistributed, Bidirectional
from keras.utils.np_utils import to_categorical
from keras import initializers
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import ZeroPadding2D
from keras.utils import np_utils
from keras.layers.merge import *
import pandas as pd
import numpy as np
import math
import os
import re
import sys
import time
import re
import string
import nltk
from utils import *
import h5py
from PIL import Image
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.optimizers import SGD,Adam
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img,img_to_array
from keras.utils import plot_model
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping,Callback
#from parallel_model import ParallelModel
from keras.backend import tf as ktf
from math import ceil
need_256 = True
import tensorflow as tf
import tensorlayer as tl
import numpy as np
from keras.preprocessing.image import img_to_array, load_img
import random
import os
from keras.optimizers import SGD
import pickle
from PIL import Image
from keras.preprocessing.image import load_img,img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
import keras
from keras import optimizers
from keras.applications import vgg16, xception
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Input, Embedding, Dropout, Flatten, Dense
from keras.models import Model, Sequential
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
#from keras.preprocessing.text import hashing_trick
#from keras.preprocessing.text import text_to_word_sequence
import pandas as pd
from skimage import io,transform
import chardet
import gc
num_labels=200

img_width, img_height = 224, 224
nb_train_samples = 540  #4000
nb_validation_samples = 60 #2000
epochs = 100
batch_size = 32
#input_shape = (img_width, img_height, 3)
early_stop=EarlyStopping(monitor='acc',patience=3,verbose=2,mode='max')
import gc
gc.collect()

###========================Design model ====================================###
img_width, img_height = 224, 224
 
input_shapes=(img_width, img_height,3)
i_shape=(224,224)
    # build the VGG16 network
learning_rate = 1e-3  # Layer specific learning rate
# Weight decay not implemented


def BN(name=""):
    return BatchNormalization(momentum=0.95, name=name, epsilon=1e-5)


class Interp(layers.Layer):

    def __init__(self, new_size, **kwargs):
        self.new_size = new_size
        super(Interp, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Interp, self).build(input_shape)

    def call(self, inputs, **kwargs):
        new_height, new_width = self.new_size
        resized = ktf.image.resize_images(inputs, [new_height, new_width],
                                          align_corners=True)
        return resized

    def compute_output_shape(self, input_shape):
        return tuple([None, self.new_size[0], self.new_size[1], input_shape[3]])

    def get_config(self):
        config = super(Interp, self).get_config()
        config['new_size'] = self.new_size
        return config


# def Interp(x, shape):
#    new_height, new_width = shape
#    resized = ktf.image.resize_images(x, [new_height, new_width],
#                                      align_corners=True)
#    return resized


def residual_conv(prev, level, pad=1, lvl=1, sub_lvl=1, modify_stride=False):
    lvl = str(lvl)
    sub_lvl = str(sub_lvl)
    names = ["conv" + lvl + "_" + sub_lvl + "_1x1_reduce",
             "conv" + lvl + "_" + sub_lvl + "_1x1_reduce_bn",
             "conv" + lvl + "_" + sub_lvl + "_3x3",
             "conv" + lvl + "_" + sub_lvl + "_3x3_bn",
             "conv" + lvl + "_" + sub_lvl + "_1x1_increase",
             "conv" + lvl + "_" + sub_lvl + "_1x1_increase_bn"]
    if modify_stride is False:
        prev = Conv2D(64 * level, (1, 1), strides=(1, 1), name=names[0],
                      use_bias=False)(prev)
    elif modify_stride is True:
        prev = Conv2D(64 * level, (1, 1), strides=(2, 2), name=names[0],
                      use_bias=False)(prev)

    prev = BN(name=names[1])(prev)
    prev = Activation('relu')(prev)

    prev = ZeroPadding2D(padding=(pad, pad))(prev)
    prev = Conv2D(64 * level, (3, 3), strides=(1, 1), dilation_rate=pad,
                  name=names[2], use_bias=False)(prev)

    prev = BN(name=names[3])(prev)
    prev = Activation('relu')(prev)
    prev = Conv2D(256 * level, (1, 1), strides=(1, 1), name=names[4],
                  use_bias=False)(prev)
    prev = BN(name=names[5])(prev)
    return prev


def short_convolution_branch(prev, level, lvl=1, sub_lvl=1, modify_stride=False):
    lvl = str(lvl)
    sub_lvl = str(sub_lvl)
    names = ["conv" + lvl + "_" + sub_lvl + "_1x1_proj",
             "conv" + lvl + "_" + sub_lvl + "_1x1_proj_bn"]

    if modify_stride is False:
        prev = Conv2D(256 * level, (1, 1), strides=(1, 1), name=names[0],
                      use_bias=False)(prev)
    elif modify_stride is True:
        prev = Conv2D(256 * level, (1, 1), strides=(2, 2), name=names[0],
                      use_bias=False)(prev)

    prev = BN(name=names[1])(prev)
    return prev


def empty_branch(prev):
    return prev


def residual_short(prev_layer, level, pad=1, lvl=1, sub_lvl=1, modify_stride=False):
    prev_layer = Activation('relu')(prev_layer)
    block_1 = residual_conv(prev_layer, level,
                            pad=pad, lvl=lvl, sub_lvl=sub_lvl,
                            modify_stride=modify_stride)

    block_2 = short_convolution_branch(prev_layer, level,
                                       lvl=lvl, sub_lvl=sub_lvl,
                                       modify_stride=modify_stride)
    added = Add()([block_1, block_2])
    return added


def residual_empty(prev_layer, level, pad=1, lvl=1, sub_lvl=1):
    prev_layer = Activation('relu')(prev_layer)

    block_1 = residual_conv(prev_layer, level, pad=pad,
                            lvl=lvl, sub_lvl=sub_lvl)
    block_2 = empty_branch(prev_layer)
    added = Add()([block_1, block_2])
    return added


def ResNet(inp, layers):
    # Names for the first couple layers of model
    names = ["conv1_1_3x3_s2",
             "conv1_1_3x3_s2_bn",
             "conv1_2_3x3",
             "conv1_2_3x3_bn",
             "conv1_3_3x3",
             "conv1_3_3x3_bn"]

    # Short branch(only start of network)

    cnv1 = Conv2D(64, (3, 3), strides=(2, 2), padding='same', name=names[0],
                  use_bias=False)(inp)  # "conv1_1_3x3_s2"
    bn1 = BN(name=names[1])(cnv1)  # "conv1_1_3x3_s2/bn"
    relu1 = Activation('relu')(bn1)  # "conv1_1_3x3_s2/relu"

    cnv1 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name=names[2],
                  use_bias=False)(relu1)  # "conv1_2_3x3"
    bn1 = BN(name=names[3])(cnv1)  # "conv1_2_3x3/bn"
    relu1 = Activation('relu')(bn1)  # "conv1_2_3x3/relu"

    cnv1 = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name=names[4],
                  use_bias=False)(relu1)  # "conv1_3_3x3"
    bn1 = BN(name=names[5])(cnv1)  # "conv1_3_3x3/bn"
    relu1 = Activation('relu')(bn1)  # "conv1_3_3x3/relu"

    res = MaxPooling2D(pool_size=(3, 3), padding='same',
                       strides=(2, 2))(relu1)  # "pool1_3x3_s2"

    # ---Residual layers(body of network)

    """
    Modify_stride --Used only once in first 3_1 convolutions block.
    changes stride of first convolution from 1 -> 2
    """

    # 2_1- 2_3
    res = residual_short(res, 1, pad=1, lvl=2, sub_lvl=1)
    for i in range(2):
        res = residual_empty(res, 1, pad=1, lvl=2, sub_lvl=i + 2)

    # 3_1 - 3_3
    res = residual_short(res, 2, pad=1, lvl=3, sub_lvl=1, modify_stride=True)
    for i in range(3):
        res = residual_empty(res, 2, pad=1, lvl=3, sub_lvl=i + 2)
    if layers is 50:
        # 4_1 - 4_6
        res = residual_short(res, 4, pad=2, lvl=4, sub_lvl=1)
        for i in range(5):
            res = residual_empty(res, 4, pad=2, lvl=4, sub_lvl=i + 2)
    elif layers is 101:
        # 4_1 - 4_23
        res = residual_short(res, 4, pad=2, lvl=4, sub_lvl=1)
        for i in range(22):
            res = residual_empty(res, 4, pad=2, lvl=4, sub_lvl=i + 2)
    else:
        print("This ResNet is not implemented")

    # 5_1 - 5_3
    res = residual_short(res, 8, pad=4, lvl=5, sub_lvl=1)
    for i in range(2):
        res = residual_empty(res, 8, pad=4, lvl=5, sub_lvl=i + 2)

    res = Activation('relu')(res)
    return res


def interp_block(prev_layer, level, feature_map_shape, input_shape):
    if input_shape == (224, 224):
        kernel_strides_map = {1: 28,2: 21,3: 14,6: 7}
    else:
        print("Pooling parameters for input shape ",input_shape, " are not defined.")
        exit(1)
    names = [
        "conv5_3_pool" + str(level) + "_conv",
        "conv5_3_pool" + str(level) + "_conv_bn"
    ]
    kernel = (kernel_strides_map[level], kernel_strides_map[level])
    strides = (kernel_strides_map[level], kernel_strides_map[level])
    print(kernel)
    print(strides)
    prev_layer = AveragePooling2D(kernel, strides=strides)(prev_layer)
    prev_layer = Conv2D(512, (1, 1), strides=(1, 1), name=names[0],use_bias=False)(prev_layer)
    prev_layer = BN(name=names[1])(prev_layer)
    prev_layer = Activation('relu')(prev_layer)
    # prev_layer = Lambda(Interp, arguments={
    #                    'shape': feature_map_shape})(prev_layer)
    prev_layer = Interp(feature_map_shape)(prev_layer)
    return prev_layer

def interp_block_t(prev_layer, level, feature_map_shape, input_shape):
    if input_shape == (224, 224):
        kernel_strides_map = {1: 28,2: 21,3: 14,6: 7}
    else:
        print("Pooling parameters for input shape ",input_shape, " are not defined.")
        exit(1)
    names = [
        "conv5_3_pool" + str(level) + "_conv_1",
        "conv5_3_pool" + str(level) + "_conv_bn_1"
    ]
    kernel = (kernel_strides_map[level], kernel_strides_map[level])
    strides = (kernel_strides_map[level], kernel_strides_map[level])
    print(kernel)
    print(strides)
    prev_layer = AveragePooling2D(kernel, strides=strides)(prev_layer)
    prev_layer = Conv2D(512, (1, 1), strides=(1, 1), name=names[0],use_bias=False)(prev_layer)
    prev_layer = BN(name=names[1])(prev_layer)
    prev_layer = Activation('relu')(prev_layer)
    # prev_layer = Lambda(Interp, arguments={
    #                    'shape': feature_map_shape})(prev_layer)
    prev_layer = Interp(feature_map_shape)(prev_layer)
    return prev_layer

def interp_fake_block(num,prev_layer, level, feature_map_shape, input_shape):
    if input_shape == (224, 224):
        kernel_strides_map = {1: 28,2: 21,3: 14,6: 7}
    else:
        print("Pooling parameters for input shape ",input_shape, " are not defined.")
        exit(1)
    names = [
        "conv5_3_pool" + str(level) + "_conv_"+str(num),
        "conv5_3_pool" + str(level) + "_conv_bn_"+str(num)
    ]
    kernel = (kernel_strides_map[level], kernel_strides_map[level])
    strides = (kernel_strides_map[level], kernel_strides_map[level])
    print(kernel)
    print(strides)
    prev_layer = AveragePooling2D(kernel, strides=strides)(prev_layer)
    prev_layer = Conv2D(512, (1, 1), strides=(1, 1), name=names[0],use_bias=False)(prev_layer)
    prev_layer = BN(name=names[1])(prev_layer)
    prev_layer = Activation('relu')(prev_layer)
    # prev_layer = Lambda(Interp, arguments={
    #                    'shape': feature_map_shape})(prev_layer)
    prev_layer = Interp(feature_map_shape)(prev_layer)
    return prev_layer

def build_pyramid_pooling_module(res, input_shape):
    """Build the Pyramid Pooling Module."""
    # ---PSPNet concat layers with Interpolation
    feature_map_size = tuple(int(ceil(input_dim / 8.0))
                             for input_dim in input_shape)
    print("PSP module will interpolate to a final feature map size of %s" %
          (feature_map_size, ))

    interp_block1 = interp_block(res, 1, feature_map_size, input_shape)
    interp_block2 = interp_block(res, 2, feature_map_size, input_shape)
    interp_block3 = interp_block(res, 3, feature_map_size, input_shape)
    interp_block6 = interp_block(res, 6, feature_map_size, input_shape)

    # concat all these layers. resulted
    # shape=(1,feature_map_size_x,feature_map_size_y,4096)
    res = Concatenate()([res,
                         interp_block6,
                         interp_block3,
                         interp_block2,
                         interp_block1])
    return res

def build_fake_pyramid_pooling_module(num,res, input_shape):
    """Build the Pyramid Pooling Module."""
    # ---PSPNet concat layers with Interpolation
    feature_map_size = tuple(int(ceil(input_dim / 8.0))
                             for input_dim in input_shape)
    print("PSP module will interpolate to a final feature map size of %s" %
          (feature_map_size, ))

    interp_block1 = interp_fake_block(num,res, 1, feature_map_size, input_shape)
    interp_block2 = interp_fake_block(num,res, 2, feature_map_size, input_shape)
    interp_block3 = interp_fake_block(num,res, 3, feature_map_size, input_shape)
    interp_block6 = interp_fake_block(num,res, 6, feature_map_size, input_shape)

    # concat all these layers. resulted
    # shape=(1,feature_map_size_x,feature_map_size_y,4096)
    res = Concatenate()([res,
                         interp_block6,
                         interp_block3,
                         interp_block2,
                         interp_block1])
    return res


def build_pyramid_pooling_sub_module(res, input_shape):
    """Build the Pyramid Pooling Module."""
    # ---PSPNet concat layers with Interpolation
    feature_map_size = tuple(int(ceil(input_dim / 8.0))
                             for input_dim in input_shape)
    print("PSP module will interpolate to a final feature map size of %s" %
          (feature_map_size, ))

    interp_block1 = interp_block(res, 1, feature_map_size, input_shape)
    interp_block2 = interp_block(res, 2, feature_map_size, input_shape)
    interp_block3 = interp_block(res, 3, feature_map_size, input_shape)
    interp_block6 = interp_block(res, 6, feature_map_size, input_shape)

    # concat all these layers. resulted
    # shape=(1,feature_map_size_x,feature_map_size_y,4096)
    res = Concatenate()([res,interp_block6,
                         interp_block3,
                         interp_block2,
                         interp_block1])
    return res

def build_pyramid_pooling_mult_module(res,org, input_shape):
    """Build the Pyramid Pooling Module."""
    # ---PSPNet concat layers with Interpolation
    feature_map_size = tuple(int(ceil(input_dim / 8.0))
                             for input_dim in input_shape)
    print("PSP module will interpolate to a final feature map size of %s" %(feature_map_size, ))
    interp_rblock1 = interp_block(res, 1, feature_map_size, input_shape)
    interp_rblock2 = interp_block(res, 2, feature_map_size, input_shape)
    interp_rblock3 = interp_block(res, 3, feature_map_size, input_shape)
    interp_rblock6 = interp_block(res, 6, feature_map_size, input_shape)
    interp_oblock1 = interp_block_t(org, 1, feature_map_size, input_shape)
    interp_oblock2 = interp_block_t(org, 2, feature_map_size, input_shape)
    interp_oblock3 = interp_block_t(org, 3, feature_map_size, input_shape)
    interp_oblock6 = interp_block_t(org, 6, feature_map_size, input_shape)
    interp_block1 =Multiply()([interp_rblock1, interp_oblock1])
    interp_block2 =Multiply()([interp_rblock2, interp_oblock2])
    interp_block3 =Multiply()([interp_rblock3, interp_oblock3])
    interp_block6 =Multiply()([interp_rblock6, interp_oblock6])
    rr=Multiply()([res, org])
    re1 = Concatenate()([rr,
                         interp_block6,
                         interp_block3,
                         interp_block2,
                         interp_block1])
    return re1

def build_pyramid_pooling_aver_module(res,org, input_shape):
    """Build the Pyramid Pooling Module."""
    # ---PSPNet concat layers with Interpolation
    feature_map_size = tuple(int(ceil(input_dim / 8.0))
                             for input_dim in input_shape)
    print("PSP module will interpolate to a final feature map size of %s" %(feature_map_size, ))
    interp_rblock1 = interp_block(res, 1, feature_map_size, input_shape)
    interp_rblock2 = interp_block(res, 2, feature_map_size, input_shape)
    interp_rblock3 = interp_block(res, 3, feature_map_size, input_shape)
    interp_rblock6 = interp_block(res, 6, feature_map_size, input_shape)
    interp_oblock1 = interp_block_t(org, 1, feature_map_size, input_shape)
    interp_oblock2 = interp_block_t(org, 2, feature_map_size, input_shape)
    interp_oblock3 = interp_block_t(org, 3, feature_map_size, input_shape)
    interp_oblock6 = interp_block_t(org, 6, feature_map_size, input_shape)
    interp_block1 =Average()([interp_rblock1, interp_oblock1])
    interp_block2 =Average()([interp_rblock2, interp_oblock2])
    interp_block3 =Average()([interp_rblock3, interp_oblock3])
    interp_block6 =Average()([interp_rblock6, interp_oblock6])
    #rr=Multiply()([res, org])
    re1 = Concatenate()([org,
                         interp_block6,
                         interp_block3,
                         interp_block2,
                         interp_block1])
    return re1


def build_pspnet(nb_classes, resnet_layers, input_shape, activation='softmax'):
    """Build PSPNet."""
    print("Building a PSPNet based on ResNet %i expecting inputs of shape %s predicting %i classes" % (resnet_layers, input_shape, nb_classes))
    inp = Input((input_shape[0], input_shape[1], 3))
    res = ResNet(inp, layers=resnet_layers)
    print (res.shape)
    psp = build_pyramid_pooling_module(res, input_shape)
    print (psp.shape)
    x = Conv2D(512, (3, 3), strides=(1, 1), padding="same", name="conv5_4",use_bias=False)(psp)
    x = BN(name="conv5_4_bn")(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)
    x = Conv2D(nb_classes, (1, 1), strides=(1, 1), name="conv6")(x)
    # x = Lambda(Interp, arguments={'shape': (
    #    input_shape[0], input_shape[1])})(x)
    x = Interp([input_shape[0], input_shape[1]])(x)
    x = Activation('softmax')(x)
    model = Model(inputs=inp, outputs=x)
    model.summary()
    # Solver
    sgd = SGD(lr=learning_rate, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])
    return model

#1: get weigth,2 
def identity_block(X, f, filters, stage, block):
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    # Second component of main path (3 lines)
    X = Conv2D(filters= F2, kernel_size=(f,f),strides=(1,1),padding='same',name=conv_name_base + '2b')(X)
    X = BatchNormalization(axis=3,name=bn_name_base+'2b')(X)
    X = Activation('relu')(X)
 
    # Third component of main path ( lines)
    X = Conv2D(filters=F3,kernel_size=(1,1),strides=(1,1),padding='valid',name=conv_name_base+'2c')(X)
    X = BatchNormalization(axis=3,name=bn_name_base+'2c')(X)
 
    # Final step: Add shortcut value to main path, and pass it through a RELU activation (lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    return X
 
def convolutional_block(X, f, filters, stage, block, s = 2):
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X
 
    ##### MAIN PATH #####
    # First component of main path 
    X = Conv2D(F1, (1, 1), strides = (s,s),padding='valid',name = conv_name_base + '2a')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
 
    # Second component of main path 
    X = Conv2D(F2,(f,f),strides=(1,1),padding='same',name=conv_name_base+'2b')(X)
    X = BatchNormalization(axis=3,name=bn_name_base+'2b')(X)
    X = Activation('relu')(X)
 
    # Third component of main path 
    X = Conv2D(F3,(1,1),strides=(1,1),padding='valid',name=conv_name_base+'2c')(X)
    X = BatchNormalization(axis=3,name=bn_name_base+'2c')(X)
 
    ##### SHORTCUT PATH #### 
    X_shortcut = Conv2D(F3,(1,1),strides=(s,s),padding='valid',name=conv_name_base+'1')(X_shortcut)
    X_shortcut = BatchNormalization(axis=3,name =bn_name_base+'1')(X_shortcut)
 
    # Final step: Add shortcut value to main path, and pass it through a RELU activation 
    X = Add()([X,X_shortcut])
    X = Activation('relu')(X)
    return X
    
# GRADED FUNCTION: ResNet50
 
def RResNet50(input_shape = (64, 64, 3), classes=200):
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)
 
    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)
    
    # Stage 1
    X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1')(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)
 
    # Stage 2
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')
 
    ### START CODE HERE ###
 
    # Stage 3 
    X = convolutional_block(X, f = 3,filters= [128,128,512],stage=3,block='a',s=2)
    X = identity_block(X,3,[128,128,512],stage=3,block='b')
    X = identity_block(X,3,[128,128,512],stage=3,block='c')
    X = identity_block(X,3,[128,128,512],stage=3,block='d')
 
    # Stage 4 
    X = convolutional_block(X,f=3,filters=[256,256,1024],stage=4,block='a',s=2)
    X = identity_block(X,3,[256,256,1024],stage=4,block='b')
    X = identity_block(X,3,[256,256,1024],stage=4,block='c')
    X = identity_block(X,3,[256,256,1024],stage=4,block='d')
    X = identity_block(X,3,[256,256,1024],stage=4,block='e')
    X = identity_block(X,3,[256,256,1024],stage=4,block='f')
 
    # Stage 5 
    X = convolutional_block(X, f = 3,filters= [512,512,2048],stage=5,block='a',s=2)
    X = identity_block(X,3,[512,512,2048],stage=5,block='b')
    X = identity_block(X,3,[512,512,2048],stage=5,block='c')
 
    # AVGPOOL 
    X = AveragePooling2D((2,2),strides=(2,2))(X)
 
    # output layer
    X = Flatten()(X)
    model = Model(inputs = X_input, outputs = X, name='ResNet50')
 
    return model

from keras.applications.resnet50 import ResNet50

def create_resnet50(input_img):
    net = ResNet50(weights='imagenet', include_top=False,
                      input_tensor=input_img)

    for layer in net.layers[1:]:
        layer.trainable = False
    net = Reshape((-1,))(net.outputs[0])
    return net 

def true_ResNet50(classes):
    base_model = RResNet50(input_shape=(224,224,3),classes=200)
    base_model.load_weights('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
    for layer in base_model.layers:
        layer.trainable=False
    res = base_model.get_layer('activation_49').output
   # print res.shape
    res = BatchNormalization()(res)
    model = Model(inputs=base_model.input, outputs=res,name='true-ResNet50')
    #model.summary()
    return model

def fake_ResNet50_base(index,input_shape=(224,224,3),classes=200):    
    base_model = RResNet50(input_shape=(224,224,3),classes=200)
    base_model.load_weights('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
    for layer in base_model.layers:
        layer.trainable=False
        layer.name = layer.name + str("_")+str(index)
    base_model.summary()
    Num=(index+2)*49+index*6
    res_layer='activation_'+str(Num)+ str("_")+str(index)
    res = base_model.get_layer(res_layer).output
    #print res.shape
    res = BatchNormalization()(res)
    model = Model(inputs=base_model.input, outputs=res)
    return model


def fake_ResNet50_base_new(index,input_shape=(224,224,3),classes=200):    
    base_model = RResNet50(input_shape=(224,224,3),classes=200)
    base_model.load_weights('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
    for layer in base_model.layers:
        layer.trainable=False
        layer.name = layer.name + str("_")+str(index)
    #base_model.summary()
    Num=(index+2)*49+index*6
    res_layer='activation_'+str(Num)+ str("_")+str(index)
    res = base_model.get_layer(res_layer).output
    #print res.shape
    res = BatchNormalization()(res)
    model = Model(inputs=base_model.input, outputs=res)
    return model

def text_cnnmodel(classes=200):
    main_input = Input(shape=(64,), dtype='float64')
    embedder = Embedding(len(vocab) + 1, 256, input_length = 64)
    #embedder = Embedding(9999, 256, input_length = 64)
    embed = embedder(main_input)
    conv1_1 = Conv1D(256, 3, padding='same')(embed)
    bn1_1 = BatchNormalization()(conv1_1)
    relu1_1 = Activation('relu')(bn1_1)
    conv1_2 = Conv1D(128, 3, padding='same')(relu1_1)
    bn1_2 = BatchNormalization()(conv1_2)
    relu1_2 = Activation('relu')(bn1_2)
    cnn1 = MaxPooling1D(pool_size=4)(relu1_2)
# kernel_size = 4
    conv2_1 = Conv1D(256, 4, padding='same')(embed)
    bn2_1 = BatchNormalization()(conv2_1)
    relu2_1 = Activation('relu')(bn2_1)
    conv2_2 = Conv1D(128, 4, padding='same')(relu2_1)
    bn2_2 = BatchNormalization()(conv2_2)
    relu2_2 = Activation('relu')(bn2_2)
    cnn2 = MaxPooling1D(pool_size=4)(relu2_2)
# kernel_size = 5
    conv3_1 = Conv1D(256, 5, padding='same')(embed)
    bn3_1 = BatchNormalization()(conv3_1)
    relu3_1 = Activation('relu')(bn3_1)
    conv3_2 = Conv1D(128, 5, padding='same')(relu3_1)
    bn3_2 = BatchNormalization()(conv3_2)
    relu3_2 = Activation('relu')(bn3_2)
    cnn3 = MaxPooling1D(pool_size=4)(relu3_2)
# 
    conc = Concatenate()([cnn1,cnn2,cnn3])  
    flat = Flatten()(conc)
    drop = Dropout(0.5)(flat)
    fc = Dense(2048)(drop)
    bn = BatchNormalization(name='bn')(fc)
    model = Model(inputs = main_input, outputs = bn)
    return model



def text_cnnmodel_base(index,classes):
    base_model = text_cnnmodel(classes)
    for layer in base_model.layers:
        layer.trainable=False
        layer.name = layer.name + str("_")+str(index)   
    res = base_model.output
    #print res.shape
    model = Model(inputs=base_model.input, outputs=res)
    return model

#es = EarlyStopping(monitor='val_loss', patience=1)
#model.fit(x=X_train,y=Y_train,epochs=20,batch_size=32,validation_data=(X_val, Y_val),callbacks=[es])
#tt=build_pspnet(102, 50, input_shape=(224,224), activation='softmax')
def mult_text_cnnmodel(classes):
    capt1_model=text_cnnmodel_base(0,classes)
    capt1_feature=capt1_model.output
    capt1_in=capt1_model.input
    capt2_model=text_cnnmodel_base(1,classes)
    capt2_feature=capt2_model.output
    capt2_in=capt2_model.input
    capt3_model=text_cnnmodel_base(2,classes)
    capt3_feature=capt3_model.output
    capt3_in=capt3_model.input
    capt4_model=text_cnnmodel_base(3,classes)
    capt4_feature=capt4_model.output
    capt4_in=capt4_model.input
    capt5_model=text_cnnmodel_base(4,classes)
    capt5_feature=capt5_model.output
    capt5_in=capt5_model.input
    capt6_model=text_cnnmodel_base(5,classes)
    capt6_feature=capt6_model.output
    capt6_in=capt6_model.input
    capt7_model=text_cnnmodel_base(6,classes)
    capt7_feature=capt7_model.output
    capt7_in=capt7_model.input
    capt8_model=text_cnnmodel_base(7,classes)
    capt8_feature=capt8_model.output
    capt8_in=capt8_model.input
    capt9_model=text_cnnmodel_base(8,classes)
    capt9_feature=capt9_model.output
    capt9_in=capt9_model.input
    capt10_model=text_cnnmodel_base(9,classes)
    capt10_feature=capt10_model.output
    capt10_in=capt10_model.input
    outs =Average()([capt1_feature, capt2_feature,capt3_feature, capt4_feature,capt5_feature,capt6_feature,capt7_feature, capt8_feature,capt9_feature, capt10_feature])
    model = Model(inputs= [capt1_in,capt2_in,capt3_in,capt4_in,capt5_in,capt6_in,capt7_in,capt8_in,capt9_in,capt10_in], outputs=outs,name='mult_text_cnnmodel')
    #model.summary()
    return model

def fake_ResNet50_new(classes):
    fake_base_model1=fake_ResNet50_base(0,input_shape = (224, 224, 3),classes=200)
    temp_feature1=fake_base_model1.output
    in1=fake_base_model1.input
    fake_base_model2=fake_ResNet50_base(1,input_shape = (224, 224, 3),classes=200)
    temp_feature2=fake_base_model2.output
    in2=fake_base_model2.input
    fake_base_model3=fake_ResNet50_base(2,input_shape = (224, 224, 3),classes=200)
    temp_feature3=fake_base_model3.output
    in3=fake_base_model3.input
    fake_base_model4=fake_ResNet50_base(3,input_shape = (224, 224, 3),classes=200)
    temp_feature4=fake_base_model4.output
    in4=fake_base_model4.input
    fake_base_model5=fake_ResNet50_base(4,input_shape = (224, 224, 3),classes=200)
    temp_feature5=fake_base_model5.output
    in5=fake_base_model5.input
    fake_base_model6=fake_ResNet50_base(5,input_shape = (224, 224, 3),classes=200)
    temp_feature6=fake_base_model6.output
    in6=fake_base_model6.input
    fake_base_model7=fake_ResNet50_base(6,input_shape = (224, 224, 3),classes=200)
    temp_feature7=fake_base_model7.output
    in7=fake_base_model7.input
    fake_base_model8=fake_ResNet50_base(7,input_shape = (224, 224, 3),classes=200)
    temp_feature8=fake_base_model8.output
    in8=fake_base_model8.input
    fake_base_model9=fake_ResNet50_base(8,input_shape = (224, 224, 3),classes=200)
    temp_feature9=fake_base_model9.output
    in9=fake_base_model9.input
    fake_base_model10=fake_ResNet50_base(9,input_shape = (224, 224, 3),classes=200)
    temp_feature10=fake_base_model10.output
    in10=fake_base_model10.input
    outs =Average()([temp_feature1, temp_feature2,temp_feature3, temp_feature4,temp_feature5,temp_feature6,temp_feature7, temp_feature8,temp_feature9, temp_feature10])
    model = Model(inputs=[in1,in2,in3,in4,in5,in6,in7,in8,in9,in10], outputs=outs,name='fake-ResNet50')
    return model
    
def true_text_ResNet50_2(classes =200):
    print ('bulid true image model')
    true_image_model = true_ResNet50( classes =200)
    
    output1=true_image_model.output
    input1=true_image_model.input
    #output1=Conv2D(512, (1, 1), padding='same', activation='relu')(output1)
    print('bulid caption model')
    text_model=mult_text_cnnmodel(classes=200)
    output3=text_model.output
    input3=text_model.input
    merged=Add()([output1,output3])
    Flat= Flatten()(merged)
    Dor=Dropout(0.5)(Flat)
    fc = Dense(512)(Dor)
    bnn = BatchNormalization(name='bn2')(fc)
    Den=Dense(classes, activation='softmax')(bnn)
    m_model=Model(inputs=[input1,input3[0],input3[1],input3[2],input3[3],input3[4],input3[5],input3[6],input3[7],input3[8],input3[9]], outputs=Den)
    #plot_model(s_model, to_file='true-fake-restnet50-fine-20181104.png',show_shapes=True)
    #m_model.summary()
    return m_model

def true_fake_text_ResNet50_3(classes =200):
    print('bulid true image model')
    true_image_model = true_ResNet50( classes =200)
    print ('bulid fake image model')
    fake_image_model = fake_ResNet50_new( classes =200)
    output1=true_image_model.output
    input1=true_image_model.input
    output2=fake_image_model.output
    input2=fake_image_model.input
   # print(input1.shape)
   # print(input2)
    print ('bulid caption model')
    text_model=mult_text_cnnmodel_new(classes=200)
    output3=text_model.output
    input3=text_model.input
    merged=Add()([output2,output3])
    print(output2.shape)
    print(output3.shape)
    print(merged.shape)
    Flat= Flatten()(merged)
    Dor=Dropout(0.5)(Flat)
    fc = Dense(2048)(Dor)
    bnn = BatchNormalization(name='bn2')(fc)
    merged1=Add()([output1,bnn])
    Flat1= Flatten()(merged1)
    Dor1=Dropout(0.5)(Flat1)
    fc1 = Dense(512)(Dor1)
    #fc2=Dropout(0.6)(fc1)
    bnn1 = BatchNormalization(name='bn3')(fc1)
    Den1=Dense(classes, activation='softmax')(bnn1)
    m_model=Model(inputs=[input1,input2[0],input2[1],input2[2],input2[3],input2[4],input2[5],input2[6],input2[7],input2[8],input2[9],input3[0],input3[1],input3[2],input3[3],input3[4],input3[5],input3[6],input3[7],input3[8],input3[9]], outputs=Den1)

    #plot_model(s_model, to_file='true-fake-restnet50-fine-20181104.png',show_shapes=True)
   # m_model.summary()
    return m_model

#from keras_attention_block import *


def true_fake_text_ResNet50_4(classes):
    print('bulid true image model')
    true_image_model = true_ResNet50( classes )
    print ('bulid fake image model')
    fake_image_model = fake_ResNet50_new( classes)
    output1=true_image_model.output
    input1=true_image_model.input
    output2=fake_image_model.output
    input2=fake_image_model.input
    print(input1.shape)
    print(input2)
    print ('bulid caption model')
    text_model=mult_text_cnnmodel_new(classes)
    output3=text_model.output
    input3=text_model.input
    attentIuput1=SelfAttention1DLayer(similarity="multiplicative",dropout_rate=0.2)(output1)
    attentIuput2=SelfAttention1DLayer(similarity="multiplicative",dropout_rate=0.2)(output2)
    attentIuput3=SelfAttention1DLayer(similarity="multiplicative",dropout_rate=0.2)(output3)

    merged=Add()([attentInput2,attentInput3])
    Flat= Flatten()(merged)
    Dor=Dropout(0.1)(Flat)
    fc = Dense(2048)(Dor)
    bnn = BatchNormalization(name='bn2')(fc)
    merged1=Add()([attentInput1,bnn])
    Flat1= Flatten()(merged1)
    Dor1=Dropout(0.1)(Flat1)
    fc1 = Dense(512)(Dor1)
    bnn1 = BatchNormalization(name='bn3')(fc1)
    Den1=Dense(classes, activation='softmax')(bnn1)
    m_model=Model(inputs=[input1,input2[0],input2[1],input2[2],input2[3],input2[4],input2[5],input2[6],input2[7],input2[8],input2[9],input3[0],input3[1],input3[2],input3[3],input3[4],input3[5],input3[6],input3[7],input3[8],input3[9]], outputs=Den1)

        #plot_model(s_model, to_file='true-fake-restnet50-fine-20181104.png',show_shapes=True)
    #m_model.summary()
    return m_model
    
def true_fake_ResNet50(classes =200):
    #print 'bulid true image model'
    true_image_model = true_ResNet50( classes =200)
    #print 'bulid fake image model'
    fake_image_model = fake_ResNet50_new( classes =200)
    output1=true_image_model.output
    input1=true_image_model.input
    output2=fake_image_model.output
    input2=fake_image_model.input
    #print input1.shape
    #print input2
    merged=Add()([output1,output2])
    #print merged.shape
    Flat= Flatten()(merged)
    Dor=Dropout(0.5)(Flat)
    fc = Dense(512)(Dor)
    bnn = BatchNormalization()(fc)
    Den1=Dense(classes, activation='softmax')(bnn)
    s_model=Model(inputs=[input1,input2[0],input2[1],input2[2],input2[3],input2[4],input2[5],input2[6],input2[7],input2[8],input2[9]], outputs=Den1)
    #plot_model(s_model, to_file='true-fake-restnet50-fine-20181104.png',show_shapes=True)
    s_model.summary()
    return s_model

def Our_ResNet50(classes):
    #K.set_learning_phase(0)
    base_model = RResNet50(input_shape=(224,224,3),classes=200) 
    base_model.load_weights('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
    for layer in base_model.layers:
        layer.trainable=False
    res = base_model.get_layer('activation_49').output
   # print res.shape
    #K.set_learning_phase(1)
    #x = GlobalAveragePooling2D()(res)
    x = Flatten()(res)
    #res1 = Activation('relu')(x)
    predictions = Dense(classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.summary()
    return model
    
import keras_resnet.models
def Our_ResNet50_1(classes=200):
    #K.set_learning_phase(0)
    shape=(224,224,3)
    #x=keras.layers.Input(shape)
    #base_model = keras_resnet.models.ResNet50(x, classes=102)
    #predictions=base_model.output
    #model = Model(inputs=base_model.input, outputs=predictions)
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=shape)
    for layer in base_model.layers:
        layer.trainable = False
    x = base_model.output
    #K.set_learning_phase(1)
    x = Flatten(name='flatten')(x)
    predictions = Dense(classes, activation='softmax', name='predictions')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.summary()
    for layer in model.layers[:141]:
        layer.trainable=False
    for layer in model.layers[141:]:
        layer.trainable=True
    return model
def Our_ResNet50_2(classes=200):
 #   K.set_learning_phase(0)
    model = keras.applications.resnet50.ResNet50()
    model.layers.pop()
    for layer in model.layers:
        layer.trainable=False
    last = model.layers[-1].output
    x = Dense(classes, activation="softmax")(last)
    finetuned_model = Model(model.input, x)
    finetuned_model.summary()
    return finetuned_model
    
def Our_ResNet50_based_2(classes=200):
 #   K.set_learning_phase(0)
    model = keras.applications.resnet50.ResNet50()
    model.layers.pop()
    #model.summary()
    for layer in model.layers:
        layer.trainable=False
    res_layer='activation_49'
    last = model.get_layer(res_layer).output
    finetuned_model = Model(model.input, last)
    finetuned_model.summary()
    return finetuned_model
    
def Our_ResNet50_facke_based_2(index,classes=200):
 #   K.set_learning_phase(0)
    base_model = keras.applications.resnet50.ResNet50()
    base_model.layers.pop()
    #base_model.summary()
    for layer in base_model.layers:
        layer.trainable=False
        layer.name = layer.name + str("_")+str(index)
    last = base_model.layers[-1].output
    #print(base_model.layers[-1])
    Num=(index+2)*49+index*6
    res_layer='activation_'+str(Num)+ str("_")+str(index)
    #res_layer='activation_'+str(Num)
    print(res_layer)
    res = base_model.get_layer(res_layer).output
    finetuned_model = Model(base_model.input, last)
    #finetuned_model.summary()
    return finetuned_model
    

    
def true_text_ResNet50_2(classes):
    print ('bulid true image model')
    true_image_model = true_ResNet50( classes )
    
    output1=true_image_model.output
    input1=true_image_model.input
    #output1=Conv2D(512, (1, 1), padding='same', activation='relu')(output1)
    #print input1.shape
    print ('bulid caption model')
    text_model=mult_text_cnnmodel(classes)
    output3=text_model.output
    input3=text_model.input
    merged=Add()([output1,output3])
    Flat= Flatten()(merged)
    Dor=Dropout(0.5)(Flat)
    fc = Dense(512)(Dor)
    bnn = BatchNormalization(name='bn2')(fc)
    Den=Dense(classes, activation='softmax')(bnn)
    m_model=Model(inputs=[input1,input3[0],input3[1],input3[2],input3[3],input3[4],input3[5],input3[6],input3[7],input3[8],input3[9]], outputs=Den)
    #plot_model(s_model, to_file='true-fake-restnet50-fine-20181104.png',show_shapes=True)
    m_model.summary()
    return m_model
    
def caption_fake1_ResNet50_2(index,classes):
    print('merge the fake images %d' % index)
    fake_base_model1=Our_ResNet50_facke_based_2(index,classes=200)
    temp_feature1=fake_base_model1.output
    in1=fake_base_model1.input
    #print(temp_feature1.shape)
    caption_model=text_cnnmodel_base(index,classes)
    caption_feature=caption_model.output
    in2=caption_model.input
    merged=Add()([temp_feature1,caption_feature])
    #Flat= Flatten()(merged)
    #Dor=Dropout(0.5)(Flat)
    #fc = Dense(2048)(Dor)
    model=Model(inputs=[in1,in2],outputs=merged,name='caption_fake1_ResNet50')
    return model

def Muit_fake1_Feature_model_2(classes):
    print('bulid caption_fakeImage model')
    fakeCaption_model1=caption_fake1_ResNet50_2(0,classes)
    fakeCaption_featuer1=fakeCaption_model1.output
    in1=fakeCaption_model1.input
    fakeCaption_model2=caption_fake1_ResNet50_2(1,classes)
    fakeCaption_featuer2=fakeCaption_model2.output
    in2=fakeCaption_model2.input
    fakeCaption_model3=caption_fake1_ResNet50_2(2,classes)
    fakeCaption_featuer3=fakeCaption_model3.output
    in3=fakeCaption_model3.input
    fakeCaption_model4=caption_fake1_ResNet50_2(3,classes)
    fakeCaption_featuer4=fakeCaption_model4.output
    in4=fakeCaption_model4.input
    fakeCaption_model5=caption_fake1_ResNet50_2(4,classes)
    fakeCaption_featuer5=fakeCaption_model5.output
    in5=fakeCaption_model5.input
    fakeCaption_model6=caption_fake1_ResNet50_2(5,classes)
    fakeCaption_featuer6=fakeCaption_model6.output
    in6=fakeCaption_model6.input
    fakeCaption_model7=caption_fake1_ResNet50_2(6,classes)
    fakeCaption_featuer7=fakeCaption_model7.output
    in7=fakeCaption_model7.input
    fakeCaption_model8=caption_fake1_ResNet50_2(7,classes)
    fakeCaption_featuer8=fakeCaption_model8.output
    in8=fakeCaption_model8.input
    fakeCaption_model9=caption_fake1_ResNet50_2(8,classes)
    fakeCaption_featuer9=fakeCaption_model9.output
    in9=fakeCaption_model9.input
    fakeCaption_model10=caption_fake1_ResNet50_2(9,classes)
    fakeCaption_featuer10=fakeCaption_model10.output
    in10=fakeCaption_model10.input

    print(fakeCaption_featuer1.shape)
    outs =Average()([fakeCaption_featuer1, fakeCaption_featuer2,fakeCaption_featuer3,fakeCaption_featuer4, fakeCaption_featuer5,fakeCaption_featuer6,fakeCaption_featuer7, fakeCaption_featuer8,fakeCaption_featuer9, fakeCaption_featuer10])
    #print(outs.shape)
    model = Model(inputs= [in1[0],in1[1],in2[0],in2[1],in3[0],in3[1],in4[0],in4[1],in5[0],in5[1],in6[0],in6[1],in7[0],in7[1],in8[0],in8[1],in9[0],in9[1],in10[0],in10[1]], outputs=outs,name='Muit_fake1_Feain1[0],in1[1],ture_model')
    return model
    
def finnal_muilt1Feature_model_2(classes):
    print('bulid true image model')
    true_image_model = Our_ResNet50_based_2(classes)
    true_image_feature=true_image_model.output
    in0=true_image_model.input
    print('build Muit_fake_Feature_model')
    mult_fake1_caption_model=Muit_fake1_Feature_model_2(classes)
    mult_fake1_caption_feature=mult_fake1_caption_model.output
    in1=mult_fake1_caption_model.input
    print(mult_fake1_caption_feature.shape)
    merged=Add()([true_image_feature,mult_fake1_caption_feature])
    print(merged.shape)
    Flat= Flatten()(merged)
    Dor=Dropout(0.5)(Flat)
    fc = Dense(512)(Dor)
    bnn = BatchNormalization(name='bn2')(fc)
    Den=Dense(classes, activation='softmax')(bnn)
    m_model=Model(inputs=[in0,in1[:][0],in1[:][1],in1[:][2],in1[:][3],in1[:][4],in1[:][5],in1[:][6],in1[:][7],in1[:][8],in1[:][9],in1[:][10],in1[:][11],in1[:][12],in1[:][13],in1[:][14],in1[:][15],in1[:][16],in1[:][17],in1[:][18],in1[:][19]], outputs=Den)
    return m_model


def caption_fake1_ResNet50(index,classes):
    print('merge the fake images %d' % index)
    fake_base_model1=fake_ResNet50_base_new(index,input_shape = (224, 224, 3),classes=200)
    temp_feature1=fake_base_model1.output
    in1=fake_base_model1.input
    #print(temp_feature1.shape)
    caption_model=text_cnnmodel_base(index,classes)
    caption_feature=caption_model.output
    in2=caption_model.input
    merged=Add()([temp_feature1,caption_feature])
    #Flat= Flatten()(merged)
    #Dor=Dropout(0.5)(Flat)
    #fc = Dense(2048)(Dor)
    model=Model(inputs=[in1,in2],outputs=merged,name='caption_fake1_ResNet50')
    return model

def Muit_fake1_Feature_model(classes):
    print('bulid caption_fakeImage model')
    fakeCaption_model1=caption_fake1_ResNet50(0,classes)
    fakeCaption_featuer1=fakeCaption_model1.output
    in1=fakeCaption_model1.input
    fakeCaption_model2=caption_fake1_ResNet50(1,classes)
    fakeCaption_featuer2=fakeCaption_model2.output
    in2=fakeCaption_model2.input
    fakeCaption_model3=caption_fake1_ResNet50(2,classes)
    fakeCaption_featuer3=fakeCaption_model3.output
    in3=fakeCaption_model3.input
    fakeCaption_model4=caption_fake1_ResNet50(3,classes)
    fakeCaption_featuer4=fakeCaption_model4.output
    in4=fakeCaption_model4.input
    fakeCaption_model5=caption_fake1_ResNet50(4,classes)
    fakeCaption_featuer5=fakeCaption_model5.output
    in5=fakeCaption_model5.input
    fakeCaption_model6=caption_fake1_ResNet50(5,classes)
    fakeCaption_featuer6=fakeCaption_model6.output
    in6=fakeCaption_model6.input
    fakeCaption_model7=caption_fake1_ResNet50(6,classes)
    fakeCaption_featuer7=fakeCaption_model7.output
    in7=fakeCaption_model7.input
    fakeCaption_model8=caption_fake1_ResNet50(7,classes)
    fakeCaption_featuer8=fakeCaption_model8.output
    in8=fakeCaption_model8.input
    fakeCaption_model9=caption_fake1_ResNet50(8,classes)
    fakeCaption_featuer9=fakeCaption_model9.output
    in9=fakeCaption_model9.input
    fakeCaption_model10=caption_fake1_ResNet50(9,classes)
    fakeCaption_featuer10=fakeCaption_model10.output
    in10=fakeCaption_model10.input

    #print(fakeCaption_featuer1.shape)
    outs =Average()([fakeCaption_featuer1, fakeCaption_featuer2,fakeCaption_featuer3,fakeCaption_featuer4, fakeCaption_featuer5,fakeCaption_featuer6,fakeCaption_featuer7, fakeCaption_featuer8,fakeCaption_featuer9, fakeCaption_featuer10])
    #print(outs.shape)
    model = Model(inputs= [in1[0],in1[1],in2[0],in2[1],in3[0],in3[1],in4[0],in4[1],in5[0],in5[1],in6[0],in6[1],in7[0],in7[1],in8[0],in8[1],in9[0],in9[1],in10[0],in10[1]], outputs=outs,name='Muit_fake1_Feain1[0],in1[1],ture_model')
    return model
    
def finnal_muilt1Feature_model(classes):
    print('bulid true image model')
    true_image_model = true_ResNet50(classes)
    true_image_feature=true_image_model.output
    in0=true_image_model.input
    print('build Muit_fake_Feature_model')
    mult_fake1_caption_model=Muit_fake1_Feature_model(classes)
    mult_fake1_caption_feature=mult_fake1_caption_model.output
    in1=mult_fake1_caption_model.input
    merged=Add()([true_image_feature,mult_fake1_caption_feature])
    Flat= Flatten()(merged)
    Dor=Dropout(0.5)(Flat)
    fc = Dense(512)(Dor)
    bnn = BatchNormalization(name='bn2')(fc)
    Den=Dense(classes, activation='softmax')(bnn)
    #in1_data=in1[0][0]
    #print(in1_data.shape)
    #in2_data=in1[1][0]
    #print(in2_data.shape)
   
    #m_model=Model(inputs=[in0,in1_0,in1_1,in2_0,in2_1,in3_0,in3_1,in4_0,in4_1,in5_0,in5_1,in6_0,in6_1,in7_0,in7_1,in8_0,in8_1,in9_0,in9_1,in10_0,in10_1],outputs=Den)
    m_model=Model(inputs=[in0,in1[:][0],in1[:][1],in1[:][2],in1[:][3],in1[:][4],in1[:][5],in1[:][6],in1[:][7],in1[:][8],in1[:][9],in1[:][10],in1[:][11],in1[:][12],in1[:][13],in1[:][14],in1[:][15],in1[:][16],in1[:][17],in1[:][18],in1[:][19]], outputs=Den)
    #m_model=Model(inputs=[in0,inall],outputs=Den)
    #plot_model(s_model, to_file='true-fake-restnet50-fine-20181104.png',show_shapes=True)
    m_model.summary()
    return m_model
    
    
    
def fake2_ResNet50(index,classes):
    t1=index+0
    fake_base_model1=fake_ResNet50_base(t1,input_shape = (224, 224, 3),classes=200)
    temp_feature1=fake_base_model1.output
    in1=fake_base_model1.input
    t2=index+1
    fake_base_model2=fake_ResNet50_base(t2,input_shape = (224, 224, 3),classes=200)
    temp_feature2=fake_base_model2.output
    in2=fake_base_model2.input
    outs =Average()([temp_feature1, temp_feature2])
    model = Model(inputs= [in1,in2], outputs=outs,name='fake-ResNet50')
    return model
    
def caption_fake2_ResNet50(index,classes):
    print('merge the fake images')
    mult_fake_model=fake3_ResNet50(classes)
    mult_fakeimage_feature=mult_fake_model.output
    in1=mult_fake_model.input
    nIndex=index*2
    caption_model=text_cnnmodel_base(nIndex,classes)
    caption_feature=caption_model.output
    in2=caption_model.input
    merged=Add()([mult_fakeimage_feature,caption_feature])
    Flat= Flatten()(merged)
    Dor=Dropout(0.1)(Flat)
    fc = Dense(2048)(Dor)
    model=Model(inputs=[in1,in2],outputs=fc,name='caption_fake2_ResNet50')
    return model
    
def Muit_fake2_Feature_model(classes):
    print('bulid caption_fakeImage model')
    fakeCaption_model1=caption_fake2_ResNet50(0,classes)
    fakeCaption_featuer1=fakeCaption_model1.output
    in1=fakeCaption_model1.input
    fakeCaption_model2=caption_fake2_ResNet50(1,classes)
    fakeCaption_featuer2=fakeCaption_model2.output
    in2=fakeCaption_model2.input
    fakeCaption_model3=caption_fake2_ResNet50(2,classes)
    fakeCaption_featuer3=fakeCaption_model3.output
    in3=fakeCaption_model3.input
    fakeCaption_model4=caption_fake2_ResNet50(3,classes)
    fakeCaption_featuer4=fakeCaption_model4.output
    in4=fakeCaption_model4.input
    fakeCaption_model5=caption_fake2_ResNet50(4,classes)
    fakeCaption_featuer5=fakeCaption_model5.output
    in5=fakeCaption_model5.input
    fakeCaption_model6=caption_fake2_ResNet50(5,classes)
    fakeCaption_featuer6=fakeCaption_model6.output
    in6=fakeCaption_model6.input
    fakeCaption_model7=caption_fake2_ResNet50(6,classes)
    fakeCaption_featuer7=fakeCaption_model7.output
    in7=fakeCaption_model7.input
    fakeCaption_model8=caption_fake2_ResNet50(7,classes)
    fakeCaption_featuer8=fakeCaption_model8.output
    in8=fakeCaption_model8.input
    fakeCaption_model9=caption_fake2_ResNet50(8,classes)
    fakeCaption_featuer9=fakeCaption_model9.output
    in9=fakeCaption_model9.input
    fakeCaption_model10=caption_fake2_ResNet50(9,classes)
    fakeCaption_featuer10=fakeCaption_model10.output
    in10=fakeCaption_model10.input
    outs =Average()([fakeCaption_featuer1, fakeCaption_featuer2,fakeCaption_featuer3,fakeCaption_featuer4, fakeCaption_featuer5,fakeCaption_featuer6,fakeCaption_featuer7, fakeCaption_featuer8,fakeCaption_featuer9, fakeCaption_featuer10])
    model = Model(inputs= [in1,in2,in3,in4,in5,in6,in7,in8,in9,in10], outputs=outs,name='Muit_fake3_Feature_model')
    return model
    
def finnal_muilt2Feature_model(classes):
    print('bulid true image model')
    true_image_model = true_ResNet50(classes)
    true_image_feature=true_image_model.output
    in0=true_image_model.input
    print('build Muit_fake3_Feature_model')
    mult_fake2_caption_model=Muit_fake2_Feature_model(classes)
    mult_fake2_caption_feature=mult_fake2_caption_model.output
    in1=mult_fake3_caption_model.input
    merged=Add()([true_image_feature,mult_fake3_caption_feature])
    Flat= Flatten()(merged)
    Dor=Dropout(0.5)(Flat)
    fc = Dense(512)(Dor)
    bnn = BatchNormalization(name='bn2')(fc)
    Den=Dense(classes, activation='softmax')(bnn)
    m_model=Model(inputs=[in0,in1], outputs=Den)
    #plot_model(s_model, to_file='true-fake-restnet50-fine-20181104.png',show_shapes=True)
    m_model.summary()
    return m_model

def fake3_ResNet50(index,classes):
    t1=index+0
    fake_base_model1=fake_ResNet50_base(t1,input_shape = (224, 224, 3),classes=200)
    temp_feature1=fake_base_model1.output
    in1=fake_base_model1.input
    t2=index+1
    fake_base_model2=fake_ResNet50_base(t2,input_shape = (224, 224, 3),classes=200)
    temp_feature2=fake_base_model2.output
    in2=fake_base_model2.input
    t3=index+2
    fake_base_model3=fake_ResNet50_base(t3,input_shape = (224, 224, 3),classes=200)
    temp_feature3=fake_base_model3.output
    in3=fake_base_model3.input
    outs =Average()([temp_feature1, temp_feature2,temp_feature3])
    model = Model(inputs= [in1,in2,in3], outputs=outs,name='fake-ResNet50')
    return model
    
def caption_fake3_ResNet50(index,classes):
    print('merge the fake images')
    mult_fake_model=fake3_ResNet50(classes)
    mult_fakeimage_feature=mult_fake_model.output
    in1=mult_fake_model.input
    nIndex=index*3
    caption_model=text_cnnmodel_base(nIndex,classes)
    caption_feature=caption_model.output
    in2=caption_model.input
    merged=Add()([mult_fakeimage_feature,caption_feature])
    Flat= Flatten()(merged)
    Dor=Dropout(0.1)(Flat)
    fc = Dense(2048)(Dor)
    model=Model(inputs=[in1,in2],outputs=fc,name='caption_fake3_ResNet50')
    return model

    
def Muit_fake3_Feature_model(classes):
    print('bulid caption_fakeImage model')
    fakeCaption_model1=caption_fake3_ResNet50(0,classes)
    fakeCaption_featuer1=fakeCaption_model1.output
    in1=fakeCaption_model1.input
    fakeCaption_model2=caption_fake3_ResNet50(1,classes)
    fakeCaption_featuer2=fakeCaption_model2.output
    in2=fakeCaption_model2.input
    fakeCaption_model3=caption_fake3_ResNet50(2,classes)
    fakeCaption_featuer3=fakeCaption_model3.output
    in3=fakeCaption_model3.input
    fakeCaption_model4=caption_fake3_ResNet50(3,classes)
    fakeCaption_featuer4=fakeCaption_model4.output
    in4=fakeCaption_model4.input
    fakeCaption_model5=caption_fake3_ResNet50(4,classes)
    fakeCaption_featuer5=fakeCaption_model5.output
    in5=fakeCaption_model5.input
    fakeCaption_model6=caption_fake3_ResNet50(5,classes)
    fakeCaption_featuer6=fakeCaption_model6.output
    in6=fakeCaption_model6.input
    fakeCaption_model7=caption_fake3_ResNet50(6,classes)
    fakeCaption_featuer7=fakeCaption_model7.output
    in7=fakeCaption_model7.input
    fakeCaption_model8=caption_fake3_ResNet50(7,classes)
    fakeCaption_featuer8=fakeCaption_model8.output
    in8=fakeCaption_model8.input
    fakeCaption_model9=caption_fake3_ResNet50(8,classes)
    fakeCaption_featuer9=fakeCaption_model9.output
    in9=fakeCaption_model9.input
    fakeCaption_model10=caption_fake3_ResNet50(9,classes)
    fakeCaption_featuer10=fakeCaption_model10.output
    in10=fakeCaption_model10.input
    outs =Average()([fakeCaption_featuer1, fakeCaption_featuer2,fakeCaption_featuer3,fakeCaption_featuer4, fakeCaption_featuer5,fakeCaption_featuer6,fakeCaption_featuer7, fakeCaption_featuer8,fakeCaption_featuer9, fakeCaption_featuer10])
    model = Model(inputs= [in1,in2,in3,in4,in5,in6,in7,in8,in9,in10], outputs=outs,name='Muit_fake3_Feature_model')
    return model
    
def finnal_muilt3Feature_model(classes):
    print('bulid true image model')
    true_image_model = true_ResNet50(classes)
    true_image_feature=true_image_model.output
    in0=true_image_model.input
    print('build Muit_fake3_Feature_model')
    mult_fake3_caption_model=Muit_fake3_Feature_model(classes)
    mult_fake3_caption_feature=mult_fake3_caption_model.output
    in1=mult_fake3_caption_model.input
    merged=Add()([true_image_feature,mult_fake3_caption_feature])
    Flat= Flatten()(merged)
    Dor=Dropout(0.5)(Flat)
    fc = Dense(512)(Dor)
    bnn = BatchNormalization(name='bn2')(fc)
    Den=Dense(classes, activation='softmax')(bnn)
    m_model=Model(inputs=[in0,in1], outputs=Den)
    #plot_model(s_model, to_file='true-fake-restnet50-fine-20181104.png',show_shapes=True)
    m_model.summary()
    return m_model

def fake5_ResNet50(classes):
    fake_base_model1=fake_ResNet50_base55(0,input_shape = (224, 224, 3),classes=200)
    temp_feature1=fake_base_model1.output
    in1=fake_base_model1.input
    fake_base_model2=fake_ResNet50_base55(1,input_shape = (224, 224, 3),classes=200)
    temp_feature2=fake_base_model2.output
    in2=fake_base_model2.input
    fake_base_model3=fake_ResNet50_base55(2,input_shape = (224, 224, 3),classes=200)
    temp_feature3=fake_base_model3.output
    in3=fake_base_model3.input
    fake_base_model4=fake_ResNet50_base55(3,input_shape = (224, 224, 3),classes=200)
    temp_feature4=fake_base_model4.output
    in4=fake_base_model4.input
    fake_base_model5=fake_ResNet50_base55(4,input_shape = (224, 224, 3),classes=200)
    temp_feature5=fake_base_model5.output
    in5=fake_base_model5.input
    #ins =Add()([inputall[0], inputall[1],inputall[2], inputall[3],inputall[4], inputall[5],inputall[6], inputall[7],inputall[8], inputall[9]])
    outs =Average()([temp_feature1, temp_feature2,temp_feature3, temp_feature4,temp_feature5])
    model = Model(inputs= [in1,in2,in3,in4,in5], outputs=outs,name='fake-ResNet50')
    return model
    
def caption_fake5_ResNet50(index,classes):
    print('merge the fake images')
    mult_fake_model=fake5_ResNet50(classes)
    mult_fakeimage_feature=mult_fake_model.output
    in1=mult_fake_model.input
    caption_model=text_cnnmodel_base(index,classes)
    caption_feature=caption_model.output
    in2=caption_model.input
    merged=Add()([mult_fakeimage_feature,caption_feature])
    Flat= Flatten()(merged)
    Dor=Dropout(0.1)(Flat)
    fc = Dense(2048)(Dor)
    model=Model(inputs=[in1,in2],outputs=fc,name='caption_fake5_ResNet50')
    return model
    
def Muit_fake5_Feature_model(classes):
    print('bulid caption_fakeImage model')
    fakeCaption_model1=caption_fake5_ResNet50(0,classes)
    fakeCaption_featuer1=fakeCaption_model1.output
    in1=fakeCaption_model1.input
    fakeCaption_model2=caption_fake5_ResNet50(1,classes)
    fakeCaption_featuer2=fakeCaption_model2.output
    in2=fakeCaption_model2.input
    fakeCaption_model3=caption_fake5_ResNet50(2,classes)
    fakeCaption_featuer3=fakeCaption_model3.output
    in3=fakeCaption_model3.input
    fakeCaption_model4=caption_fake5_ResNet50(3,classes)
    fakeCaption_featuer4=fakeCaption_model4.output
    in4=fakeCaption_model4.input
    fakeCaption_model5=caption_fake5_ResNet50(4,classes)
    fakeCaption_featuer5=fakeCaption_model5.output
    in5=fakeCaption_model5.input
    fakeCaption_model6=caption_fake5_ResNet50(5,classes)
    fakeCaption_featuer6=fakeCaption_model6.output
    in6=fakeCaption_model6.input
    fakeCaption_model7=caption_fake5_ResNet50(6,classes)
    fakeCaption_featuer7=fakeCaption_model7.output
    in7=fakeCaption_model7.input
    fakeCaption_model8=caption_fake5_ResNet50(7,classes)
    fakeCaption_featuer8=fakeCaption_model8.output
    in8=fakeCaption_model8.input
    fakeCaption_model9=caption_fake5_ResNet50(8,classes)
    fakeCaption_featuer9=fakeCaption_model9.output
    in9=fakeCaption_model9.input
    fakeCaption_model10=caption_fake5_ResNet50(9,classes)
    fakeCaption_featuer10=fakeCaption_model10.output
    in10=fakeCaption_model10.input
    outs =Average()([fakeCaption_featuer1, fakeCaption_featuer2,fakeCaption_featuer3,fakeCaption_featuer4, fakeCaption_featuer5,fakeCaption_featuer6,fakeCaption_featuer7, fakeCaption_featuer8,fakeCaption_featuer9, fakeCaption_featuer10])
    model = Model(inputs= [in1,in2,in3,in4,in5,in6,in7,in8,in9,in10], outputs=outs,name='Muit_fake3_Feature_model')
    return model
    
def finnal_muilt5Feature_model(classes):
    print('bulid true image model')
    true_image_model = true_ResNet50(classes)
    true_image_feature=true_image_model.output
    in0=true_image_model.input
    print('build Muit_fake5_Feature_model')
    mult_fake5_caption_model=Muit_fake5_Feature_model(classes)
    mult_fake5_caption_feature=mult_fake5_caption_model.output
    in1=mult_fake5_caption_model.input
    merged=Add()([true_image_feature,mult_fake3_caption_feature])
    Flat= Flatten()(merged)
    Dor=Dropout(0.5)(Flat)
    fc = Dense(512)(Dor)
    bnn = BatchNormalization(name='bn2')(fc)
    Den=Dense(classes, activation='softmax')(bnn)
    m_model=Model(inputs=[in0,in1], outputs=Den)
    #plot_model(s_model, to_file='true-fake-restnet50-fine-20181104.png',show_shapes=True)
    m_model.summary()
    return m_model

###======================== PREPARE DATA ====================================###
#build myself data generator
#imgInfo_file_path: pickle (file name with path)
#classInfo_file_path: pickle( file class)
#image_direction: true image path
#fackimage_direction: fack image path
#txt_direction: text path
#image_size: input image size of model
#num: the value of K(StackMGAN++)
tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',lower=True,split=" ")
Alltxt=open('../data/flower-dataset/vacab.txt','r')
Alltext=Alltxt.read()
tokenizer.fit_on_texts(Alltext)
vocab = tokenizer.word_index

def data_generator_4(imgInfo_file_path,classInfo_file_path,image_direction,txt_direction,fackimage_direction,image_size,BATCHSIZE,num): 
    testfilenames = open(imgInfo_file_path,'rb')
    rmesf= pickle.load(testfilenames)
    testfilenames = open(classInfo_file_path,'rb')
    rmesc= pickle.load(testfilenames)
    txt1=[]
    txt2=[]
    txt3=[]
    txt4=[]
    txt5=[]
    txt6=[]
    txt7=[]
    txt8=[]
    txt9=[]
    txt10=[]
    fake1=[]
    fake2=[]
    fake3=[]
    fake4=[]
    fake5=[]
    fake6=[]
    fake7=[]
    fake8=[]
    fake9=[]
    fake10=[]
    
    images=[]
    labels=[]
    imagefile=[]
    textfile=[]
    iclass=[]
    imagename=[]
    num_of_examples=len(rmesf)
    for i in range(len(rmesf)):
        temp=rmesf[i]
        tempimagename=image_direction+temp
        #print(tempimagename)
        if os.path.isfile(tempimagename)==False:
            print('error! no such ture file: %s' %tempimagename)
            continue
        else:
            #class_001/image_00000.txt
            img=Image.open(tempimagename)
            img = img.resize((image_size[0], image_size[1]),Image.ANTIALIAS)
            img=np.array(img)
            ttemp=rmesc[i]
            templable=int(ttemp)
            templable1=int(ttemp)-1
            templable='%03d' % templable
            #print(templable)
            ftemp=temp[:-4] 
            txtPath=txt_direction+'class_'+templable+'/'+ftemp+'.txt'
            #print(txtPath)
            if os.path.isfile(txtPath)==False:
                print('error! no such caption file: %s' %txtPath)
                continue
            else:
                temptxt=[]
                tempfake=[]
                tmask=False
                mm=0
                for line in open(txtPath,'r'):
                    fftemp=temp[:-4]
                    fakefname=fackimage_direction+fftemp+'_sentence'+str(mm)+'.png'
                    mm=mm+1
                    #print(fakefname)
                    if os.path.isfile(fakefname)==False:
                        print('error! no such fake image file: %s' %fakefname)
                        tmask=False
                        continue
                    else:
                        ftimg=Image.open(fakefname)
                        ftimg = ftimg.resize((image_size[0], image_size[1]),Image.ANTIALIAS)
                        ftimg=np.array(ftimg)
                        temptxt.append(line)
                        tempfake.append(ftimg)
                        tmask=True
                if tmask==True:                 
                    txt1.append(temptxt[0])
                    txt2.append(temptxt[1])
                    txt3.append(temptxt[2])
                    txt4.append(temptxt[3])
                    txt5.append(temptxt[4])
                    txt6.append(temptxt[5])
                    txt7.append(temptxt[6])
                    txt8.append(temptxt[7])
                    txt9.append(temptxt[8])
                    txt10.append(temptxt[9])
                    fake1.append(tempfake[0])
                    fake2.append(tempfake[1])
                    fake3.append(tempfake[2])
                    fake4.append(tempfake[3])
                    fake5.append(tempfake[4])
                    fake6.append(tempfake[5])
                    fake7.append(tempfake[6])
                    fake8.append(tempfake[7])
                    fake9.append(tempfake[8])
                    fake10.append(tempfake[9])
                    labels.append(templable1)
                    images.append(img)
    capt_train_word_ids1 = tokenizer.texts_to_sequences(txt1)
    txt1 = pad_sequences(capt_train_word_ids1, maxlen=64)
    capt_train_word_ids2 = tokenizer.texts_to_sequences(txt2)
    txt2 = pad_sequences(capt_train_word_ids2, maxlen=64)
    capt_train_word_ids3 = tokenizer.texts_to_sequences(txt3)
    txt3 = pad_sequences(capt_train_word_ids3, maxlen=64)
    capt_train_word_ids4 = tokenizer.texts_to_sequences(txt4)
    txt4 = pad_sequences(capt_train_word_ids4, maxlen=64)
    capt_train_word_ids5 = tokenizer.texts_to_sequences(txt5)
    txt5 = pad_sequences(capt_train_word_ids5, maxlen=64)
    capt_train_word_ids6 = tokenizer.texts_to_sequences(txt6)
    txt6 = pad_sequences(capt_train_word_ids6, maxlen=64)
    capt_train_word_ids7 = tokenizer.texts_to_sequences(txt7)
    txt7 = pad_sequences(capt_train_word_ids7, maxlen=64)
    capt_train_word_ids8 = tokenizer.texts_to_sequences(txt8)
    txt8 = pad_sequences(capt_train_word_ids8, maxlen=64)
    capt_train_word_ids9 = tokenizer.texts_to_sequences(txt9)
    txt9 = pad_sequences(capt_train_word_ids9, maxlen=64)
    capt_train_word_ids10 = tokenizer.texts_to_sequences(txt10)
    txt10 = pad_sequences(capt_train_word_ids10, maxlen=64)
    images=np.array(images)
    fake1=np.array(fake1)
    fake2=np.array(fake2)
    fake3=np.array(fake3)
    fake4=np.array(fake4)
    fake5=np.array(fake5)
    fake6=np.array(fake6)
    fake7=np.array(fake7)
    fake8=np.array(fake8)
    fake9=np.array(fake9)
    fake10=np.array(fake10)
    labels = to_categorical(labels, num)
    labels=np.array(labels)
   # gc.collect()

    return images,fake1,fake2,fake3,fake4,fake5,fake6,fake7,fake8,fake9,fake10,txt1,txt2,txt3,txt4,txt5,txt6,txt7,txt8,txt9,txt10,labels
    

num=200


fake_direction='birds-dataset/birds-dataset/single_samples/fake/'
print('read birds test data set')
imgInfo_file_path='birds-dataset/birds-dataset/Test/filenames.pickle'
classInfo_file_path='birds-dataset/birds-dataset/Test/class_infor.pickle'
image_direction='birds-dataset/birds-dataset/images/'
txt_direction='birds-dataset/birds-dataset/text/'


width,height=224,224
image_size=(width,height,3)
BATCHSIZE=64
num=200
#val_gen=data_generator_3(imgInfo_file_path,classInfo_file_path,image_direction,fackimage_direction,txt_direction,image_size,BATCHSIZE,num)
val_images,val_fake1,val_fake2,val_fake3,val_fake4,val_fake5,val_fake6,val_fake7,val_fake8,val_fake9,val_fake10,val_txt1,val_txt2,val_txt3,val_txt4,val_txt5,val_txt6,val_txt7,val_txt8,val_txt9,val_txt10,val_labels=data_generator_4(imgInfo_file_path,classInfo_file_path,image_direction,txt_direction,fake_direction,image_size,BATCHSIZE,num)

print(len(val_images))

import random
LLL=len(val_images)
randnum = random.randint(0,LLL)
random.seed(randnum)
random.shuffle(val_images)
random.seed(randnum)
random.shuffle(val_fake1)
random.seed(randnum)
random.shuffle(val_fake2)
random.seed(randnum)
random.shuffle(val_fake3)
random.seed(randnum)
random.shuffle(val_fake4)
random.seed(randnum)
random.shuffle(val_fake5)
random.seed(randnum)
random.shuffle(val_fake6)
random.seed(randnum)
random.shuffle(val_fake7)
random.seed(randnum)
random.shuffle(val_fake8)
random.seed(randnum)
random.shuffle(val_fake9)
random.seed(randnum)
random.shuffle(val_fake10)
random.seed(randnum)
random.shuffle(val_txt1)
random.seed(randnum)
random.shuffle(val_txt2)
random.seed(randnum)
random.shuffle(val_txt3)
random.seed(randnum)
random.shuffle(val_txt4)
random.seed(randnum)
random.shuffle(val_txt5)
random.seed(randnum)
random.shuffle(val_txt6)
random.seed(randnum)
random.shuffle(val_txt7)
random.seed(randnum)
random.shuffle(val_txt8)
random.seed(randnum)
random.shuffle(val_txt9)
random.seed(randnum)
random.shuffle(val_txt10)
random.seed(randnum)
random.shuffle(val_labels)


    
print('read flower train data set')
imgInfo_file_path='birds-dataset/birds-dataset/Train/filenames.pickle'
classInfo_file_path='birds-dataset/birds-dataset/Train/class_infor.pickle'
image_direction='birds-dataset/birds-dataset/images/'
txt_direction='birds-dataset/birds-dataset/text/'

width,height=224,224
image_size=(width,height,3)
BATCHSIZE=64
num=200
#train_gen=data_generator_3(imgInfo_file_path,classInfo_file_path,image_direction,fackimage_direction,txt_direction,image_size,BATCHSIZE,num)
train_images,train_fake1,train_fake2,train_fake3,train_fake4,train_fake5,train_fake6,train_fake7,train_fake8,train_fake9,train_fake10,train_txt1,train_txt2,train_txt3,train_txt4,train_txt5,train_txt6,train_txt7,train_txt8,train_txt9,train_txt10,train_labels=data_generator_4(imgInfo_file_path,classInfo_file_path,image_direction,txt_direction,fake_direction,image_size,BATCHSIZE,num)

train_images=np.concatenate((train_images,val_images[1234:]), axis=0)
train_fake1=np.concatenate((train_fake1,val_fake1[1234:]), axis=0)
train_fake2=np.concatenate((train_fake2,val_fake2[1234:]), axis=0)
train_fake3=np.concatenate((train_fake3,val_fake3[1234:]), axis=0)
train_fake4=np.concatenate((train_fake4,val_fake4[1234:]), axis=0)
train_fake5=np.concatenate((train_fake5,val_fake5[1234:]), axis=0)
train_fake6=np.concatenate((train_fake6,val_fake6[1234:]), axis=0)
train_fake7=np.concatenate((train_fake7,val_fake7[1234:]), axis=0)
train_fake8=np.concatenate((train_fake8,val_fake8[1234:]), axis=0)
train_fake9=np.concatenate((train_fake9,val_fake9[1234:]), axis=0)
train_fake10=np.concatenate((train_fake10,val_fake10[1234:]), axis=0)
train_txt1=np.concatenate((train_txt1,val_txt1[1234:]), axis=0)
train_txt2=np.concatenate((train_txt2,val_txt2[1234:]), axis=0)
train_txt3=np.concatenate((train_txt3,val_txt3[1234:]), axis=0)
train_txt4=np.concatenate((train_txt4,val_txt4[1234:]), axis=0)
train_txt5=np.concatenate((train_txt5,val_txt5[1234:]), axis=0)
train_txt6=np.concatenate((train_txt6,val_txt6[1234:]), axis=0)
train_txt7=np.concatenate((train_txt7,val_txt7[1234:]), axis=0)
train_txt8=np.concatenate((train_txt8,val_txt8[1234:]), axis=0)
train_txt9=np.concatenate((train_txt9,val_txt9[1234:]), axis=0)
train_txt10=np.concatenate((train_txt10,val_txt10[1234:]), axis=0)
train_labels=np.concatenate((train_labels,val_labels[1234:]), axis=0)

val_images=val_images[:1234]
val_fake1=val_fake1[:1234]
val_fake2=val_fake2[:1234]
val_fake3=val_fake3[:1234]
val_fake4=val_fake4[:1234]
val_fake5=val_fake5[:1234]
val_fake6=val_fake6[:1234]
val_fake7=val_fake7[:1234]
val_fake8=val_fake8[:1234]
val_fake9=val_fake9[:1234]
val_fake10=val_fake10[:1234]
val_txt1=val_txt1[:1234]
val_txt2=val_txt2[:1234]
val_txt3=val_txt3[:1234]
val_txt4=val_txt4[:1234]
val_txt5=val_txt5[:1234]
val_txt6=val_txt6[:1234]
val_txt7=val_txt7[:1234]
val_txt8=val_txt8[:1234]
val_txt9=val_txt9[:1234]
val_txt10=val_txt10[:1234]
val_labels=val_labels[:1234]

print(len(val_labels))


epochs=50
fp_model=true_text_ResNet50_2(num)
opt = Adam(lr=0.0001, decay=0.00001)
#fp_model=Our_ResNet50(classes =102)
fp_model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
#fp_model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

from keras.callbacks import ReduceLROnPlateau,EarlyStopping
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, mode='auto',  cooldown=0, min_lr=0)
esstoping=EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')



out22=fp_model.fit([train_images,train_txt1,train_txt2,train_txt3,train_txt4,train_txt5,train_txt6,train_txt7,train_txt8,train_txt9,train_txt10], train_labels, 
				   batch_size=32,
				   epochs=50,
				   shuffle=True,
				   validation_data=([val_images,val_txt1,val_txt2,val_txt3,val_txt4,val_txt5,val_txt6,val_txt7,val_txt8,val_txt9,val_txt10],val_labels),
				   callbacks=[reduce_lr])
				   
with open('birds-true-text-20190320_v3_1.txt','w') as f2:
    f2.write(str(out22.history))
fp_model.save('birds-true-text-20190320_v3_1.txt.h5',include_optimizer=True)

