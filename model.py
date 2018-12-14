# -*- coding: utf-8 -*-
# Author: stephon
# Time: 2018.11.9

from keras.layers import Input,Conv2D,MaxPooling2D,ZeroPadding2D
from keras.layers import Flatten,BatchNormalization,Permute,TimeDistributed,Dense,Bidirectional,GRU,Dropout
from keras.models import Model

import numpy as np
from PIL import Image
import keras.backend  as K
import tensorflow as tf

#import keys
import os
from keras.models import load_model
from keras.layers import Lambda
from keras.optimizers import SGD

"""
Define the Model and CTC function 
"""
def ctc_lambda_func(args):
    """
    given label and y_pred, return loss value.
    """
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, 2:, :]   #?? why ignore col 0 and col 1 
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)    

def get_model(height, nclass):
    """
    height: 32; nclass is the the number of alphas
    height means input must be height = 32. Actually not
    """
    """
    some idea: 
    1. first: height must reduce 1
    2. 
    """
    rnnunit  = 256

    input = Input(shape=(height,None,1), name='the_input')
    m = Conv2D(64,kernel_size=(3,3),activation='relu',padding='same',name='conv1')(input)　　　#conv1
    m = MaxPooling2D(pool_size=(2,2),strides=(2,2),name='pool1')(m)                           #pool1
    print(m)

    m = Conv2D(128,kernel_size=(3,3),activation='relu',padding='same',name='conv2')(m)        #conv2
    m = MaxPooling2D(pool_size=(2,2),strides=(2,2),name='pool2')(m)                           #pool2
    print(m)

    m = Conv2D(256,kernel_size=(3,3),activation='relu',padding='same',name='conv3')(m)        #conv3
    m = Conv2D(256,kernel_size=(3,3),activation='relu',padding='same',name='conv4')(m)        #conv4
    print(m)

    m = ZeroPadding2D(padding=(0,1))(m)
    print(m)

    m = MaxPooling2D(pool_size=(2,2),strides=(2,1),padding='valid',name='pool3')(m)           #pool3
    print(m)

    # Also do conv
    m = Conv2D(512,kernel_size=(3,3),activation='relu',padding='same',name='conv5')(m)        #conv5
    m = BatchNormalization(axis=1)(m)      
    m = Conv2D(512,kernel_size=(3,3),activation='relu',padding='same',name='conv6')(m)        #conv6
    m = BatchNormalization(axis=1)(m)
    m = ZeroPadding2D(padding=(0,1))(m)
    m = MaxPooling2D(pool_size=(2,2),strides=(2,1),padding='valid',name='pool4')(m)           #pool4
    m = Conv2D(512,kernel_size=(2,2),activation='relu',padding='valid',name='conv7')(m)       #conv7

    #why do permute
    m = Permute((2,1,3),name='permute')(m)
    m = TimeDistributed(Flatten(),name='timedistrib')(m)

    #rnn model
    m = Bidirectional(GRU(rnnunit,return_sequences=True),name='blstm1')(m)
    m = Dense(rnnunit,name='blstm1_out',activation='linear')(m)
    m = Dropout(0.2)(m)

    m = Bidirectional(GRU(rnnunit,return_sequences=True),name='blstm2')(m)  #rununit is the hidden layer vector length
    y_pred = Dense(nclass,name='blstm2_out',activation='softmax')(m)  
    basemodel = Model(inputs=input,outputs=y_pred)    

    #define the labels
    labels = Input(name='the_labels', shape=[None,], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
    model = Model(inputs=[input, labels, input_length, label_length], outputs=[loss_out])

    sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)     #loss function？
    model.summary()

    return model, basemodel

if __name__=="__main__":
    """
    load existing model
    """
    modelPath = os.path.join(os.getcwd(), "model/basemodel53.h5")
    if os.path.exists(modelPath):
        basemodel = load_model(modelPath)
    print(basemodel)
    print("\n")

    #define a model
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    height = 32
    alphas = "0123456789"
    get_model(height, len(alphas))

    #tips: 
    #1. the input is gray image
    #2. Two Bidirectional GRU

