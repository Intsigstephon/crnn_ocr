#coding=utf-8
#Author: stephon
#Time: 2018.11.11

"""
use to test one image, or handle a set of image
of arbitray size
"""

from keras.models import Model
import numpy as np
from PIL import Image
import keras.backend  as K
import os
from keras.models import load_model
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
import cv2
import time
from glob import glob
from model import get_model
from preproc import preprocess2

charset = "0123456789 "               #15
checkpoint = "./model/basemodel11.h5"     #15: keras2.1.5 train;   86: keras2.0.8 train
imgpath = "./bad/2517.jpg"

def init(gpu_index, gpu_memory_fraction, modelPath):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    config = tf.ConfigProto()
    if gpu_memory_fraction < 0.001:
        config.gpu_options.allow_growth=True
    else:
        config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
    set_session(tf.Session(config=config))

    basemodel = load_model(modelPath)

    # model, basemodel = get_model(32, len(charset) + 1)   #
    # basemodel.load_weights(modelPath)

    return basemodel

def decode(pred):
    batch_size = pred.shape[0]   
    length = pred.shape[1]    
    t = pred.argmax(axis=2)   
    char_lists = []
    n = len(charset)  #n=15

    #handle batch
    for i in range(batch_size):
        char_list = ''
        for ii in range(length):
            c = t[i]   
            if c[ii] != n and (not (ii > 0 and c[ii - 1] == c[ii])):
               char_list = char_list + charset[c[ii]]  
        char_lists.append(char_list)
    return char_lists

def predict(im, basemodel):
    """
    batch predict: if one model used, should add the comment part when decode.
    """
    out = basemodel.predict(im)
    y_pred = out[:,2:,:]   

    # numlabel = [26,93,25,94,632,631,933,29,27,1109,5530,5531]
    # a = np.zeros((5532))  
    # a[numlabel] = 1
    # y_pred[0] = np.multiply(a, y_pred[0])   
    # y_pred[1] = np.multiply(a, y_pred[1])

    #---------------------------------------
    out = decode(y_pred)  

    return out

def forward(imgs, model): 
    """
    given a set of images, do parallel predict on batch
    """   
    Max = 0
    for im in imgs:
        if im is None:
            return

        h = im.shape[0]
        w = im.shape[1]
        w = int(float(w)/(float(h)/32.0))
        if w > Max:
           Max = w
    
    im_group = None
    for im in imgs:
        h = im.shape[0]
        w = im.shape[1]
        w = int(float(w)/(float(h) / 32.0))
        im = cv2.resize(im,(w,32))
        im = cv2.copyMakeBorder(im,0,0,0,Max-w,cv2.BORDER_CONSTANT,value=250)
        im = im.astype(np.float32)
        im = ((im/255.0)-0.5) * 2
        X  = im.reshape((32, Max, 1))
        X = np.array([X])
        if im_group is None:
           im_group = X
           continue

        im_group = np.concatenate((im_group, X), axis=0) 

    #ocr
    result = predict(im_group, model)       #handle parallel 

    return result

if __name__ == '__main__':
    #load basemodel
    t = time.time()
    basemodel = init(0, 0.3, checkpoint)     #last 1.2s   
    print("init time is: {}".format(time.time() - t))

    #prepare data
    imagedir = "./bad"
    imgs = glob(imagedir + "/*.jpeg")
    #imgs.extend(glob(imagedir + "/*.jpg"))

    #represent of batch
    batch_size = len(imgs)
    imgs = imgs[0: batch_size]
    print(len(imgs))
    for img in imgs:
        print(img)

    #resize and concatenate
    aa = None
    Max = 0

    #get the max len after resize
    for i in range(batch_size):
        path_1 = imgs[i]
        im = cv2.imread(path_1)
        if im is None:
            exit()

        h = im.shape[0]
        w = im.shape[1]
        w = int(float(w)/(float(h)/32.0))
        if w > Max:
           Max = w
    
    print("Max length is %d" % Max)
    
    #concatenate data
    for i in range(batch_size):
        path_1 = imgs[i]
        im = cv2.imread(path_1)
        h = im.shape[0]
        w = im.shape[1]
        w = int(float(w)/(float(h)/32.0))
        im = cv2.resize(im, (w, 32))
        im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        #im = cv2.equalizeHist(im)
        im = cv2.copyMakeBorder(im,0,0,0,Max-w,cv2.BORDER_CONSTANT,value=250)
        im = preprocess2(im, 2)

        im = im.astype(np.float32)
        im = ((im/255.0)-0.5)*2   #-1~1
        X  = im.reshape((32, Max, 1))   #
        X = np.array([X])
        if aa is None:
           aa = X
           continue

        aa = np.concatenate((aa, X), axis=0)   #last 0.007s
    

    #do ocr
    start = time.time()
    out = basemodel.predict(aa)    #python2: #1: 0.8;  #5:0.85; #10: 1s  #60:1.8s, parallel speed 
    end = time.time()

    # numlabel = [26,93,25,94,632,631,933,29,27,1109,5201,5530,5531]
    # numlabel.extend([1, 466])       
    # a = np.zeros((5532))
    # a[numlabel] = 1
    # for i in range(len(out)):
    #     out[i] = np.multiply(a, out[i])

    y_pred = out[:,2:,:]    
    result = decode(y_pred)       #handle parallel   #last 0.004s

    #time cost
    print("time cost:" + str(end - start))
    print("-------------------")

    #show result
    
    cv2.imshow("img", aa[0])
    cv2.waitKey(0)
    for i in range(len(result)):
        print(result[i])
        path_t = imgs[i]
        im_t = cv2.imread(path_t, 1)
        #cv2.imshow("img", im_t)
        cv2.imshow("img", aa[i])
        cv2.waitKey(0)






