#coding=utf-8
#Author: stephon
#Time: 11.15

import os
import cv2
import codecs
from test import init, predict
import time
import Levenshtein
import numpy as np
from preproc import preprocess2

debug = False
t =  time.time()
checkpoint = "./model/basemodel48.h5"                  #model: 36; 48的比较好, 达到98.22%; 去除一些倾斜比较严重的，准确率应该可以上到99%;
basemodel = init(0, 0.2, checkpoint)                   #last 1.2s   
print("init time is: {}".format(time.time() - t))      #按需分配，则模型只需要占用1172-1095 = 77M ~ 80M;  正常运行，占用显存大小：2571 - 1095 = 1471M左右； 
#exit()

test_file = "./truth_test.txt"
all_num = 0
right_num = 0
item_right_num = 0
wrong_images = []
w = codecs.open("result.txt", 'w', 'utf-8')

with codecs.open(test_file, 'r', 'utf-8') as r:
    alllines = r.readlines()
    for oneline in alllines:
        imgname, label = oneline.split(',')
        label = label.strip()
        #print(imgname)

        #Actually, it has been resized to 32 * 350
        im = cv2.imread(imgname, 0)
        im = preprocess2(im, 2)
        im = im.astype(np.float32)
        im = ((im/255.0)-0.5) * 2
        im = im.reshape((32, 350, 1))
        X = np.array([im])
        out = predict(X, basemodel)
        out = ''.join([unicode(a) for a in out])

        #print(out)
        #print(label)

        if debug:
            cv2.imshow("src", img)
            cv2.waitKey(0)
        
        #print(type(label))
        #print(type(out))
        #exit()

        right_ratio = 1 -  Levenshtein.distance(out, label) / (len(label) + 0.0)
        if right_ratio > 0.99:
            item_right_num +=1
        else:                                #handle for wrong
            wrong_images.append(imgname)
            w.write(imgname + ':' + out + '\n')
            img_temp = cv2.imread(imgname, 1)
            cv2.imwrite(os.path.join("./bad",  os.path.basename(imgname)), img_temp)

        right_num += right_ratio
        all_num +=1
        #print("{}".format(right_ratio))

result = right_num / all_num
item_accu = item_right_num / (all_num + 0.0)
print("total right ratio is: {}".format(result))
print("item  right ratio is: {}".format(item_accu))

#show wrong result
for aa in wrong_images:
    print(aa)