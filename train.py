#coding: utf-8
#Author: stephon
#Time: 2018.11.09

from glob import glob
import tensorflow as tf
import keras.callbacks
import random
import numpy as np
import codecs
from model import get_model
from PIL import Image
import keras.backend  as K
import os
from keras.models import Model
import codecs
from keras.backend.tensorflow_backend import set_session

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

#first, charset should add space: so, length is 15
charset = "0123456789 "         #15
charset_len = len(charset)      #15

lines = []
sd = codecs.open("truth.txt", "r", "utf-8") 
line = sd.readlines()
lines.extend(line)
sd.close()

sd = codecs.open("truth_real.txt", "r", "utf-8") 
line = sd.readlines()
lines.extend(line)
sd.close()

sd = codecs.open("truth_real2.txt", "r", "utf-8") 
line = sd.readlines()
lines.extend(line)
sd.close()

lines = [dd.strip() for dd in lines]
random.shuffle(lines)
imgpaths = []
imglabels = []
imgpaths = [dd.split(",")[0] for dd in lines]
imglabels = [dd.split(",")[1] for dd in lines]

class TrainDataGenerator(keras.callbacks.Callback):
    """
    Use this class to generate Train data
    """
    def __init__(self, batch_size, img_h, img_w, downsample_factor, val_split_ratio, max_string_len):
        self.charset = charset
        self.batch_size = batch_size
        self.img_h = img_h
        self.img_w = img_w
        self.downsample_factor = downsample_factor
        self.blank_label = self.get_output_size() - 1       #define the blank   #15
        self.max_string_len = max_string_len
        self.imgfiles = imgpaths
        self.labelfiles = imglabels

        print("imgs:{}".format(len(self.imgfiles)))
        print("labels:{}".format(len(self.labelfiles)))

        assert len(self.imgfiles) == len(self.labelfiles) and len(self.labelfiles) != 0 and len(self.imgfiles) != 0
        self.gt_num = len(self.imgfiles)

        self.Y_data = None
        self.X_text = None
        self.Y_len = None

        self.build_word_list()

        #separate into train and valid
        self.cur_train_index = 0
        self.val_num = int(self.gt_num * val_split_ratio)
        self.cur_val_index = self.gt_num - self.val_num

    def text_to_labels(self, text):
        labels = []
        for c in text:
            ll = self.charset.find(c)
            if ll == -1:
                ll = charset_len - 1
            labels.append(ll) 
        if labels == []:
           labels = " "
        return labels

    def get_output_size(self):
        return len(self.charset) + 1    #16

    def build_word_list(self):
        self.Y_data = np.ones([self.gt_num, self.max_string_len]) * (charset_len - 1)  # first: all 
        self.X_text = []
        self.Y_len = [0] * self.gt_num

        i = -1
        for file in self.labelfiles:
            i += 1
            gt = file

            self.Y_len[i] = len(gt)
            self.Y_data[i, 0:len(gt)] = self.text_to_labels(gt)
            self.X_text.append(gt)

        self.Y_len = np.expand_dims(np.array(self.Y_len), 1)
        self.shuffle_data()
        print("charset_len: ", len(self.charset))

    def shuffle_data(self):
        data = list(zip(self.Y_len, self.Y_data, self.X_text, self.imgfiles, self.labelfiles))
        random.shuffle(data)
        self.Y_len, self.Y_data, self.X_text, self.imgfiles, self.labelfiles = zip(*data)

    def get_batch(self, index, batch_size, trainAct=True):
        if K.image_data_format() == 'channels_first':
            X_data = np.ones([batch_size, 1, self.img_h, self.img_w])
        else:
            X_data = np.ones([batch_size, self.img_h, self.img_w, 1])

        labels = np.ones([batch_size, self.max_string_len])
        input_length = np.zeros([batch_size, 1])
        label_length = np.zeros([batch_size, 1])
        source_str = []

        for i in range(batch_size):
            # print(index,i)
            if trainAct and i > batch_size - 8:
                # channels last
                X_data[i] = self.get_image(i, blank=True)
                labels[i, 0] = self.blank_label
                input_length[i] = self.img_w // self.downsample_factor - 1
                label_length[i] = 1
                source_str.append('')
            else:
                X_data[i] = self.get_image(index + i)
                labels[i, :] = self.Y_data[index + i]
                input_length[i] = self.img_w // self.downsample_factor - 1
                label_length[i] = self.Y_len[index + i]
                source_str.append(self.X_text[index + i])

        inputs = {
            'the_input': X_data,
            'the_labels': labels,
            'input_length': input_length,
            'label_length': label_length,
            'source_str': source_str
        }
        outputs = {'ctc': np.zeros([batch_size])}
        return inputs, outputs

    def next_train(self):
        while 1:
            ret = self.get_batch(self.cur_train_index, self.batch_size)
            self.cur_train_index += self.batch_size
            yield ret

    def next_val(self):
        while 1:
            ret = self.get_batch(self.cur_val_index, self.batch_size, trainAct=False)
            self.cur_val_index += self.batch_size
            yield ret

    def get_image(self, i, blank=False):
        """
        generate image: goal is img_h * img_w
        """
        blank_img = np.ones((self.img_h, self.img_w, 1), dtype=np.float32) * 255
        if blank:
            return blank_img

        im = Image.open(self.imgfiles[i])
        im = im.convert('L')
        img = ((np.array(im).astype(np.float32) / 255.0) - 0.5) * 2

        #if the img.shape[1] < width, then put the image in blank
        if img.shape[1] < self.img_w:
            blank_img[:, :img.shape[1], 0] = img
            return blank_img
            
        img = img.reshape((self.img_h, self.img_w, 1))  #image resize to 32 * 350
        return img

    def on_epoch_end(self, epoch, logs=None):
        print("epoch end.shuffle data.")
        self.shuffle_data()
        self.cur_train_index = 0
        self.cur_val_index = self.gt_num - self.val_num

def labels_to_text(labels):
    ret = []
    for label in labels:
        if label == len(charset):
            ret.append('-')
            continue
        ret.append(charset[label])
    return u"".join(ret)

def decode_batch(basemodel, word_batch):
    out = basemodel.predict(word_batch)
    y_pred = out[:, 2:, :]
    print("\n---ctc raw predict---")
    for pred in out:
        labels = []
        for s in pred[:, 1:]:
            c = np.argmax(s)
            labels.append(c + 1)
        print(u"labels: {}".format(labels))
        print(u"text  : {}".format(labels_to_text(labels)))

    print(u"---end---")
    out = K.get_value(K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0]) * y_pred.shape[1], )[0][0])[:, :]
    ret = []
    for label in out:
        ret.append(labels_to_text(label.tolist()))
    return ret

class VizCallback(keras.callbacks.Callback):
    def __init__(self, output_dir, data_gen, num_display_words=4, save_model=True):
        self.output_dir = output_dir
        self.data_gen = data_gen
        self.num_display_words = num_display_words
        self.sava_model = save_model
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def on_epoch_end(self, epoch, logs=None):

        basemodel = Model(inputs=self.model.get_layer('the_input').input,
                          outputs=self.model.get_layer('blstm2_out').output)
        if self.sava_model:
            print("\nepoch {} end.save model.".format(epoch))
            basemodel.save(os.path.join(self.output_dir, 'basemodel%02d.h5' % epoch))
            self.model.save_weights(os.path.join(self.output_dir, 'weights%02d.h5' % epoch))

        batch = next(self.data_gen)[0]
        res = decode_batch(basemodel, batch['the_input'][0:self.num_display_words])
        print("\n---CTC decode result---")
        for i in range(self.num_display_words):
            print(u"y_true: {}   y_pred: {}".format(batch['source_str'][i], res[i]))
        print("---end---")

def init(gpu_memory_fraction):
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
    set_session(tf.Session(config=config))

def train(img_w, initial_epoch=0, end_epochs=100, max_string_len=10, batch_size=256, weights_path=None):
    """
    core function of  train
    """

    tb_cb = keras.callbacks.TensorBoard(log_dir="./log/idnum", write_images=1, histogram_freq=1)

    #first init model
    img_h = 32
    model, basemodel = get_model(height = img_h, nclass = charset_len + 1 )

    # Network parameters
    downsample_factor = 4
    val_split_ratio = 0.2
    
    data_gen = TrainDataGenerator(batch_size=batch_size,
                                  img_h=img_h,
                                  img_w=img_w,
                                  downsample_factor=downsample_factor,
                                  val_split_ratio=val_split_ratio,
                                  max_string_len=max_string_len)

    print("model output size:", charset_len + 1 )

    #define save and log info
    viz_cb = VizCallback(output_dir='./model',
                         data_gen=data_gen.next_val(),
                         num_display_words=batch_size,
                         save_model=True)

    #fine tuning.
    if weights_path is not None:
        model.load_weights(weights_path)
    
    model.fit_generator(generator=data_gen.next_train(),
                    steps_per_epoch=(data_gen.gt_num - data_gen.val_num) // batch_size - 11,
                    epochs=end_epochs,
                    validation_data=data_gen.next_val(),
                    validation_steps=data_gen.val_num // batch_size - 11,
                    initial_epoch=initial_epoch,
                    callbacks=[viz_cb, data_gen, tb_cb])

if __name__ == '__main__':
    """
    train: fine tuning
    """
    init(0.7)
    train(img_w=350, initial_epoch=50, end_epochs=51,  max_string_len=16, batch_size=64, weights_path="./model/basemodel49.h5")
