# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 23:36:45 2020

@author: hanxi
"""

from skimage import io, transform
import glob
import os
import tensorflow as tf;  
tf.reset_default_graph()
import numpy as np
import time

w = 100
h = 100
c = 3
NUM_CLASS=10

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 数据集地址
path = './imgs/'
# 模型保存地址
model_path = './scenes_model12_depthXXX.ckpt'

def read_img(path):
    cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]

    imgs = []
    labels = []
    for idx, folder in enumerate(cate):
        print(folder)
        for im in glob.glob(folder + '/*.jpg'):
            #print('reading the images:%s' % (im))
            img = io.imread(im)
            img = transform.resize(img, (w, h,c))
            imgs.append(img)
            labels.append(idx)
    return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)


data, label = read_img(path)
# print(len(data))
# with tf.Session() as sess:
#     print (sess.run(data))
print()