# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 20:20:48 2020

@author: hanxi
"""

#from skimage import io, transform
#import glob
#import os
#import tensorflow as tf
#import numpy as np
#
#path = './FP3/'
#path1 = "./Images/airport_inside/airport_inside_0001.jpg"
#path2= "./Images/artstudio/art_painting_studio_01_15_altavista.jpg"
#path3= "./Images/auditorium/7629_auditorium_8_1__17.jpg"
#path4= "./Images/bowling/bowling_0011.jpg"
#path5= "./Images/casino/casino_0004.jpg"
#
#scenes_dict = {0: 'airport_inside', 1: 'artstudio', 2: 'auditorium', 3: 'bakery', 4: 'bar',
#               5: 'bedroom', 6: 'bookstore', 7: 'bowling', 8: 'buffet', 9: 'casino'}
##字典存放类名
#
#w = 100#设置图片宽度
#h = 100#设置图片高度
#
#
#cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
#
##    imgs = []
##    labels = []
#data = []
#data_name = []
#for idx, folder in enumerate(cate):
#    print(idx)
#    print(folder)
#    for im in glob.glob(folder + '/*.jpg'):
#        #print('reading the images:%s' % (im))
#        img = io.imread(im)        
#        img = transform.resize(img, (w, h))
#        imgname = os.path.dirname(im)
#        imgname = imgname.strip('./FP3/')
#        data.append(np.asarray(img))
#        data_name.append(imgname)
#        
#list_01 = list(enumerate(cate))
#
#category_str = str(list_01[0])
#print(data_name)

import tensorflow as tf

a_cpu = tf.Variable(0, name="a_cpu")
with tf.device('/gpu:0'):
    a_gpu = tf.Variable(0, name="a_gpu")

# 通过allow_soft_placement参数自动将无法放在GPU上的操作放回CPU上。
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_ placement=True))
sess.run(tf.initialize_all_variables())