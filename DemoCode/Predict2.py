# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 13:44:08 2020

@author: hanxi
"""

from skimage import io, transform

import tensorflow as tf
import numpy as np


path1 = "./Images/airport_inside/airport_inside_0001.jpg"
path2= "./Images/artstudio/art_painting_studio_01_15_altavista.jpg"
path3= "./Images/auditorium/7629_auditorium_8_1__17.jpg"
path4= "./Images/bowling/bowling_0011.jpg"
path5= "./Images/casino/casino_0004.jpg"

scenes_dict = {0: 'airport_inside', 1: 'artstudio', 2: 'auditorium', 3: 'bakery', 4: 'bar',
               5: 'bedroom', 6: 'bookstore', 7: 'bowling', 8: 'buffet', 9: 'casino'}
#字典存放类名

w = 100#设置图片宽度
h = 100#设置图片高度
c = 3











def Pred(*arg):
    def read_one_image(path):#读取图片方法
        img = io.imread(path)
        img = transform.resize(img, (w, h))
        return np.asarray(img)
    
    
    with tf.Session() as sess:
        data = []
        data1 = read_one_image(path1)

        data.append(data1)

    
        saver = tf.train.import_meta_graph('./scenes_model.ckpt.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./'))
    
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("x:0")
        feed_dict = {x: data}
    
        logits = graph.get_tensor_by_name("logits_eval:0")
    
        classification_result = sess.run(logits, feed_dict)
    
        # 打印出预测矩阵
        # print(classification_result)
        # # 打印出预测矩阵每一行最大值的索引
        # print(tf.argmax(classification_result, 1).eval())
        # # 根据索引通过字典对应场景分类
        output = []
        output = tf.argmax(classification_result, 1).eval()
        f = open("out.txt", "w")    # 打开文件以便写入
        
         #  关闭文件
        for i in range(len(output)):
            print("The", i + 1, "st image predicted result:" + scenes_dict[output[i]],file=f)
        f.close 
            
Pred(path)