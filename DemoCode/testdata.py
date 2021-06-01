# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 03:15:58 2020

@author: hanxi
"""

from skimage import io, transform
import glob
import os
import tensorflow as tf
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

with tf.device('/gpu:0'):

    path = './FP4/'
    
    scenes_dict = {0: 'airport_inside', 1: 'artstudio', 2: 'auditorium', 3: 'bakery', 4: 'bar',
                   5: 'bedroom', 6: 'bookstore', 7: 'bowling', 8: 'buffet', 9: 'casino'}
    #字典存放类名
    
    w = 100#设置图片宽度
    h = 100#设置图片高度
    c = 3
    
    
    cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    
    #    imgs = []
    #    labels = []
    data = []
    data_name = []
    
    for idx, folder in enumerate(cate):
        
        for im in glob.glob(folder + '/*.jpg'):
            #print('reading the images:%s' % (im))
            img = io.imread(im)        
            img = transform.resize(img, (w, h, c))
            imgname = os.path.dirname(im)
            imgname = imgname.strip('./FP4/')
            data.append(np.asarray(img))
            data_name.append(imgname)
            
    
    acc = 0
    num_img = 0#print(len(data))
    mylog = open('recode.log', mode = 'a',encoding='utf-8')
    for num_data in range(len(data)):
        
    #    print(num_data)
    #    print('11111111111')
    #    print(data[num_data])
        
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
       
    
            saver = tf.train.import_meta_graph('./scenes_model.ckpt.meta')
            saver.restore(sess, tf.train.latest_checkpoint('./'))
        
            graph = tf.get_default_graph()
            x = graph.get_tensor_by_name("x:0")
            feed_dict = {x: data}
        
            logits = graph.get_tensor_by_name("logits_eval:0")
        
            classification_result = sess.run(logits, feed_dict)
    
            # 打印出预测矩阵
    #        print(classification_result)
            # 打印出预测矩阵每一行最大值的索引
    #        print(tf.argmax(classification_result, 1).eval())
            # 根据索引通过字典对应场景分类
            output = []
            output = tf.argmax(classification_result, 1).eval()
            if data_name[num_data] == scenes_dict[output[num_data]]:
                acc += 1
                print("The prediction of this time:" + scenes_dict[output[num_data]], '***prediction Correct ')
            else:
                acc = acc
                print("The prediction of this time:" + scenes_dict[output[num_data]], '***prediction error')
            num_img += 1
            print("The current prediction result accuracy rate is:", (acc / num_img) * 100,'%' )
            print("The prediction of the ", num_data + 1, "image is:" + scenes_dict[output[num_data]])
            print('------------------------------------------------------------')
    mylog.close()
    #    num_data += 1
    #        return np.asarray(img)

