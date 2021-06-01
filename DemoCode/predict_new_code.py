# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 15:37:07 2020

@author: hanxi
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 13:33:13 2020

@author: hanxi
"""

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
    
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
   

        saver = tf.train.import_meta_graph('./scenes_model12_1_4D100_1000.ckpt.meta')
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
        rig = []
        mistake = []
        output = tf.argmax(classification_result, 1).eval()
            
    for num_data in range(len(data)):
        
    #    print(num_data)
    #    print('11111111111')
    #    print(data[num_data])
        
    #     with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
       
    
    #         saver = tf.train.import_meta_graph('./scenes_model12_1_4D100_1000.ckpt.meta')
    #         saver.restore(sess, tf.train.latest_checkpoint('./'))
        
    #         graph = tf.get_default_graph()
    #         x = graph.get_tensor_by_name("x:0")
    #         feed_dict = {x: data}
        
    #         logits = graph.get_tensor_by_name("logits_eval:0")
        
    #         classification_result = sess.run(logits, feed_dict)
    
    #         # 打印出预测矩阵
    # #        print(classification_result)
    #         # 打印出预测矩阵每一行最大值的索引
    # #        print(tf.argmax(classification_result, 1).eval())
    #         # 根据索引通过字典对应场景分类
    #         output = []
    #         rig = []
    #         mistake = []
    #         output = tf.argmax(classification_result, 1).eval()
            if data_name[num_data] == scenes_dict[output[num_data]]:
                acc += 1
                print("The prediction of this time:" + scenes_dict[output[num_data]], '***prediction Correct ')
            else:
                acc = acc
                rig.append(data_name[num_data])
                mistake.append(scenes_dict[output[num_data]])
                print("The prediction of this time:" + scenes_dict[output[num_data]], '***prediction error')
            num_img += 1
            print("The current prediction result accuracy rate is:", (acc / num_img) * 100,'%' )
            print("The prediction of the ", num_data + 1, "image is:" + scenes_dict[output[num_data]])
            print('------------------------------------------------------------')
    mylog.close()
print(rig)  
print(mistake)

data_name_num = data_name
for i in range(0,len(data_name)): 
    if data_name[i] == 'airport_inside':
        data_name_num[i] = 0
    if data_name[i] == 'artstudio':
        data_name_num[i] = 1
    if data_name[i] == 'auditorium':
        data_name_num[i] = 2
    if data_name[i] == 'bakery':
        data_name_num[i] = 3
    if data_name[i] == 'bar':
        data_name_num[i] = 4
    if data_name[i] == 'bedroom':
        data_name_num[i] = 5
    if data_name[i] == 'bookstore':
        data_name_num[i] = 6
    if data_name[i] == 'bowling':
        data_name_num[i] = 7
    if data_name[i] == 'buffet':
        data_name_num[i] = 8
    if data_name[i] == 'casino':
        data_name_num[i] = 9
        

print(data_name_num)
print(output)
   #    num_data += 1
    #        return np.asarray(img)