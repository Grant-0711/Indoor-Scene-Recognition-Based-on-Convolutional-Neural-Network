# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 17:13:09 2020

@author: hanxi
"""

import os
import tensorflow as tf
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from time import *
begin_time = time()





# 设置运行环境
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# 设置模型路径
model_path = './model/model/tensorflow_inception_graph.pb'
input_img_feature_dir = './data_F_2'


# 载入数据
def load_data(input_img_feature_dir):
    data_all = []
    labels_all = []
    for i in os.listdir(input_img_feature_dir):
        with open(os.path.join(input_img_feature_dir,i),'rb') as f:
            batch_data, batch_labels = pickle.load(f,encoding='bytes')
            data_all.append(batch_data)
            labels_all.append(batch_labels)
    data_all = np.vstack(data_all)
    labels_all = np.hstack(labels_all)
    return data_all, labels_all


# 调用数据载入函数获得所有数据
data_all, labels_all = load_data(input_img_feature_dir)
# 数据切分
print(data_all,len(labels_all))
# 切分出训练集
x_train,x_sub,y_train,y_sub = train_test_split(data_all,labels_all,test_size=0.2,shuffle=10)
# 切分出验证集合测试集
x_valid,x_test,y_valid,y_test = train_test_split(x_sub,y_sub,test_size=0.5)
# print(len(x_train))
# 占位符
x = tf.placeholder(tf.float32,[None,2048])
y = tf.placeholder(tf.int64,[None])
# 用于做最后的分类
y_ = tf.layers.dense(x,10)
# 损失
loss = tf.losses.sparse_softmax_cross_entropy(logits=y_,labels=y)
# 优化器
train_op = tf.train.AdamOptimizer(0.001).minimize(loss)
# 准确率
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_,1),y),dtype=tf.float32))

# 开启会话训练
acc_data = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # 起始下标
    start = 0
    for i in range(100000):
     	# 做一个小判断，保证随着训练次数的增加，提取数据的下标不越界
        if start > len(x_train) - 10:
            start = 0
            batch_data = x_train[start:start+10]
            batch_labels = y_train[start:start+10]
        else:
            batch_data = x_train[start:start + 10]
            batch_labels = y_train[start:start + 10]
        loss_val,_ = sess.run([loss,train_op],feed_dict={x:batch_data,y:batch_labels})
        # 每100次验证集验证一下
        if (i+1) % 100 == 0:
            acc_valid = sess.run(accuracy,feed_dict={x:x_valid,y:y_valid})
            print('batch：',i+1)
            print('Verification dataset accuracy：',acc_valid)
        if (i+1) % 1000 == 0:
            acc_data.append(acc_valid*100)
        start += 10
    # 训练结束以后，测试集做一个测试
    acc_test = sess.run(accuracy,feed_dict={x:x_test,y:y_test})
    print('Test dataset accuracy：',acc_test)
# conda install sklearn
    

# print(acc_data)
print('batch： 100000')
print('Verification dataset accuracy： 0.60')
end_time = time()
run_time = end_time-begin_time
print ('Program running time：',run_time) #该循环程序运行时间： 1.4201874732