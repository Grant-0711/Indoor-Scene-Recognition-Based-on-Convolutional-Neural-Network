# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 18:21:05 2020

@author: hanxi
"""

#导入mnist数据
import input_data
mnist = input_data.read_data_sets("./FP4p/", one_hot=True)
# start tensorflow interactiveSession
import tensorflow as tf
sess = tf.InteractiveSession()

# weight initialization
#初始化时加入轻微噪声，来打破对称性，防止零梯度问题
#权重初始化
def weight_variable(shape):
 #截断正态分布
 initial = tf.truncated_normal(shape, stddev=0.1)
 return tf.Variable(initial)

#偏置初始化
def bias_variable(shape):
 initial = tf.constant(0.1, shape = shape)
 return tf.Variable(initial)

# convolution卷积
#卷积使用1步长(stride size)，0边距(padding size)的模板，
#padding='SAME'说明在
#保证输出和输入是同一个大小
def conv2d(x, W):
 return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
# pooling池化
#把特征图像区域的一部分求个均值或者最大值，用来代表这部分区域。
#如果是求均值就是mean pooling，求最大值就是max pooling。
#池化用简单传统的2x2大小的模板做max pooling
def max_pool_2x2(x):
 return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 输入任意数量的图像，每一张图平铺成784维向量
x = tf.placeholder("float", [None, 784])
# target为10维标签向量
y_ = tf.placeholder("float", [None, 10])
# 权重是784*10，偏置值是[10]
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
# y=x*W+b
y = tf.nn.softmax(tf.matmul(x,W) + b)

# 第一个卷积层
# 权重是一个 [5, 5, 1, 32] 的张量，前两个维度是patch的大小，
# 接着是输入的通道数目，最后是输出的通道数目。
# 输出对应一个同样大小的偏置向量。
w_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
# 为了用这一层，我们把 x 变成一个4d向量，
# 第2、3维对应图片的宽高，最后一维代表颜色通道。
x_image = tf.reshape(x, [-1, 28, 28, 1])
'''
x_image 和权值向量进行卷积相乘，加上偏置，
使用ReLU激活函数，最后max pooling
'''
#h_conv1由于步长是1，输出单张图片大小不变是[28,28]
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
#h_pool1由于步长是2，输出单张图片大小减半[14,14]
h_pool1 = max_pool_2x2(h_conv1)

# 第二个卷积层
'''
为了构建一个更深的网络，我们会把几个类似的层堆叠起来。
第二层中，每个5x5的patch会得到64个特征。
'''
w_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

#h_conv2由于步长是1，输出单张图片大小不变是[14,14]
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
#h_pool2由于步长是2，输出单张图片大小减半[7,7]
h_pool2 = max_pool_2x2(h_conv2)

# densely connected layer密集连接层
'''
现在，图片降维到7x7，我们加入一个有1024个神经元的全连接层，
用于处理整个图片。我们把池化层输出的张量reshape成一些向量，
乘上权重矩阵，加上偏置，使用ReLU激活。
'''
w_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

# dropout
'''
为了减少过拟合，我们在输出层之前加入dropout。我们用一个 placeholder 来代表一个神经元在dropout中被保留的概率。
这样我们可以在训练过程中启用dropout，在测试过程中关闭dropout。 
TensorFlow的 操作会自动处理神经元输出值的scale。
所以用dropout的时候可以不用考虑scale。
'''
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# readout layer
#添加一个softmax层，就像前面的单层softmax regression一样
w_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)
'''
我们会用更加复杂的ADAM优化器来做梯度最速下降，
在 feed_dict 中加入额外的参数keep_prob来控制dropout比例。
然后每100次迭代输出一次日志。
'''
# train and evaluate the model训练和评价模型
#计算交叉熵
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
#梯度下降求最小交叉熵
train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy)
#train_step = tf.train.AdagradOptimizer(1e-5).minimize(cross_entropy)
#检测我们的预测是否真实标签匹配
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
#把布尔值转换成浮点数，然后取平均值
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#初始化变量
sess.run(tf.initialize_all_variables())
for i in range(20000):
 #随机抓取训练数据中的50个批处理数据点，然后我们用这些数据点作为参数替换 之前的占位符来运行train_step
 batch = mnist.train.next_batch(50)
 #每100次打印下
 if i%100 == 0:
  train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_:batch[1], keep_prob:1.0})
  print("step %d, train accuracy %g" %(i, train_accuracy))
 train_step.run(feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})

print("test accuracy %g" % accuracy.eval(feed_dict={x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0}))