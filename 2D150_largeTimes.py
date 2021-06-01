# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 21:32:52 2020

@author: hanxi
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 21:00:10 2020

@author: hanxi
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 22:57:16 2020

@author: hanxi
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 20:32:29 2020

@author: hanxi
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 22:59:45 2020

@author: hanxi
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 16:53:10 2020

@author: hanxi
"""

from skimage import io, transform
import glob
import os
import tensorflow as tf;  
tf.reset_default_graph()
import numpy as np
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 数据集地址
path = './FP4/'
# 模型保存地址
model_path = './FP/scenes_model12_depth_2D150.ckpt'

# 将所有的图片resize成100*100
w = 150
h = 150
c = 3
NUM_CLASS=10

# 读取图片
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

# 打乱顺序
num_example = data.shape[0]
arr = np.arange(num_example)
np.random.shuffle(arr)
# This function only shuffles the array along the first index of a multi-dimensional array（多维矩阵中，只对第一维（行）做打乱顺序操作）:
data = data[arr]
label = label[arr]

# 将所有数据分为训练集和验证集
ratio = 0.9
s = np.int(num_example * ratio)
x_train = data[:s]
y_train = label[:s]
x_val = data[s:]
y_val = label[s:]

# -----------------构建网络----------------------
# 占位符
x = tf.placeholder(tf.float32, shape=[None, w, h, c], name='x')
y_ = tf.placeholder(tf.int32, shape=[None, ], name='y_')

# tf.placeholder(    dtype,    shape=None,    name=None)


# 参数：

# dtype：数据类型。常用的是tf.float32,tf.float64等数值类型
# 	shape：数据形状。默认是None，就是一维值，也可以是多维（比如[2,3], [None, 3]表示列是3，行不定）
# 	name：名称


def inference(input_tensor, train, regularizer):
    with tf.variable_scope('layer1-conv1'):
#         用于定义创建变量（层）的操作的上下文管理器。
# 此上下文管理器验证（可选）values是否来自同一图形，确保图形是默认的图形，并推送名称范围和变量范围。
        conv1_weights = tf.get_variable("weight", [11, 11, 3, 96],
                                        # NUM_CLASS:filter 大小 3：chennel 32：32个filters
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
# mean：一个python标量或一个标量张量。要生成的随机值的均值。

# stddev：一个python标量或一个标量张量。要生成的随机值的标准偏差。

# seed：一个Python整数。用于创建随机种子。查看 tf.set_random_seed 行为。

# dtype：数据类型。只支持浮点类型。
        conv1_biases = tf.get_variable("bias", [96], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 4, 4, 1], padding='SAME',use_cudnn_on_gpu=True)
        # strides=[batch size, h, w, c]
#         input：张量，必须是 half、float32、float64 三种类型之一。
# filter：张量必须具有与输入相同的类型。
# strides：整数列表。长度是 4 的一维向量。输入的每一维度的滑动窗口步幅。必须与指定格式维度的顺序相同。
# padding：可选字符串为 SAME、VALID。要使用的填充算法的类型。
# use_cudnn_on_gpu：一个可选的布尔值，默认为 True。
# data_format：可选字符串为 NHWC、NCHW，默认为 NHWC。指定输入和输出数据的数据格式。使用默认格式 NHWC，数据按照以下顺序存储：[batch，in_height，in_width，in_channels]。或者，格式可以是 NCHW，数据存储顺序为：[batch，in_channels，in_height，in_width]。
# name：操作的名称（可选）。
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    with tf.name_scope("layer2-pool1"):
        # 从输出结果可以看出，在tf.name_scope()下的所有对象和操作，其name属性前都加了cgx_name_scope，用以表示这些内容全在其范围下。
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")
# 参数说明如下： 
# value：形状为 [batch，height，width，channels] 和类型是 tf.float32 的四维张量。
# ksize：长度 >=4 的整数列表。输入张量的每个维度的窗口大小。
# strides：长度 >=4 的整数列表。输入张量的每个维度的滑动窗口的步幅。
# padding：一个字符串，可以是 VALID 或 SAME。
# data_format：一个字符串，支持 NHWC 和 NCHW。
# name：操作的可选名称。
       #   Tensorflow的设计理念称之为计算流图，在编写程序时，首先构筑整个系统的graph，代码并不会直接生效，这一点和python的其他数值计算库（如Numpy等）不同，graph为静态的，类似于docker中的镜像。然后，在实际的运行时，启动一个session，程序才会真正的运行。这样做的好处就是：避免反复地切换底层程序实际运行的上下文，tensorflow帮你优化整个系统的代码。我们知道，很多python程序的底层为C语言或者其他语言，执行一行脚本，就要切换一次，是有成本的，tensorflow通过计算流图的方式，帮你优化整个session需要执行的代码，还是很有优势的。

       # 所以placeholder()函数是在神经网络构建graph的时候在模型中的占位，此时并没有把要输入的数据传入模型，它只会分配必要的内存。等建立session，在会话中，运行模型的时候通过feed_dict()函数向占位符喂入数据。

    with tf.variable_scope("layer3-conv2"):
        conv2_weights = tf.get_variable("weight", [5, 5, 96, 256],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [256], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME',use_cudnn_on_gpu=True)
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
# 这个函数的作用是计算激活函数 relu，即 max(features, 0)。将大于0的保持不变，小于0的数置为0。
    with tf.name_scope("layer4-pool2"):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
        nodes = 8 * 8 * 256
        reshaped = tf.reshape(pool2, [-1, nodes])

    # with tf.variable_scope("layer5-conv3"):
    #     conv3_weights = tf.get_variable("weight", [3, 3, 64, 128],
    #                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
    #     conv3_biases = tf.get_variable("bias", [128], initializer=tf.constant_initializer(0.0))
    #     conv3 = tf.nn.conv2d(pool2, conv3_weights, strides=[1, 1, 1, 1], padding='SAME',use_cudnn_on_gpu=True)
    #     relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))

    # with tf.name_scope("layer6-pool3"):
    #     pool3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # with tf.variable_scope("layer7-conv4"):
    #     conv4_weights = tf.get_variable("weight", [NUM_CLASS, NUM_CLASS, 32, 64],
    #                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
    #     conv4_biases = tf.get_variable("bias", [64], initializer=tf.constant_initializer(0.0))
    #     conv4 = tf.nn.conv2d(pool1, conv4_weights, strides=[1, 1, 1, 1], padding='SAME',use_cudnn_on_gpu=True)
    #     relu4 = tf.nn.relu(tf.nn.bias_add(conv4, conv4_biases))

    # with tf.name_scope("layer8-pool4"):
    #     pool4 = tf.nn.max_pool(relu4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    #     nodes = 20 * 20 * 64
    #     reshaped = tf.reshape(pool4, [-1, nodes])

    with tf.variable_scope('layer9-fc1'):
        fc1_weights = tf.get_variable("weight", [nodes, 1024],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable("bias", [1024], initializer=tf.constant_initializer(0.1))

        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        if train: fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope('layer10-fc2'):
        fc2_weights = tf.get_variable("weight", [1024, 512],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable("bias", [512], initializer=tf.constant_initializer(0.1))

        fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_biases)
        if train: fc2 = tf.nn.dropout(fc2, 0.5)

    with tf.variable_scope('layer11-fc3'):
        fc3_weights = tf.get_variable("weight", [512, NUM_CLASS],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc3_weights))
        fc3_biases = tf.get_variable("bias", [NUM_CLASS], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc2, fc3_weights) + fc3_biases

    return logit


# ---------------------------网络结束---------------------------
regularizer = tf.contrib.layers.l2_regularizer(0.0001)
logits = inference(x, False, regularizer)

# (小处理)将logits乘以1赋值给logits_eval，定义name，方便在后续调用模型时通过tensor名字调用输出tensor
b = tf.constant(value=1, dtype=tf.float32)
logits_eval = tf.multiply(logits, b, name='logits_eval')

loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_)
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# 定义一个函数，按批次取数据
def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]


# 训练和测试数据，可将n_epoch设置更大一些
config = tf.ConfigProto() 

config.gpu_options.per_process_gpu_memory_fraction = 0.9 # 占用GPU40%的显存 
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

n_epoch = 500
batch_size = 16
saver = tf.train.Saver()

plot_data_loss = []
plot_data_ac = []

with tf.device('/gpu:0'):
    for epoch in range(n_epoch):
        start_time = time.time()
    
        # training
        train_loss, train_acc, n_batch = 0, 0, 0 
        for x_train_a, y_train_a in minibatches(x_train, y_train, batch_size, shuffle=True):
            _, err, ac = sess.run([train_op, loss, acc], feed_dict={x: x_train_a, y_: y_train_a})
            train_loss += err;
            train_acc += ac;
            n_batch += 1
        print("===========%d ============="%epoch)
        print("   train loss: %f" % (np.sum(train_loss) / n_batch))
        plot_data_loss.append((np.sum(train_loss) / n_batch))
        print("   train acc: %f" % (np.sum(train_acc) / n_batch))
        plot_data_ac.append((np.sum(train_acc) / n_batch))

print(plot_data_loss)
print(plot_data_ac)

f = open('plotloss.txt','w')

print(plot_data_loss,file = f)
# print(plot_data_ac,file = f)

f.close

f = open('plotac.txt','w')

# print(plot_data_loss,file = f)
print(plot_data_ac,file = f)

f.close
        # # validation
        # val_loss, val_acc, n_batch = 0, 0, 0
        # for x_val_a, y_val_a in minibatches(x_val, y_val, batch_size, shuffle=False):
        #     err, ac = sess.run([loss, acc], feed_dict={x: x_val_a, y_: y_val_a})
        #     val_loss += err;
        #     val_acc += ac;
        #     n_batch += 1
        # print("   validation loss: %f" % np.sum(val_loss) )
        # print("   validation loss: %f" % (np.sum(val_loss) / n_batch))
        # print("   validation acc: %f" % (np.sum(val_acc) / n_batch))
saver.save(sess, model_path)
sess.close()
