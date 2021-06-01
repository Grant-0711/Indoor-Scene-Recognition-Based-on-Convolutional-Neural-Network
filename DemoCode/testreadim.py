from skimage import io, transform
import glob
import os
import tensorflow as tf
import numpy as np
import time
from PIL import Image
import os.path
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
from PIL import Image
import warnings
import pickle
# 数据集地址
path = './FP6/'
# 模型保存地址

features_dir = 'D:/FP/data_F_1'



output_folder = './data_F_2'
config = tf.ConfigProto() 

config.gpu_options.per_process_gpu_memory_fraction = 0.9 # 占用GPU40%的显存
# def one_to_threeim(image,label):
#     with tf.Session(config=config):
#     	image, label = sess.run(next_batch)	# batch_size=1
    	
#     	image = np.concatenate((image, image, image), axis=-1)
#     	return image



# 加一个判断保证文件夹存在
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

transform1 = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()])


cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]

imgs = []
img1_mx = []
feature_path_mx = []
labels = []
for idx, folder in enumerate(cate):
        # print(folder)
    for im in glob.glob(folder + '/*.jpg'):
        file_name = im.split('./FP5/airport_inside')[-1]
        feature_path = os.path.join('./FP5/data' + features_dir, file_name + '.txt')
        feature_path_mx.append(feature_path)
        img = Image.open(im)
        img1 = transform1(img)
        # img1 = one_to_threeim(img1, idx)
        img1_mx.append(img1)
# print(img1_mx)
        labels.append(idx)
    # return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)
batch_size = 100
num_batches = int(len(img1_mx) / batch_size)



resnet50_feature_extractor = models.resnet50(pretrained = True)
resnet50_feature_extractor.fc = nn.Linear(2048, 2048)
torch.nn.init.eye(resnet50_feature_extractor.fc.weight)

for param in resnet50_feature_extractor.parameters():
    param.requires_grad = False



# # for im in range(0, len(img1_mx) -1):
# #     x = Variable(torch.unsqueeze(img1_mx[im], dim=0).float(), requires_grad=False)

# # # print(x)
# #     y = resnet50_feature_extractor(x)
# #     y = y.data.numpy()
# #     feature_v.append(y)
# # feature_v = np.vstack(feature_v)

#         # 打开这个全路径文件


for i in range(num_batches):
        batch_img_data = img1_mx[i*batch_size:(i+1)*batch_size]
        batch_img_labels = labels[i*batch_size:(i+1)*batch_size]
        # print('1')
        # print(batch_img_labels)
        feature_v = []
        for j in batch_img_data:
            # print('111')
            x_va = Variable(torch.unsqueeze(j, dim=0).float(), requires_grad=False)

# print(x)
            y_va = resnet50_feature_extractor(x_va)
            y_va = y_va.data.numpy()
            feature_v.append(y_va)
            # print('111')
            # print(feature_v)
        feature_v1 = np.vstack(feature_v)
        save_path = os.path.join(output_folder,'data_%d.pickle'%i)
        with tf.gfile.FastGFile(save_path,'w') as f:
     	 # 写入这个批次的向量，便于后续提取
                pickle.dump((feature_v1,batch_img_labels),f)

        print(save_path,'is_ok!')
