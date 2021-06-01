# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 04:15:06 2020

@author: hanxi
"""

import numpy as np
import matplotlib.pyplot as plt
num = np.arange(1,100001,1000)#生成10行4列的数组

num1 = np.arange(1,501,1)#生成10行4列的数组
# plt.rcParams['font.sans-serif'] = ['SimHei']#可以plt绘图过程中中文无法显示的问题
# plt_label = 0
# for link in range(len(num)):
    # plt_label += 1

#Ac100 = [12.999999523162842, 15.000000596046448, 15.999999642372131, 28.999999165534973, 25.0, 38.999998569488525, 30.000001192092896, 43.00000071525574, 34.99999940395355, 44.999998807907104, 38.999998569488525, 50.0, 40.99999964237213, 51.99999809265137, 46.00000083446503, 51.99999809265137, 49.000000953674316, 55.000001192092896, 49.000000953674316, 56.00000023841858, 50.0, 56.00000023841858, 50.0, 56.00000023841858, 50.0, 56.00000023841858, 50.999999046325684, 56.00000023841858, 51.99999809265137, 56.99999928474426, 50.0, 54.00000214576721, 51.99999809265137, 50.999999046325684, 51.99999809265137, 50.999999046325684, 52.99999713897705, 51.99999809265137, 50.999999046325684, 52.99999713897705, 52.99999713897705, 50.999999046325684, 52.99999713897705, 49.000000953674316, 55.000001192092896, 49.000000953674316, 56.00000023841858, 46.99999988079071, 55.000001192092896, 43.00000071525574, 56.00000023841858, 41.999998688697815, 51.99999809265137, 41.999998688697815, 49.000000953674316, 41.999998688697815, 49.000000953674316, 41.999998688697815, 49.000000953674316, 41.999998688697815, 50.0, 43.00000071525574, 51.99999809265137, 43.00000071525574, 51.99999809265137, 43.99999976158142, 51.99999809265137, 43.99999976158142, 52.99999713897705, 43.99999976158142, 51.99999809265137, 43.99999976158142, 50.999999046325684, 40.00000059604645, 50.0, 38.999998569488525, 50.0, 37.99999952316284, 52.99999713897705, 37.00000047683716, 52.99999713897705, 34.99999940395355, 52.99999713897705, 37.99999952316284, 52.99999713897705, 40.00000059604645, 51.99999809265137, 40.99999964237213, 50.999999046325684, 41.999998688697815, 50.0, 41.999998688697815, 50.0, 41.999998688697815, 51.99999809265137, 41.999998688697815, 50.0, 40.99999964237213, 47.999998927116394, 40.00000059604645][12.999999523162842, 15.000000596046448, 15.999999642372131, 28.999999165534973, 25.0, 38.999998569488525, 30.000001192092896, 43.00000071525574, 34.99999940395355, 44.999998807907104, 38.999998569488525, 50.0, 40.99999964237213, 51.99999809265137, 46.00000083446503, 51.99999809265137, 49.000000953674316, 55.000001192092896, 49.000000953674316, 56.00000023841858, 50.0, 56.00000023841858, 50.0, 56.00000023841858, 50.0, 56.00000023841858, 50.999999046325684, 56.00000023841858, 51.99999809265137, 56.99999928474426, 50.0, 54.00000214576721, 51.99999809265137, 50.999999046325684, 51.99999809265137, 50.999999046325684, 52.99999713897705, 51.99999809265137, 50.999999046325684, 52.99999713897705, 52.99999713897705, 50.999999046325684, 52.99999713897705, 49.000000953674316, 55.000001192092896, 49.000000953674316, 56.00000023841858, 46.99999988079071, 55.000001192092896, 43.00000071525574, 56.00000023841858, 41.999998688697815, 51.99999809265137, 41.999998688697815, 49.000000953674316, 41.999998688697815, 49.000000953674316, 41.999998688697815, 49.000000953674316, 41.999998688697815, 50.0, 43.00000071525574, 51.99999809265137, 43.00000071525574, 51.99999809265137, 43.99999976158142, 51.99999809265137, 43.99999976158142, 52.99999713897705, 43.99999976158142, 51.99999809265137, 43.99999976158142, 50.999999046325684, 40.00000059604645, 50.0, 38.999998569488525, 50.0, 37.99999952316284, 52.99999713897705, 37.00000047683716, 52.99999713897705, 34.99999940395355, 52.99999713897705, 37.99999952316284, 52.99999713897705, 40.00000059604645, 51.99999809265137, 40.99999964237213, 50.999999046325684, 41.999998688697815, 50.0, 41.999998688697815, 50.0, 41.999998688697815, 51.99999809265137, 41.999998688697815, 50.0, 40.99999964237213, 47.999998927116394, 40.00000059604645]
C_Ac =[15.999999642372131, 18.000000715255737, 28.00000011920929, 34.00000035762787, 36.000001430511475, 34.00000035762787, 36.000001430511475, 36.000001430511475, 37.99999952316284, 40.00000059604645, 41.999998688697815, 41.999998688697815, 43.99999976158142, 43.99999976158142, 46.00000083446503, 46.00000083446503, 47.999998927116394, 47.999998927116394, 47.999998927116394, 47.999998927116394, 46.00000083446503, 46.00000083446503, 46.999998927116394, 47.099998927116394, 47.399998927116394, 47.599998927116394, 47.799998927116394, 47.999998927116394, 47.999998927116394, 47.999998927116394, 47.999998927116394, 47.999998927116394, 47.999998927116394, 47.999998927116394, 47.999998927116394, 47.999998927116394, 47.999998927116394, 54.00000214576721, 62.00000047683716, 50.0, 51.99999809265137, 50.0, 43.99999976158142, 43.99999976158142, 43.99999976158142, 43.99999976158142, 43.99999976158142, 43.99999976158142, 43.99999976158142, 46.00000083446503, 43.99999976158142, 46.00000083446503, 46.00000083446503, 46.50000083446503, 46.90000083446503, 47.00000083446503, 47.299998927116394, 47.399998927116394, 47.599998927116394, 47.799998927116394, 50.0, 48.999998927116394, 49.999998927116394, 54.00000214576721, 51.999998688697815, 53.0, 55.00000214576721, 56.00000023841858, 56.00000023841858, 54.00000214576721, 56.00000023841858, 57.999998331069946, 60.00000238418579, 60.999998569488525, 59.999998569488525, 59.999998569488525, 59.999998569488525, 63.999998569488525, 59.999998569488525, 59.999998569488525, 59.899998569488525, 58.599998569488525, 58.999998569488525, 59.00000262260437, 61.00000262260437, 63.00000262260437, 61.00000262260437, 62.00000262260437, 62.00000262260437, 61.00000262260437, 65.00000071525574, 61.9999988079071, 62.50000286102295, 65.10000286102295, 62.9999988079071, 63.00000071525574, 61.00000071525574, 62.00000262260437, 63.00000071525574, 60.00000238418579]

Ac300= [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 93.75, 94.11764705882352, 94.44444444444444, 94.73684210526315, 95.0, 95.23809523809523, 95.45454545454545, 95.65217391304348, 95.83333333333334, 96.0, 96.15384615384616, 96.29629629629629, 92.85714285714286, 93.10344827586206, 93.33333333333333, 93.54838709677419, 93.75, 93.93939393939394, 94.11764705882352, 94.28571428571428, 94.44444444444444, 94.5945945945946, 94.73684210526315, 94.87179487179486, 95.0, 95.1219512195122, 95.23809523809523, 93.02325581395348, 90.9090909090909, 91.11111111111111, 91.30434782608695, 91.48936170212765, 91.66666666666666, 91.83673469387756, 92.0, 92.15686274509804, 92.3076923076923, 92.45283018867924, 92.5925925925926, 92.72727272727272, 92.85714285714286, 92.98245614035088, 91.37931034482759, 91.52542372881356, 91.66666666666666, 90.1639344262295, 90.32258064516128, 90.47619047619048, 90.625, 90.76923076923077, 90.9090909090909, 91.04477611940298, 91.17647058823529, 91.30434782608695, 91.42857142857143, 90.14084507042254, 90.27777777777779, 90.41095890410958, 90.54054054054053, 90.66666666666666, 90.78947368421053, 90.9090909090909, 91.02564102564102, 91.13924050632912, 91.25, 91.35802469135803, 91.46341463414635, 91.56626506024097, 91.66666666666666, 91.76470588235294, 91.86046511627907, 91.95402298850574, 92.04545454545455, 92.13483146067416, 92.22222222222223, 92.3076923076923, 92.3913043478261, 92.47311827956989, 92.5531914893617, 92.63157894736842, 92.70833333333334, 92.78350515463917, 92.85714285714286, 92.92929292929293, 93.0]

# plt.plot(num,Loss,label = 'Training Loss of 150-4conv pixels(relu)')
# plt.plot(num,C_Loss,label = 'Training Loss of 150-4conv-notBlance class pixels(relu)')

# plt.plot(num,Ac300,label = 'Predict Accuracy of 150-4conv 300 n_epoch(relu)')
# plt.plot(num,Ac100,label = 'Predict Accuracy of 150-4conv 100 n_epoch(relu)')

plt.plot(num,C_Ac,label = 'Predict Accuracy of Get Resnet Feature ')
plt.xlabel('Training times'); plt.ylabel('Predict Accuracy(%)')
# notBlance class


plt.legend()#显示图例，如果注释改行，即使设置了图例仍然不显示
plt.show()#显示图片，如果注释改行，即使设置了图片仍然不显示