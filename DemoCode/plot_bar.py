# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 13:35:07 2020

@author: hanxi
"""

import numpy as np
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 10), dpi=80)
# 再创建一个规格为 1 x 1 的子图
# plt.subplot(1, 1, 1)
# 柱子总数
N = 10
# 包含每个柱子对应值的序列
num_list= ['airport_in','artstudio','auditorium','bakery','bar',
                    'bedroom', 'bookstore', 'bowling',  'buffet','casino']
values = (12.4,6.94
,11.9
,7.2
,8.26
,12.9
,18.3
,6.7
,8.4
,7)
# 包含每个柱子下标的序列
index = np.arange(N)
# 柱子的宽度
width = 0.45
# 绘制柱状图, 每根柱子的颜色为紫罗兰色
p2 = plt.bar(index, values, width, label="Error rate value", color="#87CEFA")
# 设置横轴标签
plt.xlabel('Category')
# 设置纵轴标签
plt.ylabel('Error rate')
# 添加标题
plt.title('System Error Distribution Ratio')
# 添加纵横轴的刻度
plt.xticks(index, ('airport_in','artstudio','auditorium','bakery','bar',
                    'bedroom', 'bookstore', 'bowling',  'buffet','casino'),fontsize=10)
plt.yticks(np.arange(0, 45, 10))
# 添加图例
plt.legend(loc="upper right")
for a,b in zip(index,values):   #柱子上的数字显示
 plt.text(a,b,'%.2f'%b,ha='center',va='bottom',fontsize=18);

plt.show()
            # scenes_dict = {0: 'airport_inside', 1: 'artstudio', 2: 'auditorium', 3: 'bakery', 4: 'bar',
            #        5: 'bedroom', 6: 'bookstore', 7: 'bowling', 8: 'buffet', 9: 'casino'}