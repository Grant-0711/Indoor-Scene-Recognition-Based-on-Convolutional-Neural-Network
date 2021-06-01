# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 15:14:23 2020

@author: hanxi
"""

Rig = ['airport_inside', 'airport_inside', 'airport_inside', 'airport_inside', 'airport_inside', 'airport_inside', 'airport_inside',
       'airport_inside', 'airport_inside', 'airport_inside', 'airport_inside', 'airport_inside',
       'airport_inside', 'airport_inside', 'airport_inside', 'airport_inside', 'airport_inside',
       'airport_inside', 'airport_inside', 'airport_inside', 'artstudio', 'artstudio', 'artstudio',
       'artstudio', 'artstudio', 'artstudio', 'artstudio', 'auditorium', 'auditorium', 'auditorium',
       'auditorium', 'auditorium', 'auditorium', 'auditorium', 'auditorium', 'auditorium', 'auditorium',
       'auditorium', 'auditorium', 'auditorium', 'auditorium', 'auditorium', 'auditorium', 'auditorium', 'auditorium',
       'auditorium', 'auditorium', 'auditorium', 'auditorium', 'auditorium', 'auditorium', 'auditorium', 'auditorium', 'auditorium',
       'auditorium', 'auditorium', 'auditorium', 'auditorium', 'auditorium', 'auditorium', 'bakery', 'bakery', 'bakery', 'bakery', 'bakery',
       'bakery', 'bakery', 'bakery', 'bakery', 'bakery', 'bakery', 'bakery', 'bakery', 'bakery', 'bakery', 'bakery', 'bakery', 'bar', 'bar',
       'bar', 'bar', 'bar', 'bar', 'bar', 'bar', 'bar', 'bar', 'bar', 'bar', 'bar', 'bar', 'bar', 'bar', 'bar', 'bar', 'bar', 'bar', 'bar', 'bar',
       'bar', 'bar', 'bar', 'bar', 'bar', 'bar', 'bar', 'bar', 'bar', 'bar', 'bar', 'bar', 'bar', 'bar', 'bar', 'bar', 'bar', 'bar', 'bar', 'bedroom', 'bedroom', 'bedroom', 'bedroom', 'bedroom', 'bedroom', 'bedroom', 'bedroom', 'bedroom', 'bedroom', 'bedroom', 'bedroom', 'bedroom', 'bedroom', 'bedroom', 'bedroom', 'bedroom', 'bedroom', 'bedroom', 'bedroom', 'bedroom', 'bedroom', 'bedroom', 'bedroom', 'bedroom', 'bedroom', 'bedroom', 'bedroom', 'bedroom', 'bedroom', 'bedroom', 'bedroom', 'bedroom', 'bookstore', 'bookstore', 'bookstore', 'bookstore', 'bookstore', 'bookstore', 'bookstore', 'bookstore', 'bookstore', 'bookstore', 'bookstore', 'bookstore', 'bookstore', 'bookstore', 'bookstore', 'bookstore', 'bookstore', 'bookstore', 'bookstore', 'bookstore', 'bookstore', 'bookstore', 'bookstore', 'bookstore', 'bookstore', 'bookstore', 'bookstore', 'bookstore', 'bookstore', 'bookstore', 'bookstore', 'bookstore', 'bookstore', 'bookstore', 'bookstore', 'bookstore', 'bookstore', 'bookstore', 'bookstore', 'bookstore', 'bookstore', 'bowling', 'bowling', 'bowling', 'bowling', 'bowling', 'bowling', 'bowling', 'bowling', 'bowling', 'bowling', 'bowling', 'bowling', 'bowling', 'bowling', 'bowling', 'bowling', 'buffet', 'buffet', 'buffet', 'buffet', 'buffet', 'buffet', 'buffet', 'buffet', 'buffet', 'buffet', 'buffet', 'buffet', 'buffet', 'buffet', 'buffet', 'buffet', 'buffet', 'buffet', 'buffet', 'buffet', 'casino', 'casino', 'casino', 'casino', 'casino', 'casino', 'casino', 'casino', 'casino']

Mis = ['bedroom', 'bookstore', 'bedroom', 'bowling', 'bowling', 'bowling', 'bedroom', 'artstudio', 'bedroom', 'casino', 'casino', 'bowling', 'auditorium', 'bowling', 'bakery', 'artstudio', 'casino', 'bakery', 'bowling', 'artstudio', 'auditorium', 'casino', 'casino', 'bowling', 'casino', 'airport_inside', 'bakery', 'buffet', 'artstudio', 'bakery', 'casino', 'airport_inside', 'bakery', 'bedroom', 'bakery', 'casino', 'bowling', 'bakery', 'bowling', 'bar', 'artstudio', 'airport_inside', 'airport_inside', 'artstudio', 'casino', 'bookstore', 'casino', 'buffet', 'bowling', 'buffet', 'bowling', 'buffet', 'bar', 'casino', 'airport_inside', 'bar', 'bowling', 'bar', 'casino', 'casino', 'airport_inside', 'bowling', 'bookstore', 'bowling', 'casino', 'bookstore', 'casino', 'bedroom', 'bookstore', 'buffet', 'bookstore', 'artstudio', 'artstudio', 'buffet', 'bookstore', 'bookstore', 'casino', 'bowling', 'bakery', 'artstudio', 'buffet', 'airport_inside', 'casino', 'casino', 'bakery', 'auditorium', 'bookstore', 'casino', 'auditorium', 'bookstore', 'buffet', 'casino', 'bowling', 'bedroom', 'casino', 'bedroom', 'casino', 'bakery', 'bookstore', 'bakery', 'casino', 'artstudio', 'bowling', 'casino', 'casino', 'casino', 'artstudio', 'bowling', 'bakery', 'bakery', 'bookstore', 'casino', 'bookstore', 'bedroom', 'bakery', 'artstudio', 'casino', 'artstudio', 'bowling', 'bowling', 'airport_inside', 'artstudio', 'artstudio', 'airport_inside', 'bowling', 'bowling', 'auditorium', 'casino', 'bowling', 'artstudio', 'bowling', 'bowling', 'artstudio', 'bakery', 'bowling', 'bar', 'bakery', 'casino', 'artstudio', 'airport_inside', 'bowling', 'bakery', 'artstudio', 'bowling', 'bakery', 'bookstore', 'bakery', 'bowling', 'artstudio', 'bowling', 'bakery', 'bowling', 'bakery', 'bedroom', 'bakery', 'casino', 'bakery', 'bedroom', 'bowling', 'casino', 'bakery', 'artstudio', 'bakery', 'auditorium', 'casino', 'bar', 'artstudio', 'bowling', 'casino', 'bedroom', 'artstudio', 'bar', 'airport_inside', 'bakery', 'casino', 'casino', 'casino', 'bakery', 'casino', 'casino', 'bakery', 'bowling', 'casino', 'artstudio', 'bakery', 'auditorium', 'bakery', 'bakery', 'bar', 'bowling', 'bar', 'bowling', 'bar', 'bookstore', 'bookstore', 'casino', 'casino', 'casino', 'bakery', 'airport_inside', 'bookstore', 'airport_inside', 'auditorium', 'auditorium', 'artstudio', 'airport_inside', 'airport_inside', 'bakery', 'bakery', 'bakery', 'bakery', 'bakery', 'bakery', 'casino', 'bakery', 'bakery', 'bedroom', 'bar', 'bakery', 'bedroom', 'bowling', 'bakery', 'bookstore', 'bakery', 'bookstore', 'bowling', 'bowling', 'casino', 'bowling', 'bedroom', 'bar', 'bakery', 'bowling', 'artstudio', 'artstudio', 'auditorium', 'bowling']


# ['bedroom', 'artstudio', 'auditorium', 'bowling', 'bowling', 'bowling', 'bedroom', 'bedroom', 'bowling', 'artstudio', 'buffet', 'bedroom', 'airport_inside', 'bakery', 'bakery', 'casino', 'bedroom', 'casino', 'airport_inside', 'auditorium', 'casino', 'buffet', 'bakery', 'casino', 'buffet', 'artstudio', 'bakery', 'bedroom', 'bowling', 'bowling', 'bowling', 'bowling', 'airport_inside', 'bowling', 'bedroom', 'bowling', 'buffet', 'artstudio', 'casino', 'buffet', 'artstudio', 'bakery', 'bakery', 'bowling', 'casino', 'casino', 'bar', 'bowling', 'casino', 'bowling', 'bakery', 'bakery', 'bedroom', 'bakery', 'casino', 'bakery', 'bakery', 'bedroom', 'bedroom', 'casino', 'casino', 'bookstore', 'bowling', 'bowling', 'bakery', 'casino', 'artstudio', 'artstudio', 'bar', 'bowling', 'bowling', 'casino', 'bedroom', 'bookstore', 'artstudio', 'bar', 'casino', 'bowling', 'auditorium', 'bookstore', 'bowling', 'buffet', 'bedroom', 'bowling', 'bedroom', 'casino', 'artstudio', 'casino', 'casino', 'casino', 'bedroom', 'buffet', 'casino', 'bookstore', 'casino', 'bookstore', 'bookstore', 'bowling', 'auditorium', 'buffet', 'buffet', 'artstudio', 'casino', 'buffet', 'bowling', 'buffet', 'bowling', 'bowling', 'casino', 'bowling', 'bakery', 'artstudio', 'casino', 'bakery', 'casino', 'auditorium', 'artstudio', 'bowling', 'casino', 'artstudio', 'airport_inside', 'bowling', 'bookstore', 'bowling', 'bakery', 'bowling', 'artstudio', 'artstudio', 'bowling', 'bakery', 'artstudio', 'bar', 'bar', 'artstudio', 'artstudio', 'bedroom', 'bookstore', 'airport_inside', 'airport_inside', 'casino', 'bookstore', 'bakery', 'bakery', 'casino', 'auditorium', 'bakery', 'casino', 'airport_inside', 'casino', 'bowling', 'airport_inside', 'bedroom', 'auditorium', 'bakery', 'bar', 'artstudio', 'artstudio', 'bowling', 'bowling', 'auditorium', 'airport_inside', 'bowling', 'airport_inside', 'bowling', 'buffet', 'artstudio', 'airport_inside', 'bakery', 'bakery', 'bar', 'bakery', 'bakery', 'bakery', 'bar', 'auditorium', 'bowling', 'bakery']
def countX(lst, x):
    count = 0
    for ele in lst:
        if (ele == x):
            count = count + 1
    return count
c_0 = []
c_1 = []
c_2 = []
c_3 = []
c_4 = []
c_5 = []
c_6 = []
c_7 = []
c_8 = []
c_9 = []
c_0m = []
c_1m = []
c_2m = []
c_3m = []
c_4m = []
c_5m= []
c_6m = []
c_7m = []
c_8m = []
c_9m = []
# e_rig = enumerate()
for i in range(0,len(Rig)):

    if Rig[i] == 'airport_inside':
       c_0.append(Rig[i])
       c_0m.append(Mis[i])

    if Rig[i] == 'artstudio':
       c_1.append(Rig[i])
       c_1m.append(Mis[i])
    if Rig[i] == 'auditorium':
       c_2.append(Rig[i])
       c_2m.append(Mis[i])

    if Rig[i] == 'bakery':
       c_3.append(Rig[i])
       c_3m.append(Mis[i])

    if Rig[i] == 'bar':
       c_4.append(Rig[i])
       c_4m.append(Mis[i])
    if Rig[i] == 'bedroom':
       c_5.append(Rig[i])
       c_5m.append(Mis[i])
    if Rig[i] == 'bookstore':
       c_6.append(Rig[i])
       c_6m.append(Mis[i])
    if Rig[i] == 'bowling':
       c_7.append(Rig[i])
       c_7m.append(Mis[i])
    if Rig[i] == 'buffet':
       c_8.append(Rig[i])
       c_8m.append(Mis[i])
    if Rig[i] == 'casino':
       c_9.append(Rig[i])
       c_9m.append(Mis[i])
#
# for i in range(0,len(Rig)):
#        if i >= 0 & i < len(c_0):
#               c_0m.append(Mis[i])
e_0 = countX(Rig,'airport_inside')
e_1 = countX(Rig,'artstudio')
e_2 = countX(Rig,'auditorium')
e_3 = countX(Rig,'bakery')
e_4 = countX(Rig,'bar')
e_5 = countX(Rig,'bedroom')
e_6 = countX(Rig,'bookstore')
e_7 = countX(Rig,'bowling')
e_8 = countX(Rig,'buffet')
e_9 = countX(Rig,'casino')

e_00 = countX(c_0m,'airport_inside')
e_10 = countX(c_0m,'artstudio')
e_20 = countX(c_0m,'auditorium')
e_30 = countX(c_0m,'bakery')
e_40 = countX(c_0m,'bar')
e_50 = countX(c_0m,'bedroom')
e_60 = countX(c_0m,'bookstore')
e_70 = countX(c_0m,'bowling')
e_80 = countX(c_0m,'buffet')
e_90 = countX(c_0m,'casino')

# artstudio各类别错误个数
e_01 = countX(c_1m,'airport_inside')
e_11 = countX(c_1m,'artstudio')
e_21 = countX(c_1m,'auditorium')
e_31 = countX(c_1m,'bakery')
e_41 = countX(c_1m,'bar')
e_51 = countX(c_1m,'bedroom')
e_61 = countX(c_1m,'bookstore')
e_71 = countX(c_1m,'bowling')
e_81 = countX(c_1m,'buffet')
e_91 = countX(c_1m,'casino')

# auditorium各类别错误个数
e_02 = countX(c_2m,'airport_inside')
e_12 = countX(c_2m,'artstudio')
e_22 = countX(c_2m,'auditorium')
e_32 = countX(c_2m,'bakery')
e_42 = countX(c_2m,'bar')
e_52 = countX(c_2m,'bedroom')
e_62 = countX(c_2m,'bookstore')
e_72 = countX(c_2m,'bowling')
e_82 = countX(c_2m,'buffet')
e_92 = countX(c_2m,'casino')

# bakery各类别错误个数
e_03 = countX(c_3m,'airport_inside')
e_13 = countX(c_3m,'artstudio')
e_23 = countX(c_3m,'auditorium')
e_33 = countX(c_3m,'bakery')
e_43 = countX(c_3m,'bar')
e_53 = countX(c_3m,'bedroom')
e_63 = countX(c_3m,'bookstore')
e_73 = countX(c_3m,'bowling')
e_83 = countX(c_3m,'buffet')
e_93 = countX(c_3m,'casino')

# bar各类别错误个数
e_04 = countX(c_4m,'airport_inside')
e_14 = countX(c_4m,'artstudio')
e_24 = countX(c_4m,'auditorium')
e_34 = countX(c_4m,'bakery')
e_44 = countX(c_4m,'bar')
e_54 = countX(c_4m,'bedroom')
e_64 = countX(c_4m,'bookstore')
e_74 = countX(c_4m,'bowling')
e_84 = countX(c_4m,'buffet')
e_94 = countX(c_4m,'casino')
# bedroom各类别错误个数
e_05 = countX(c_5m,'airport_inside')
e_15 = countX(c_5m,'artstudio')
e_25 = countX(c_5m,'auditorium')
e_35 = countX(c_5m,'bakery')
e_45 = countX(c_5m,'bar')
e_55 = countX(c_5m,'bedroom')
e_65 = countX(c_5m,'bookstore')
e_75 = countX(c_5m,'bowling')
e_85 = countX(c_5m,'buffet')
e_95 = countX(c_5m,'casino')

# bookstore各类别错误个数
e_06 = countX(c_6m,'airport_inside')
e_16 = countX(c_6m,'artstudio')
e_26 = countX(c_6m,'auditorium')
e_36 = countX(c_6m,'bakery')
e_46 = countX(c_6m,'bar')
e_56 = countX(c_6m,'bedroom')
e_66 = countX(c_6m,'bookstore')
e_76 = countX(c_6m,'bowling')
e_86 = countX(c_6m,'buffet')
e_96 = countX(c_6m,'casino')

# bookstore各类别错误个数
e_07 = countX(c_7m,'airport_inside')
e_17 = countX(c_7m,'artstudio')
e_27 = countX(c_7m,'auditorium')
e_37 = countX(c_7m,'bakery')
e_47 = countX(c_7m,'bar')
e_57 = countX(c_7m,'bedroom')
e_67 = countX(c_7m,'bookstore')
e_77 = countX(c_7m,'bowling')
e_87 = countX(c_7m,'buffet')
e_97 = countX(c_7m,'casino')
# bookstore各类别错误个数
e_08 = countX(c_8m,'airport_inside')
e_18 = countX(c_8m,'artstudio')
e_28 = countX(c_8m,'auditorium')
e_38 = countX(c_8m,'bakery')
e_48 = countX(c_8m,'bar')
e_58 = countX(c_8m,'bedroom')
e_68 = countX(c_8m,'bookstore')
e_78 = countX(c_8m,'bowling')
e_88 = countX(c_8m,'buffet')
e_98 = countX(c_8m,'casino')
# bookstore各类别错误个数
e_09 = countX(c_9m,'airport_inside')
e_19 = countX(c_9m,'artstudio')
e_29 = countX(c_9m,'auditorium')
e_39 = countX(c_9m,'bakery')
e_49 = countX(c_9m,'bar')
e_59 = countX(c_9m,'bedroom')
e_69 = countX(c_9m,'bookstore')
e_79 = countX(c_9m,'bowling')
e_89 = countX(c_9m,'buffet')
e_99 = countX(c_9m,'casino')
print('The overall prediction accuracy rate is:67.6%')
print('The error rate of category airport_inside is:',e_0/len(Rig))
print('In the error object of category airport_inside, '
      'category airport_inside accounts for:',e_00/len(c_0m))
print('In the error object of category airport_inside, '
      'category artstudio accounts for:',e_10/len(c_0m))
print('In the error object of category airport_inside, '
      'category auditorium accounts for:',e_20/len(c_0m))
print('In the error object of category airport_inside, '
      'category bakery accounts for:',e_30/len(c_0m))
print('In the error object of category airport_inside, '
      'category bar accounts for:',e_40/len(c_0m))
print('In the error object of category airport_inside, '
      'category bedroom accounts for:',e_50/len(c_0m))
print('In the error object of category airport_inside, '
      'category bookstore accounts for:',e_60/len(c_0m))
print('In the error object of category airport_inside, '
      'category bowling accounts for:',e_70/len(c_0m))
print('In the error object of category airport_inside, '
      'category buffet accounts for:',e_80/len(c_0m))
print('In the error object of category airport_inside, '
      'category casino accounts for:',e_90/len(c_0m))

print('The error rate of category artstudio is:',e_1/len(Rig))
print('In the error object of category artstudio, '
      'category airport_inside accounts for:',e_01/len(c_1m))
print('In the error object of category artstudio, '
      'category artstudio accounts for:',e_11/len(c_1m))
print('In the error object of category artstudio, '
      'category auditorium accounts for:',e_21/len(c_1m))
print('In the error object of category artstudio, '
      'category bakery accounts for:',e_31/len(c_1m))
print('In the error object of category airport_inside, '
      'category bar accounts for:',e_41/len(c_1m))
print('In the error object of category airport_inside, '
      'category bedroom accounts for:',e_51/len(c_1m))
print('In the error object of category airport_inside, '
      'category bookstore accounts for:',e_61/len(c_1m))
print('In the error object of category airport_inside, '
      'category bowling accounts for:',e_71/len(c_1m))
print('In the error object of category airport_inside, '
      'category buffet accounts for:',e_81/len(c_1m))
print('In the error object of category airport_inside, '
      'category casino accounts for:',e_91/len(c_1m))

print('The error rate of category auditorium is:',e_2/len(Rig))
print('In the error object of category auditorium, '
      'category airport_inside accounts for:',e_02/len(c_2m))
print('In the error object of category airport_inside, '
      'category artstudio accounts for:',e_12/len(c_2m))
print('In the error object of category airport_inside, '
      'category auditorium accounts for:',e_22/len(c_2m))
print('In the error object of category airport_inside, '
      'category bakery accounts for:',e_32/len(c_2m))
print('In the error object of category airport_inside, '
      'category bar accounts for:',e_42/len(c_2m))
print('In the error object of category airport_inside, '
      'category bedroom accounts for:',e_52/len(c_2m))
print('In the error object of category airport_inside, '
      'category bookstore accounts for:',e_62/len(c_2m))
print('In the error object of category airport_inside, '
      'category bowling accounts for:',e_72/len(c_2m))
print('In the error object of category airport_inside, '
      'category buffet accounts for:',e_82/len(c_2m))
print('In the error object of category airport_inside, '
      'category casino accounts for:',e_92/len(c_2m))

print('The error rate of category artstudio is:',e_3/len(Rig))
print('In the error object of category airport_inside, '
      'category airport_inside accounts for:',e_03/len(c_3m))
print('In the error object of category airport_inside, '
      'category artstudio accounts for:',e_13/len(c_3m))
print('In the error object of category airport_inside, '
      'category auditorium accounts for:',e_23/len(c_3m))
print('In the error object of category airport_inside, '
      'category bakery accounts for:',e_33/len(c_3m))
print('In the error object of category airport_inside, '
      'category bar accounts for:',e_43/len(c_3m))
print('In the error object of category airport_inside, '
      'category bedroom accounts for:',e_53/len(c_3m))
print('In the error object of category airport_inside, '
      'category bookstore accounts for:',e_63/len(c_3m))
print('In the error object of category airport_inside, '
      'category bowling accounts for:',e_73/len(c_3m))
print('In the error object of category airport_inside, '
      'category buffet accounts for:',e_83/len(c_3m))
print('In the error object of category airport_inside, '
      'category casino accounts for:',e_93/len(c_3m))
print('The error rate of category artstudio is:',e_4/len(Rig))
print('In the error object of category airport_inside, '
      'category airport_inside accounts for:',e_04/len(c_4m))
print('In the error object of category airport_inside, '
      'category artstudio accounts for:',e_14/len(c_4m))
print('In the error object of category airport_inside, '
      'category auditorium accounts for:',e_24/len(c_4m))
print('In the error object of category airport_inside, '
      'category bakery accounts for:',e_34/len(c_4m))
print('In the error object of category airport_inside, '
      'category bar accounts for:',e_44/len(c_4m))
print('In the error object of category airport_inside, '
      'category bedroom accounts for:',e_54/len(c_4m))
print('In the error object of category airport_inside, '
      'category bookstore accounts for:',e_64/len(c_4m))
print('In the error object of category airport_inside, '
      'category bowling accounts for:',e_74/len(c_4m))
print('In the error object of category airport_inside, '
      'category buffet accounts for:',e_84/len(c_4m))
print('In the error object of category airport_inside, '
      'category casino accounts for:',e_94/len(c_4m))
print('The error rate of category artstudio is:',e_5/len(Rig))
print('In the error object of category airport_inside, '
      'category airport_inside accounts for:',e_05/len(c_5m))
print('In the error object of category airport_inside, '
      'category artstudio accounts for:',e_15/len(c_5m))
print('In the error object of category airport_inside, '
      'category auditorium accounts for:',e_25/len(c_5m))
print('In the error object of category airport_inside, '
      'category bakery accounts for:',e_35/len(c_5m))
print('In the error object of category airport_inside, '
      'category bar accounts for:',e_45/len(c_5m))
print('In the error object of category airport_inside, '
      'category bedroom accounts for:',e_55/len(c_5m))
print('In the error object of category airport_inside, '
      'category bookstore accounts for:',e_65/len(c_5m))
print('In the error object of category airport_inside, '
      'category bowling accounts for:',e_75/len(c_5m))
print('In the error object of category airport_inside, '
      'category buffet accounts for:',e_85/len(c_5m))
print('In the error object of category airport_inside, '
      'category casino accounts for:',e_95/len(c_5m))
print('The error rate of category artstudio is:',e_6/len(Rig))
print('In the error object of category airport_inside, '
      'category airport_inside accounts for:',e_06/len(c_6m))
print('In the error object of category airport_inside, '
      'category artstudio accounts for:',e_16/len(c_6m))
print('In the error object of category airport_inside, '
      'category auditorium accounts for:',e_26/len(c_6m))
print('In the error object of category airport_inside, '
      'category bakery accounts for:',e_36/len(c_6m))
print('In the error object of category airport_inside, '
      'category bar accounts for:',e_46/len(c_6m))
print('In the error object of category airport_inside, '
      'category bedroom accounts for:',e_56/len(c_6m))
print('In the error object of category airport_inside, '
      'category bookstore accounts for:',e_66/len(c_6m))
print('In the error object of category airport_inside, '
      'category bowling accounts for:',e_76/len(c_6m))
print('In the error object of category airport_inside, '
      'category buffet accounts for:',e_86/len(c_6m))
print('In the error object of category airport_inside, '
      'category casino accounts for:',e_96/len(c_6m))
print('The error rate of category artstudio is:',e_7/len(Rig))
print('In the error object of category airport_inside, '
      'category airport_inside accounts for:',e_07/len(c_7m))
print('In the error object of category airport_inside, '
      'category artstudio accounts for:',e_17/len(c_7m))
print('In the error object of category airport_inside, '
      'category auditorium accounts for:',e_27/len(c_7m))
print('In the error object of category airport_inside, '
      'category bakery accounts for:',e_37/len(c_7m))
print('In the error object of category airport_inside, '
      'category bar accounts for:',e_47/len(c_7m))
print('In the error object of category airport_inside, '
      'category bedroom accounts for:',e_57/len(c_7m))
print('In the error object of category airport_inside, '
      'category bookstore accounts for:',e_67/len(c_7m))
print('In the error object of category airport_inside, '
      'category bowling accounts for:',e_77/len(c_7m))
print('In the error object of category airport_inside, '
      'category buffet accounts for:',e_87/len(c_7m))
print('In the error object of category airport_inside, '
      'category casino accounts for:',e_97/len(c_7m))
print('The error rate of category artstudio is:',e_8/len(Rig))
print('In the error object of category airport_inside, '
      'category airport_inside accounts for:',e_08/len(c_8m))
print('In the error object of category airport_inside, '
      'category artstudio accounts for:',e_18/len(c_8m))
print('In the error object of category airport_inside, '
      'category auditorium accounts for:',e_28/len(c_8m))
print('In the error object of category airport_inside, '
      'category bakery accounts for:',e_38/len(c_8m))
print('In the error object of category airport_inside, '
      'category bar accounts for:',e_48/len(c_8m))
print('In the error object of category airport_inside, '
      'category bedroom accounts for:',e_58/len(c_8m))
print('In the error object of category airport_inside, '
      'category bookstore accounts for:',e_68/len(c_8m))
print('In the error object of category airport_inside, '
      'category bowling accounts for:',e_78/len(c_8m))
print('In the error object of category airport_inside, '
      'category buffet accounts for:',e_88/len(c_8m))
print('In the error object of category airport_inside, '
      'category casino accounts for:',e_98/len(c_8m))
print('The error rate of category artstudio is:',e_9/len(Rig))
print('In the error object of category airport_inside, '
      'category airport_inside accounts for:',e_09/len(c_9m))
print('In the error object of category airport_inside, '
      'category artstudio accounts for:',e_19/len(c_9m))
print('In the error object of category airport_inside, '
      'category auditorium accounts for:',e_29/len(c_9m))
print('In the error object of category airport_inside, '
      'category bakery accounts for:',e_39/len(c_9m))
print('In the error object of category airport_inside, '
      'category bar accounts for:',e_49/len(c_9m))
print('In the error object of category airport_inside, '
      'category bedroom accounts for:',e_59/len(c_9m))
print('In the error object of category airport_inside, '
      'category bookstore accounts for:',e_69/len(c_9m))
print('In the error object of category airport_inside, '
      'category bowling accounts for:',e_79/len(c_9m))
print('In the error object of category airport_inside, '
      'category buffet accounts for:',e_89/len(c_9m))
print('In the error object of category airport_inside, '
      'category casino accounts for:',e_99/len(c_9m))

