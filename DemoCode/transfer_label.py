# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 02:25:30 2020

@author: hanxi
"""

data_name =  ['airport_inside', 'airport_inside', 'airport_inside', 'airport_inside', 'artstudio', 
       'artstudio', 'artstudio', 'artstudio', 'artstudio', 'artstudio', 'artstudio', 'artstudio', 'artstudio', 'artstudio', 'artstudio', 
       'auditorium', 'auditorium', 'auditorium', 'auditorium', 'auditorium', 'auditorium', 'auditorium', 'auditorium', 'auditorium', 'auditorium', 
       'auditorium', 'auditorium', 'auditorium', 'auditorium', 'auditorium', 'auditorium', 'auditorium', 'auditorium', 'auditorium', 'auditorium',
       'auditorium', 'auditorium', 'auditorium', 'auditorium', 'auditorium', 'auditorium', 'auditorium', 'auditorium', 'auditorium', 'auditorium',
       'auditorium', 'auditorium', 'auditorium', 'auditorium', 'auditorium', 'auditorium', 'auditorium', 'auditorium', 'auditorium', 'auditorium',
       'auditorium', 'auditorium', 'auditorium', 'auditorium', 'auditorium', 'auditorium', 'auditorium', 'auditorium', 'auditorium', 'auditorium',
       'auditorium', 'auditorium', 'bakery', 'bakery', 'bakery', 'bakery', 'bakery', 'bakery', 'bakery', 'bakery', 'bakery', 'bakery', 'bakery',
       'bakery', 'bakery', 'bakery', 'bakery', 'bakery', 'bakery', 'bakery', 'bakery', 'bakery', 'bakery', 'bakery', 'bakery', 'bakery', 'bakery',
       'bakery', 'bakery', 'bakery', 'bakery', 'bakery', 'bakery', 'bakery', 'bakery', 'bar', 'bar', 'bar', 'bar', 'bar', 'bar', 'bar', 'bar', 'bar',
       'bedroom', 'bedroom', 'bedroom', 'bedroom', 'bedroom', 'bedroom', 'bedroom', 'bookstore', 'bookstore', 'bookstore', 'bookstore', 'bookstore', 
       'bookstore', 'bookstore', 'bowling', 'bowling', 'bowling', 'bowling', 'bowling', 'bowling', 'bowling', 'bowling', 'bowling', 'bowling', 
       'buffet', 'buffet', 'buffet', 'buffet', 'buffet', 'buffet', 'casino', 'casino', 'casino', 'casino', 'casino', 'casino', 'casino', 'casino',
       'casino', 'casino', 'casino', 'casino', 'casino', 'casino', 'casino', 'casino', 'casino', 'casino', 'casino', 'casino', 'casino', 'casino',
       'casino']
data_name_num =  ['airport_inside', 'airport_inside', 'airport_inside', 'airport_inside', 'artstudio', 
       'artstudio', 'artstudio', 'artstudio', 'artstudio', 'artstudio', 'artstudio', 'artstudio', 'artstudio', 'artstudio', 'artstudio', 
       'auditorium', 'auditorium', 'auditorium', 'auditorium', 'auditorium', 'auditorium', 'auditorium', 'auditorium', 'auditorium', 'auditorium', 
       'auditorium', 'auditorium', 'auditorium', 'auditorium', 'auditorium', 'auditorium', 'auditorium', 'auditorium', 'auditorium', 'auditorium',
       'auditorium', 'auditorium', 'auditorium', 'auditorium', 'auditorium', 'auditorium', 'auditorium', 'auditorium', 'auditorium', 'auditorium',
       'auditorium', 'auditorium', 'auditorium', 'auditorium', 'auditorium', 'auditorium', 'auditorium', 'auditorium', 'auditorium', 'auditorium',
       'auditorium', 'auditorium', 'auditorium', 'auditorium', 'auditorium', 'auditorium', 'auditorium', 'auditorium', 'auditorium', 'auditorium',
       'auditorium', 'auditorium', 'bakery', 'bakery', 'bakery', 'bakery', 'bakery', 'bakery', 'bakery', 'bakery', 'bakery', 'bakery', 'bakery',
       'bakery', 'bakery', 'bakery', 'bakery', 'bakery', 'bakery', 'bakery', 'bakery', 'bakery', 'bakery', 'bakery', 'bakery', 'bakery', 'bakery',
       'bakery', 'bakery', 'bakery', 'bakery', 'bakery', 'bakery', 'bakery', 'bakery', 'bar', 'bar', 'bar', 'bar', 'bar', 'bar', 'bar', 'bar', 'bar',
       'bedroom', 'bedroom', 'bedroom', 'bedroom', 'bedroom', 'bedroom', 'bedroom', 'bookstore', 'bookstore', 'bookstore', 'bookstore', 'bookstore', 
       'bookstore', 'bookstore', 'bowling', 'bowling', 'bowling', 'bowling', 'bowling', 'bowling', 'bowling', 'bowling', 'bowling', 'bowling', 
       'buffet', 'buffet', 'buffet', 'buffet', 'buffet', 'buffet', 'casino', 'casino', 'casino', 'casino', 'casino', 'casino', 'casino', 'casino',
       'casino', 'casino', 'casino', 'casino', 'casino', 'casino', 'casino', 'casino', 'casino', 'casino', 'casino', 'casino', 'casino', 'casino',
       'casino']

Mis = ['bar', 'artstudio', 'bowling', 'artstudio', 'bookstore', 'bedroom', 'airport_inside', 'bookstore', 'bookstore', 'bedroom', 'bar', 'buffet', 
       'bookstore', 'airport_inside', 'buffet', 'buffet', 'airport_inside', 'bookstore', 'bookstore', 'bar', 'airport_inside', 'bar', 'airport_inside', 
       'artstudio', 'casino', 'buffet', 'buffet', 'bedroom', 'casino', 'airport_inside', 'bowling', 'casino', 'airport_inside', 'airport_inside', 'bar', 'casino', 'bar', 'buffet', 'artstudio', 'bookstore', 'casino', 'bakery', 'bookstore', 'bowling', 'bedroom', 'bowling', 'bar', 'bar', 'bakery', 'bookstore', 'buffet', 'bakery', 'bakery', 'artstudio', 'airport_inside', 'artstudio', 'airport_inside', 'bar', 'bar', 'artstudio', 'artstudio', 'airport_inside', 'bakery', 'bar', 'casino', 'bookstore', 'bar', 'bookstore', 'buffet', 'buffet', 'buffet', 'casino', 'airport_inside', 'bookstore', 'airport_inside', 'auditorium', 'airport_inside', 'bowling', 'buffet', 'buffet', 'bar', 'buffet', 'artstudio', 'airport_inside', 'artstudio', 'artstudio', 'bowling', 'bar', 'auditorium', 'bookstore', 'airport_inside', 'bar', 'airport_inside', 'bar', 'bookstore', 'bookstore', 'airport_inside', 'airport_inside', 'bar', 'bowling', 'airport_inside', 'casino', 'artstudio', 'bookstore', 'artstudio', 'airport_inside', 'casino', 'bookstore', 'casino', 'artstudio', 'airport_inside', 'airport_inside', 'airport_inside', 'bowling', 'auditorium', 'bar', 'bar', 'buffet', 'bedroom', 'bakery', 'bar', 'bar', 'artstudio', 'artstudio', 'airport_inside', 'bookstore', 'airport_inside', 'artstudio', 'artstudio', 'airport_inside', 'auditorium', 'airport_inside', 'auditorium', 'bowling', 'airport_inside', 'bakery', 'casino', 'artstudio', 'bedroom', 'airport_inside', 'artstudio', 'artstudio', 'bakery', 'bedroom', 'bedroom', 'bookstore', 'bookstore', 'artstudio', 'bookstore', 'bar', 'airport_inside', 'bakery', 'bookstore', 'bakery', 'artstudio', 'bookstore', 'airport_inside', 'airport_inside', 'bookstore', 'bakery', 'bakery', 'bar']


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
        
for i in range(0,len(Mis)): 
    if Mis[i] == 'airport_inside':
        Mis[i] = 0
    if Mis[i] == 'artstudio':
        Mis[i] = 1
    if Mis[i] == 'auditorium':
        Mis[i] = 2
    if Mis[i] == 'bakery':
        Mis[i] = 3
    if Mis[i] == 'bar':
        Mis[i] = 4
    if Mis[i] == 'bedroom':
        Mis[i] = 5
    if Mis[i] == 'bookstore':
        Mis[i] = 6
    if Mis[i] == 'bowling':
        Mis[i] = 7
    if Mis[i] == 'buffet':
        Mis[i] = 8
    if Mis[i] == 'casino':
        Mis[i] = 9
print(data_name_num)
print(Mis)
print(len(data_name_num))
print(len(Mis))