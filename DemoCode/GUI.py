# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 06:17:48 2020

@author: hanxi
"""
from skimage import io, transform

import tensorflow as tf
import numpy as np
from tkinter import *

from PIL import Image, ImageTk


 
root = Tk()

frame1 = Frame(root)
frame2 = Frame(root)
frame3 = Frame(root)
root.title("Indoor Scene Recognition")
def Pred(*arg):
    scenes_dict = {0: 'airport_inside', 1: 'artstudio', 2: 'auditorium', 3: 'bakery', 4: 'bar',
               5: 'bedroom', 6: 'bookstore', 7: 'bowling', 8: 'buffet', 9: 'casino'}
#字典存放类名

    w = 100
    h = 100
    c = 3
    def read_one_image(*arg):
        img = io.imread(*arg)
        img = transform.resize(img, (w, h))
        return np.asarray(img)
    
    
    with tf.Session() as sess:
        data = []
        data1 = read_one_image(*arg)

        data.append(data1)

    
        saver = tf.train.import_meta_graph('./scenes_model.ckpt.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./'))
    
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("x:0")
        feed_dict = {x: data}
    
        logits = graph.get_tensor_by_name("logits_eval:0")
    
        classification_result = sess.run(logits, feed_dict)
    

        output = []
        output = tf.argmax(classification_result, 1).eval()
        f = open("out.txt", "w")    
        
        for i in range(len(output)):
            print("The", i + 1, "st image predicted result:" + scenes_dict[output[i]],file=f)
        f.close  
        
def openfile():
    filename = str_te
    try:
        with open(filename) as f:
            for each_line in f:
                text.insert(INSERT,each_line)                
    except OSError as reason:
        print('The file does not exist！\n Please re-enter'+str(reason))
        
bottomFrame = Frame(root,bd=1, relief=SUNKEN)

str_te = 'out.txt'
# str_im = 'airport_inside_0001.jpg'
str_im = 'casino_0025.jpg'
label2=Label(frame2,text="Getting Predicted Result:")
label2.grid(row=0,column=0, sticky=W)

v1=StringVar()
# e1=Entry(frame2,textvariable=v1)
# e1.grid(row=0,column=1,sticky=W)

b2=Button(frame2,text="OK_Get Result",command=openfile,width=15,padx=1)
b2.grid(row=0,column=2,padx=5,sticky=W)

     

img = Image.open(str_im)  

photo = ImageTk.PhotoImage(img)
label_img = Label(frame1, image = photo)
label_img.grid(row=0,column=0,padx=5,sticky=W)


label1= Label(frame1,text="Press OK to predict the image",justify=LEFT)
label1.grid(row=1,column=0, sticky=W)

b1=Button(frame1,text="OK_Predict",command=Pred(str_im),width=15,padx=1)
b1.grid(row=1,column=2,padx=5,sticky=W)
# b1.pack(side=LEFT)
# label_img.pack(side=LEFT)
# label.pack(side=LEFT)
 
text = Text(bottomFrame, height=5)
text.pack(fill=BOTH)
 
frame1.pack(padx=1,pady=1)
frame2.pack(padx=10,pady=10)
bottomFrame.pack(fill=BOTH)

root.mainloop()