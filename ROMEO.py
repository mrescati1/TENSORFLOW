# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 23:38:13 2020

@author: micha
"""


from scipy.signal import convolve2d
import tensorflow as tf
import imageio
import numpy as np
def romeo(ar):
    new_ar=np.zeros((ar.shape[0], ar.shape[1], ar.shape[2]))
    for i in range(0, ar.shape[0]):
        for j in range(0, ar.shape[1]):
            if ar[i][j][0]< 160 and ar[i][j][1]< 105  and ar[i][j][2]> 95 and ar[i][j][2]< 110:
                new_ar[i][j]= ar[i][j]
    return new_ar
    
img = imageio.imread('IMG_0577.jpg')
print(img[1200][1600])
img_py=np.array(img)
imageio.imwrite('IMG-Romeo.jpg', romeo(img_py))