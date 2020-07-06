# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 23:52:10 2020

@author: micha
"""
from scipy.signal import convolve2d
import tensorflow as tf
import imageio
import numpy as np
img = imageio.imread('IMG-gray.jpg')
img_py=np.array(img)
convMatrix= np.ones((8,8))/8**2
convolved= convolve2d(img_py, convMatrix)
imageio.imwrite('IMG-sgranata.jpg', convolved)