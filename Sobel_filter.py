# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 19:33:15 2020

@author: micha
"""


# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 23:52:10 2020

@author: micha
"""
from scipy.signal import convolve2d
import tensorflow as tf
import imageio
import numpy as np
import matplotlib.pyplot as plt
img = imageio.imread('IMG-gray.jpg')
img_py=np.array(img)
Hx= np.array([[1, 0, -1], [2, 0, -2],[1, 0, -1]])
Hy= np.array([[-1, -2, -1], [0, 0, 0],[1, 2, 1]])
convolvedGx= convolve2d(img_py, Hx)
convolvedGy= convolve2d(img_py, Hy)
tot= (convolvedGx*convolvedGx + convolvedGy*convolvedGy)**0.5
plt.imshow(tot, cmap='gray')
imageio.imwrite('lines.jpg', tot)