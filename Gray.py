from scipy.signal import convolve2d
import tensorflow as tf
import imageio
import numpy as np
def gray(ar):
    new_ar=np.zeros((ar.shape[0], ar.shape[1]))
    for i in range(0, ar.shape[0]):
        for j in range(0, ar.shape[1]):
            new_ar[i][j]= np.mean(ar[i][j])
    return new_ar
    
img = imageio.imread('IMG_0104.jpg')
print(img[0][0])
img_py=np.array(img)
imageio.imwrite('IMG-gray.jpg', gray(img_py))