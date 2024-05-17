import cv2 as cv 
import os 
import numpy as np 
import matplotlib.pyplot as plt 

def convolution2d(): 
    root = os.getcwd() 
    imgPath = os.path.join(root,'demoImages//tesla.jpg')
    img = cv.imread(imgPath)
    imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    # imgRGB = imgRGB[0:100,0:100,:]
    n = 100
    kernel = np.ones((n,n),np.float32)/(n*n)
    imgFilter = cv.filter2D(imgRGB,-1,kernel)

    plt.figure() 
    plt.subplot(121)
    plt.imshow(imgRGB)

    plt.subplot(122)
    plt.imshow(imgFilter)

    plt.show() 


if __name__ == '__main__': 
    convolution2d() 