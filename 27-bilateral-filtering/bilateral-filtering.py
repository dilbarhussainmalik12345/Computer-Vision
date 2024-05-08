import cv2 as cv 
import os 
import numpy as np 
import matplotlib.pyplot as plt 

def medianFiltering(): 
    root = os.getcwd() 
    imgPath = os.path.join(root,'demoImages//tesla.jpg')
    img = cv.imread(imgPath)
    imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    height,width,_ = imgRGB.shape
    scale = 1/4 
    width = int(width*scale)
    height = int(height*scale)
    imgRGB = cv.resize(imgRGB,(width,height))
    
    '''
        d - diameter of pixel neighborhood (5 or 9 for real time)
        sigmaColor - controls how much color change affect blurring
        sigmaSpace - controls how much distance change affect blurring

        can set sigmaColor and sigmaSpace to same value 
    '''
    imgFilter = cv.bilateralFilter(imgRGB,25,100,100)

    plt.figure() 
    plt.subplot(121)
    plt.imshow(imgRGB)
    plt.subplot(122)
    plt.imshow(imgFilter)
    plt.show() 


if __name__ == '__main__': 
    medianFiltering() 