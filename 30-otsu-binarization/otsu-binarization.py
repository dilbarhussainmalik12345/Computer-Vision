import cv2 as cv 
import numpy as np
import matplotlib.pyplot as plt 
import os 


def otsuThresholding(): 
    root = os.getcwd()
    imgPath = os.path.join(root,'demoImages//tesla.jpg')
    imgGray = cv.imread(imgPath,cv.IMREAD_GRAYSCALE)
    
    plt.figure() 
    plt.subplot(131)
    plt.imshow(imgGray,cmap='gray')
    plt.title('gray')

    plt.subplot(132)
    thres = 70 # increase to about 110 to match otsu 
    maxVal = 255
    _,imgThres = cv.threshold(imgGray,thres,maxVal,cv.THRESH_BINARY)
    plt.imshow(imgThres,cmap='gray')
    plt.title('global threshold')

    plt.subplot(133)
    arbThres = 0 # unused when used with otsu 
    _,imgOtsu = cv.threshold(imgGray,arbThres,maxVal,cv.THRESH_BINARY+cv.THRESH_OTSU)
    plt.imshow(imgOtsu,cmap='gray')
    plt.title('otsu threshold')
    plt.show() 
    
    hist = cv.calcHist([imgGray], [0], None, [256], [0, 256])
    plt.figure()
    plt.plot(hist)
    plt.xlabel("Intensity")
    plt.ylabel("# of Pixels")
    plt.show()


if __name__ == '__main__': 
    otsuThresholding() 
