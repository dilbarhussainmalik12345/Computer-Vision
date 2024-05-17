import cv2 as cv 
import numpy as np
import matplotlib.pyplot as plt 
import os 


def morphTrans(): 
    root = os.getcwd()
    imgPath = os.path.join(root,'demoImages//tesla.jpg')
    imgGray = cv.imread(imgPath,cv.IMREAD_GRAYSCALE)
    # imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)

    maxValue = 255
    blockSize = 7 # odd 
    offsetC = 3 

    plt.subplot(241)
    imgGaus = cv.adaptiveThreshold(imgGray,maxValue,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,blockSize,offsetC)
    imgGaus = cv.GaussianBlur(imgGaus,(7,7),sigmaX=2)
    plt.imshow(imgGaus,cmap='gray')
    plt.title('gaus threshold')
    
    kernel = np.ones((7,7),np.uint8)

    # notice tesla outline is thicker (expand dark region)
    erosion = cv.erode(imgGaus,kernel,iterations=1)
    plt.subplot(242)
    plt.imshow(erosion,cmap='gray')
    plt.title('erosion')

    # notice tesla outline is thinner (expand white region)
    dilation = cv.dilate(imgGaus,kernel,iterations=1)
    plt.subplot(243)
    plt.imshow(dilation,cmap='gray')
    plt.title('dilation')

    morphTypes = [cv.MORPH_OPEN,     # dilate(erode)
                  cv.MORPH_CLOSE,    # erode(dilate)
                  cv.MORPH_GRADIENT, # dilate - erode 
                  cv.MORPH_TOPHAT,   # src - open 
                  cv.MORPH_BLACKHAT] # close - src 
    morphTitles = ['open','close','gradient','tophat','blackhat']

    for i in range(len(morphTypes)): 
        plt.subplot(2,4,i+4)
        plt.imshow(cv.morphologyEx(imgGaus,morphTypes[i],kernel),cmap='gray')
        plt.title(morphTitles[i])

    plt.show() 

    #creating custom kernels 
    ellipseKernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
    crossKernel = cv.getStructuringElement(cv.MORPH_CROSS,(5,5))
    print(ellipseKernel)
    print(crossKernel)

if __name__ == '__main__': 
    morphTrans() 
