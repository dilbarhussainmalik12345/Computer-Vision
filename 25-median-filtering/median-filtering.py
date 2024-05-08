import cv2 as cv 
import os 
import numpy as np 
import matplotlib.pyplot as plt 

def medianFiltering(): 
    root = os.getcwd() 
    imgPath = os.path.join(root,'demoImages//tesla.jpg')
    img = cv.imread(imgPath)
    imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)

    noisyImg = imgRGB.copy()
    noiseProb = 0.05
    noise = np.random.rand(noisyImg.shape[0], noisyImg.shape[1])
    noisyImg[noise < noiseProb / 2] = 0
    noisyImg[noise > 1 - noiseProb / 2] = 255
    
    '''
        medianBlue ksize needs to be odd         
    '''
    imgFilter = cv.medianBlur(noisyImg,5)

    plt.figure() 
    plt.subplot(121)
    plt.imshow(noisyImg)
    plt.subplot(122)
    plt.imshow(imgFilter)
    plt.show() 


if __name__ == '__main__': 
    medianFiltering() 