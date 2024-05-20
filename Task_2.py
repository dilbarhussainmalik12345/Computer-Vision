#!/usr/bin/env python
# coding: utf-8

# In[9]:


import cv2 as cv 
import os 


def grayscale(): 
    '''
        Y = 0.299 R + 0.587 G + 0.114 B
    '''
    root = os.getcwd()
    imgPath = os.path.join(root,'C:\\Users\\Asad Computrs\\Downloads\\opencv-python-tutorials-main\\opencv-python-tutorials-main\\demoImages\\cutepic1.jpg')
    img = cv.imread(imgPath)
    imgGray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    cv.imshow('gray',imgGray)
    cv.waitKey(0)

def readAsGray(): 
    root = os.getcwd()
    imgPath = os.path.join(root,'C:\\Users\\Asad Computrs\\Downloads\\opencv-python-tutorials-main\\opencv-python-tutorials-main\\demoImages\\cutepic1.jpg')
    img = cv.imread(imgPath,cv.IMREAD_GRAYSCALE)

    cv.imshow('gray',img)
    cv.waitKey(0)
    
if __name__ == '__main__': 
    # grayscale() 
    readAsGray() 


# In[ ]:




