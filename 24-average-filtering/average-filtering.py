import cv2 as cv 
import os 
import numpy as np 
import matplotlib.pyplot as plt 

def callback(input):
    pass 

def averageFiltering(): 
    root = os.getcwd() 
    imgPath = os.path.join(root,'demoImages//tesla.jpg')
    img = cv.imread(imgPath)
    winName = 'avg filter'
    cv.namedWindow(winName)
    cv.createTrackbar('n',winName,1,100,callback)
    height,width,_ = img.shape
    scale = 1/4 
    width = int(width*scale)
    height = int(height*scale)
    img = cv.resize(img,(width,height))
    
    while True: 
        if cv.waitKey(1) == ord('q'): 
            break

        n = cv.getTrackbarPos('n',winName)
        imgFilter = cv.blur(img,(n,n))
        cv.imshow(winName,imgFilter)

  
    cv.destroyAllWindows() 


if __name__ == '__main__': 
    averageFiltering() 