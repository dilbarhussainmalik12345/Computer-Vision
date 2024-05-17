import cv2 as cv 
import matplotlib.pyplot as plt 
import numpy as np
import os 

def callback(input): 
    pass 

def cannyEdge(): 
    root = os.getcwd() 
    imgPath = os.path.join(root,'demoImages//tesla.jpg')
    img = cv.imread(imgPath)
    img = cv.cvtColor(img,cv.COLOR_BGR2RGB)

    height,width,_ = img.shape
    scale = 1/5
    heightScale = int(height*scale)
    widthScale = int(width*scale)
    img = cv.resize(img,(widthScale,heightScale),interpolation=cv.INTER_LINEAR)

    winname = 'canny'
    cv.namedWindow(winname)
    cv.createTrackbar('minThres',winname,0,255,callback)
    cv.createTrackbar('maxThres',winname,0,255,callback)

    while True: 
        if cv.waitKey(1) == ord('q'): 
            break

        minThres = cv.getTrackbarPos('minThres',winname)
        maxThres = cv.getTrackbarPos('maxThres',winname)
        cannyEdge = cv.Canny(img,minThres,maxThres)

        cv.imshow(winname,cannyEdge)

    cv.destroyAllWindows()

if __name__ == '__main__': 
    cannyEdge() 
