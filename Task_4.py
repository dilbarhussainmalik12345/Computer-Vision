#!/usr/bin/env python
# coding: utf-8

# In[9]:


import cv2 as cv
import os


# In[10]:


def grayscale():
    """
    Converts an image to grayscale and displays it.
    """
    root = os.getcwd()
    imgPath = os.path.join(root, 'C:\\Users\\Asad Computrs\\Downloads\\opencv-python-tutorials-main\\opencv-python-tutorials-main\\demoImages\\cutepic1.jpg')
    img = cv.imread(imgPath)
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    cv.imshow('Gray', imgGray)
    cv.waitKey(0)


# In[11]:


def readAsGray():

    root = os.getcwd()
    imgPath = os.path.join(root, 'C:\\Users\\Asad Computrs\\Downloads\\opencv-python-tutorials-main\\opencv-python-tutorials-main\\demoImages\\cutepic1.jpg')
    img = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)

    cv.imshow('Gray', img)
    cv.waitKey(0)


# In[12]:


def extract_features(image_path):

    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return
    
    img = cv.imread(image_path)
    if img is None:
        print("Error: Image could not be read.")
        return
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    img_with_keypoints = cv.drawKeypoints(img, keypoints, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    
    cv.imshow('SIFT Keypoints', img_with_keypoints)
    cv.waitKey(0)
    cv.destroyAllWindows()
    print(f"SIFT keypoints detected: {len(keypoints)}")

if __name__ == '__main__':
    
    img_path = 'C:\\Users\\Asad Computrs\\Downloads\\opencv-python-tutorials-main\\opencv-python-tutorials-main\\demoImages\\cutepic1.jpg'
    

    extract_features(img_path)


# In[ ]:




