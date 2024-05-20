#!/usr/bin/env python
# coding: utf-8

# In[9]:


import cv2 as cv
import os


# In[10]:


def show_image(image_path):
   
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return
    
   
    img = cv.imread(image_path)
    if img is None:
        print("Error: Image could not be read.")
        return
    
  
    cv.imshow('Original Image', img)
    
    
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    
    describe_image(image_path)


# In[11]:


def invert_colors(image_path):
    """
    Reads an image from the specified path, inverts its colors,
    and displays the inverted image.
    """
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return
    
    
    img = cv.imread(image_path)
    if img is None:
        print("Error: Image could not be read.")
        return
    
    
    img_inverted = cv.bitwise_not(img)
    
   
    cv.imshow('C:\\Users\\Asad Computrs\\Downloads\\opencv-python-tutorials-main\\opencv-python-tutorials-main\\demoImages\\cutepic1.jpg', img_inverted)
    
   
    cv.waitKey(0)
    cv.destroyAllWindows()


# In[12]:


def describe_image(image_path):
    
    print(f"The image at {image_path} is displayed in both original and grayscale formats.")
    print(" image likely contains a cute or attractive subject, given the filename 'cutepic1.jpg'.")

if __name__ == '__main__':
    img_path = 'C:\\Users\\Asad Computrs\\Downloads\\opencv-python-tutorials-main\\opencv-python-tutorials-main\\demoImages\\cutepic1.jpg'
    show_image(img_path)
    invert_colors(img_path)


# In[ ]:




