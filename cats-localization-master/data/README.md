
This folder contains the images and annotations to train the model.

## Classes

* Blacky: 
    * Train: 51 images
    * Validation: 21 images
* Niche:
    * Train: 47 images
    * Validation: 22 images

## <em>data</em> folder structure:

```
data/
    train/
        Blacky/
            imgs.jpg
            ...
        Niche/
            imgs.jpg
            ...
        cats-annotations.json
    validation/
        Blacky/
            imgs.jpg
            ...
        Niche/
            imgs.jpg
            ...
        cats-annotations.json
```

The <em>cats-annotations.json</em> file in <em>train</em> folder has annotations for the training images. The file in <em>validation</em> folder has the annotations for the validation images.

## Dataset
https://drive.google.com/open?id=1o9zyd51QCqWG3DlArQnfH4q4clMVQmTY (zip file ~39 MB)

## Tools
Annotation tool: [VIA](http://www.robots.ox.ac.uk/~vgg/software/via/)