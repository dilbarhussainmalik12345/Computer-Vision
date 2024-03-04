## CATS vs DOGS Classification using Convolutional Neural Networks and Data Augmentation<br>
<b>Dataset Details:</b><Br>
  you can download dataset from <a href = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip" target="_blank">google apis</a>.<br>

<b>Dataset Description:</b><br>
Dataset contain 3000 images of Cats and Dogs,
we will train our model on 1700 images,710 images for validation and 604 images for testing.<br><br>
Training Images of cats = 850<br>
Training Images of dogs = 850<br>

Validation Images of Cats = 352<br>
Validation Images of Dogs = 358<br>

Testing Images of Cats = 304<br>
Testing Images of Dogs = 300<br>

<b>Overfiting and Underfitting aviodence Techniques Used:</b><br>
1-Data Augmentation (zoom,horizontal_flip,rotation)<br>
2-Dropout<br>

<b>Model Summary:</b><br>
I used convolutional neural networks with 32, 64 and 128 layers.<Br>
<img src = "/Other-images/seq.jpg"><br><br>
<b>Training and Validation Graph:</b><br>
<img src = "/Other-images/training.png"><br>

<b>Results:</b><br>
Achieved 84% Accuracy on Training data with <b><i>epochs = 100</i></b><br>
81% accuracy on validation data<br>
80% accuracy on testing data<br>
