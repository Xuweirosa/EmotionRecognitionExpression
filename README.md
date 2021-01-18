# Emotion Recognition and Expression on Pepper

## Introduction

This file shows the requirement for course project Emotion Recognition and Expression on Pepper for AU332, SJTU, accomplished by student Xu Wei and Piao Caiyong.

## Dependencies
* Linux or Mac OS
* Python 2.7, [OpenCV](https://opencv.org/), [Tensorflow](https://www.tensorflow.org/)
* Choregraphe 2.5.5
* Python SDK for NAOqi


## Model Document

For facial expression recognition, the model document for the CNN model is `model2.h5`

The file of front face recognition model offered by openCV is stored in `haarcascade_frontalface_default.xml`

For body emotion recognition, the model document for the CNN model is `model.h5`

## Dataset

```
For facial expression recoginition, the dataset is FER-2013 from Kaggle.
```
```
For body emotion recognition, the dataset is based on e Leeds Sports Pose Dataset.
```


## Files included
* `emotions.py` is the main code
* `model.h5`, `model2.h5`, `haarcascade_frontalface_default.xml` is the model document
* folder `model_training` includes two DCNN model code and corresponding dataset
* `teach.py` is the code to teach Pepper move
* `act.py` is the actions designed for Pepper robot
* The pdf file is report of our project.
