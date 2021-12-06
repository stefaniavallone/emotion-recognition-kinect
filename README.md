# Face and Emotion Recognition with MS Kinect
This project aims to perform face detection and emotion recognition through a Kinect sensor.

This repository contains the code for:
 * data exploration and processing of the emotions dataset FER2013 (from Kaggle); 
 * training, fine-tuning and onnx conversion for emotion model;
 * data acquisition for face recognition;
 * face detection with OpenCV Face Cascade 
 * video acquisition and processing on kinect (inference phase)

## Table of contents
   * [Description](#description)
   * [Installation](#installation)
   * [Usage](#usage)

# Description
...
...
...


# Installation
After cloning or downloading the repository, we suggest you to create a virtual environment through the command `python3 -m venv venv` in the root directory. 
It's important as you need to use specific versions of the libraries used, for compatibility reasons.

## Dependencies
Once the venv has been created, install the dependencies with `pip install -r requirements.txt` in a venv enabled shell.

## Building Freenect and the python wrapper
The code can run also on a generic camera, but we used the Kinect to take advantage of the depth sensor to decrease false positives.
If you don't have a Kinect or you don't want to use the Kinect, you can skip this part.

### Drivers
We built it on Mac OS, based on the Homebrew method described at the following link: [https://openkinect.org/wiki/Getting_Started#OS_X](https://openkinect.org/wiki/Getting_Started#OS_X)
With Homebrew you can easily install the Kinect v1 drivers.

Notice you have also to build the python wrapper. 

### Python wrapper
Install the required dependencies (cython, numpy, python-dev, build-essentials)  .
Download the repository [here](https://github.com/OpenKinect/libfreenect/tree/master/wrappers/python/) and run `python3 install setup.py` in the wrapper directory.


# Usage
First, you need to set up the project and the resources. 
It implies obtaining and processing the dataset, training and evaluating the models through some scripts and then running the script that process the video stream from the camera.

The working directory is `src`.
In the `resources` folder there are the `data` and `models` subfolders, for the dataset and the saved models respectively.
Please follow our convention to avoid path errors.

We have configured the project 
Here are all the commands you can run:\
DATASET ACQUISITION
```bash
python face_recognition/dataset_acquisition.py --camera <SOURCE_CAM> --name "<YOUR_NAME>" --num_photos <PHOTOS_TO_CAPTURE>
```
TRAIN FACE RECOGNITION 
```bash
python face_recognition/dataset_acquisition.py --camera <SOURCE_CAM> --name "<YOUR_NAME>" --num_photos <PHOTOS_TO_CAPTURE>
```
TRAIN EMOTIONS RECOGNITION 
```bash
python emotions/train.py
```
OPTIMIZE EMOTIONS RECOGNITION 
```bash
python emotions/optimization.py 
```
EVALUATE EMOTIONS RECOGNITION 
```bash
python emotions/evaluate.py
```
RUN THE MODELS ON VIDEO CAMERA
```bash
python webcam.py --camera <SOURCE_CAM>
```


# Emotions
## Data exploration and processing


## Training and Evaluate


## Optimization


## ONNX
To speed up the inference phase, we setup up the ONNX conversion and runtime tools.
After choosing the best hyperparameters for the emotion model, we trained it and got an optimized Keras model.
So we converted it to a ONNX model.   

Generally speaking, the ONNX model version is much faster than the Keras one.
This leads a less power consumption and a higher FPS for our video application.

On our machines (CPU based, no NVIDIA), the ONNX model is around 25x faster than keras version.

These are the results:
```
Keras predictor 100 times  -  Elapsed: 3.177694320678711; mean: 0.03177694320678711
ONNX predictor  100 times  -  Elapsed: 0.119029283523559; mean: 0.00119029283523559
Factor: 26.696.
```
```
Keras predictor 10000 times  -  Elapsed: 317.5036771297455; mean: 0.03175036771297455
ONNX predictor  10000 times  -  Elapsed:  11.5271108150482; mean: 0.00115271108150482
Factor: 27.544.
```

# Face detection and Recognition
## Dataset acquisition
Run the following command to capture some photos for the training dataset:
```bash
python face_recognition/dataset_acquisition.py --camera <SOURCE_CAM> --name "<YOUR_NAME>" --num_photos <PHOTOS_TO_CAPTURE>
```

For example:
```bash
python face_recognition/dataset_acquisition.py --camera 0 --name "Stefania" --num_photos 30
```

## Training

