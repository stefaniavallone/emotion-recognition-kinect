# Face and Emotion Recognition with MS Kinect
This project aims to perform face and emotion recognition through a Kinect sensor or a generic webcam. 

The emotion supported are: "Happy", "Sad", "Disgust", "Neutral", "Fear", "Angry", "Surprise". The depth camera of the Kinect Sensor is used to perform Depth Segmentation and reduce face detection false positives. In addition, a confirmation window is used to stabilize the model's predictions. 

[GIF]

## Table of contents
   * [Description](#description)
   * [Installation](#installation)
   * [Usage](#usage)
      * [Commands](#commands)
         * []
   * [Machine Learning Process](#mlprocess)
      * [Emotions](#emotions)
         * [Data Exploratory analysis](#data-esploratory-analysis)
         * [Training and evaluation](#training-and-evaluation)
         * [Optimization phase](#training)
         * [ONNX and Inference phase](#onnx-and-inference-phase)
         * [Improvements for real use]()
      * [Face Recognition](#face-recognition)
         * [Face acquisition](#face-acquisition)
         * [Training phase](#training)
         * [Inference phase](#inference)

# Description
The project involves the use of different machine learning models:
- face detection, performed with OpenCV's Cascade Classifier;
- face recognition, done with OpenCV's LBPHFaceRecognizer;
- emotion recognition, performed with a CNN built from scratch.

We decided to use only FER2013 Dataset and a CNN to understand the difficulties around this task and we tried to improve the accuracy with hyperparameter optimization. Furthermore, we added some CV techniques (confirmation window, depth segmentation) to improve real execution performance.


This repository contains the code for:
 * data exploration and processing of the emotions dataset FER2013 (from Kaggle); 
 * training, fine-tuning and onnx conversion for emotion model;
 * face detection with OpenCV Face Cascade Classifier;
 * data acquisition for face recognition;
 * face recognition with OpenCV LBPHFaceRecognizer; 
 * real-time video inference on webcam / kinect.

# Installation
After cloning or downloading the repository, we suggest you to create a virtual environment through the command `python3 -m venv venv` in the root directory. 
It's important as you need to use specific versions of the libraries used, for compatibility reasons.

## Dependencies
Once the venv has been created, install the dependencies with `pip install -r requirements.txt` in a venv enabled shell.

## Building Freenect and the python wrapper
The code can run also on a generic camera, but we used the Kinect to take advantage of the depth sensor to decrease false positives.
If you don't have a Kinect or you don't want to use the Kinect, you can skip this part.

### Drivers
We built it on Mac OS, based on the Homebrew method described at the following link: [https://openkinect.org/wiki/Getting_Started#OS_X](https://openkinect.org/wiki/Getting_Started#OS_X). With Homebrew you can easily install the Kinect v1 drivers.
Notice you have also to build the python wrapper. 

### Python wrapper
Install the required dependencies (cython, numpy, python-dev, build-essentials).
Download the repository [here](https://github.com/OpenKinect/libfreenect/tree/master/wrappers/python/) and run `python3 install setup.py` in the wrapper directory.


# Usage
Dataset and models are not provided for privacy reasons. So, to run the project you first need to set up the project and resources. 
It implies obtaining and processing the dataset, training and evaluating the models through some scripts and then running the script that process the video stream from the camera/Kinect.

> **NOTE**: The working directory is `src`.
In the `resources` folder there are the `data` and `models` subfolders, for the dataset and the saved models respectively.
Please follow our convention to avoid path errors.

```
resources
  |- data
     |- emotions
     |- faces 
  |- models
     |- emotions
     |- faces
```


## Commands
Here are all the commands you can run:

### Dataset / Face acquisition
This script can be used to collect the photos for training the OpenCV LBPHFaceRecognizer with your faces. You can run this script several times to get photo of more people in different brightness/camera/ambient situations.
Use `--camera` parameter to choose between more cameras on your pc. The default is 0. Use `--num_photos` to choose the number of photos you want to take. Default is 30. Note that 30 photo could be not sufficient for a good training. Finally, `--name` option is use to specify the label the face is associated with.
```bash
python face_acquisition.py --camera <SOURCE_CAM> --name "<YOUR_NAME>" --num_photos <PHOTOS_TO_CAPTURE>
```

### Train face recognition 

```bash
python face_recognition/dataset_acquisition.py --camera <SOURCE_CAM> --name "<YOUR_NAME>" --num_photos <PHOTOS_TO_CAPTURE>
```

### Train emotion recognition
You need to get the dataset from [Kaggle](https://www.kaggle.com/deadskull7/fer2013) and process it with the jupyter notebook situated in `src/emotions/dataset_analysis.ipynb`.
```bash
python train_emotions.py --train_dir <TRAIN_DIR_PATH> --test_dir <TEST_DIR_PATH> --epochs <NUM_EPOCHS> --batch_size <BATCH_SIZE> --lr <LEARNING_RATE> --decay <LR_DECAY> --dropout <DROPOUT> --l2 <LAMBDA_L2_REGULARIZATION> --kernel <KERNEL_SIZE> 
```

### Optimize emotion recognition
```bash
python optimize_emotions.py --train_dir <TRAIN_DIR_PATH> --test_dir <TEST_DIR_PATH> --n_trials <NUM_OF_EXPERIMENTS> 
```


### Evaluate emotion recognition
```bash
python evaluate_emotions.py --model_path <MODEL_PATH> --test_dir <TEST_DIR_PATH>
```

### Run the models on Camera or Kinect
The script allows you to choose between the several webcams on your PC or kinect through the options `--camera` and `--kinect`. If no option is specified, camera=0 is chosen. It is also required to specify the path of the trained models. The defaults are:  `../resources/models/emotions/emotion` (Tensorflow format) for the emotion model and `../resources/models/faces/faces.yaml`, `../resources/models/faces/faces_labels.bin` for the faces one. Note that the working directory has to be `src` and that the labels file for the faces has to have the `_labels` suffix. 
```bash
python display.py --camera <SOURCE_CAM> --emotion_path <SAVED_EMOTION_MODEL_PATH> --face_path <SAVED_FACES_MODEL_PATH>
python display.py --kinect True --emotion_path <SAVED_EMOTION_MODEL_PATH> --face_path <SAVED_FACES_MODEL_PATH>
```

# Machine Learning Process
In this section, we describe the step we followed to train the two models and use them on a video stream capture from a Camera / Kinect.

## Emotions
### Data exploration and processing
We used FER2013 dataset. As a first step, we analyzed the dataset to verify its distribution and verify its content, displaying some images.

We noticed that it is a very difficult dataset, since it has 3 main issues:
1. It is very unbalanced; 
2. The images it contains are dirty (some images are cartoons) or have noise (for example writing or hands on the face);
3. Some images could be misclassified (e.g. some emotion tagged as "fear" could be classified as "surprise"). 

In fact, as mentioned in [this paper (sec. III)](http://cs230.stanford.edu/projects_winter_2020/reports/32610274.pdf), the human accuracy on this dataset is about 65%.

The dataset is already divided into train, validation and test and we made sure that the three sets had the same distribution.
We then decided to merge the train and validation sets together, in order to be free in choosing the percentage to be allocated to the validation set.

Finally, we decided not to use oversampling techniques and to use the as-is dataset, to try to get the most out of the CNN network and the training process, without adding new data.

### Training and Evaluation
We used a CNN built from scratch, consisting of four convolution layers and two dense layers. We verified that the model was sufficiently powerful and we managed overfitting with regularization techniques such as l2 and dropout.

We split the dataset into 70% training, 20% for validation and 10% for testing, respectively. In the training phase, we used Keras' `ImageDataGenerator` utilities to do some data augmentation and add zoomed and horizontally mirrored images. This allowed us to increase the test phase performance a bit.


In the evaluation phase, we plotted a normalized confusion matrix to understand the performance of the model relative to each class. The results were what we expected: a strong polarization towards the most populous classes and an average error spread across all classes. In fact, the performance mirrors the challenges of the dataset.

### Optimization
After deciding that the chosen model had performances in line with what was expected, we moved on to the optimization phase, to obtain the best possible hyperparameters and therefore a higher accuracy.
We used optuna, an optimization framework specifically designed for this task. 

The hyperparameters we have optimized are:
```python
   'lr': learning rate, uniform distribution from 1e-5 to 1e-1,
   'decay': lr decay, uniform distribution from 1e-7 to 1e-4,
   'dropout': uniform distribution from 0.10 to 0.50,
   'l2': l2 regularization in Conv2D layer, uniform distribution from 0.01 to 0.05,
   'kernel_size': for Conv2D layer in range [3, 5],
   'batch_size': in range [16, 32, 64, 128],
```
The best hyperparameters, found in 50 iterations, are:
```python
   'lr': 6.454516989719096e-05,
   'decay': 4.461966074951546e-05,
   'dropout': 0.3106791934814161,
   'l2': 0.04370766874155845,
   'kernel_size': 2,
   'batch_size': 32,
```

with a test loss of: `X.XXX`, and test accuracy of: `66.035%`. 

### ONNX and Inference
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

### Improvements for real use
Given the poor performance of the model and considered the fact that the images acquired by a webcam may be "different" from those present in the dataset, we decided to add some "improvements" in real use.
We first introduced a **confirmation window**, to stabilize the model predictions over a higher frame number.
The confirmation window, implemented as a queue, collects the emotions of the last 20 frames and returns the value only if the dominant emotion is present in more than 60% of the frames. If this condition is not satisfied, "Neutral" is returned, indicating that the model is not fully convinced of the emotions collected in the last 20 frames.

A confirmation window is associated for each recognized person, so as not to mix predictions related to different faces.
**The benefit this implementation has brought is that the feeling that the model is wrong has drastically reduced**.

We also took advantage of the Kinect's depth camera to segment and filter objects based on depth. This allows us to reduce the number of false positives of face detection. Since there is no rejection threshold for face recognition, limiting false positives also allows the predictions contained in the confirmation window to be "not dirty", thus resulting in a reliable prediction.

## Face detection and Recognition
### Dataset acquisition
Run the following command to capture some photos for the training dataset:
```bash
python face_recognition/dataset_acquisition.py --camera <SOURCE_CAM> --name "<YOUR_NAME>" --num_photos <PHOTOS_TO_CAPTURE>
```

For example:
```bash
python face_recognition/dataset_acquisition.py --camera 0 --name "Stefania" --num_photos 30
```

### Training

### Inference phase
