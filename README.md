# Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---

The goals of this project were the following:
* Use the simulator to collect data of good driving behavior
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator
* Summarize the results in this written report

## My approach to the project

At first, I wanted to manually discover the tracks that are present in the simulator. I drove around the two tracks and noticed the following:

- the **first** track is, in terms of driving difficulty, fairly basic: there are no very sharp turns and no steep ascents or descents. Regarding the visuals, this track is well-lit and the road boundaries are clear. One of the difficulites may be that there are no lane lines.
- the **second** track is far more challanging, with sharp turns, steep hills, shadowy/dark areas. On of its advantages though is that there is a central line which divides the road. This might help drive the car autonomously as it can provide a high-level feature.

Since I had trouble even driving the second track manually, I decided to tackle the first track in the beginning, and if it goes well, train for the second.

### Collecting data

#### Driving styles

Udacity provided a good dataset to start with, so I started off from that - but I also added to it.
For collecting data, first, I drove around the track three times very carefully, trying to drive as smoothly as I could. I recorded this as the basis of the training (`data` folder). Then, as it was suggested in the course, I recorded multiple attempts at '*correction driving*', when I pulled the car back to the center from the edge of the road. Then, as I was experiencing issues in autonomous mode around sandy corners (where there was no obvious curb), I started to collect more data around those areas (`curves _and_sandy_edges` folder) .


#### Filtering and augmenting the dataset

For augmenting the dataset, I flipped each image horizontally, and changed the sign of the corresponding steering angle.
I decided to skip data points where the steering angle is 0 - this turned out to quicken the training.


### Working on the model architecture

Since I was sure that I wanted to crop the images so that they exclude the upper part (hills, trees, sky) and the lower part (hood of the car), I introduced a `2D cropping layer` at the beginning of the network. I then added a `normalizing Lambda layer` to normalize images.

Then, at first, I started experimenting with a very basic architecture: just one `Flatten` and one `Dense` layer, using `mse` of measuring loss and an `Adamoptimizer`.
It turned out to be usable, but not great. On the positive side, this was a very fast method of training, and since I have run out of AWS credits and I performed the training on my laptop, it was a great way of exploration.

Then, I started to add more layers to the network.

### The final architecture

To have a reasonable time/accuracy ration, and since I did not have access to a GPU, I had to keep my model simple. I've opted for the following architecture in the end:

- 2D cropping layer, with input dimension: 160, 320, 3
-

### Caveats of the autonomous driving

For quite a long time I had trouble getting the car to drive autonomously, even though I had trained my model extensively. It had a validation loss of `0.0033`, but the car would just hit the curb and not follow the road. Since the training supposedly went well, and validation loss was very low, I figured it has to be some basic difference between the images I used for training and the images that are used for autonomous driving. After some research and Slack-channel exploration, it became suspicious that the training model and `drive.py` use different color channels: `BGR` and `RGB`, respectively. This would of course confuse the model as it '**sees**' totally different color during autonomous driving than during training. Switching the color channel of training images solved this issue.


Sandy edges, dirt road? 


---- 
----
---


This repository contains starting files for the Behavioral Cloning Project.

In this project, you will use what you've learned about deep neural networks and convolutional neural networks to clone driving behavior. You will train, validate and test a model using Keras. The model will output a steering angle to an autonomous vehicle.

We have provided a simulator where you can steer a car around a track for data collection. You'll use image data and steering angles to train a neural network and then use this model to drive the car autonomously around the track.

We also want you to create a detailed writeup of the project. Check out the [writeup template](https://github.com/udacity/CarND-Behavioral-Cloning-P3/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup. The writeup can be either a markdown file or a pdf document.

To meet specifications, the project will require submitting five files:
* model.py (script used to create and train the model)
* drive.py (script to drive the car - feel free to modify this file)
* model.h5 (a trained Keras model)
* a report writeup file (either markdown or pdf)
* video.mp4 (a video recording of your vehicle driving autonomously around the track for at least one full lap)

This README file describes how to output the video in the "Details About Files In This Directory" section.

Creating a Great Writeup
---
A great writeup should include the [rubric points](https://review.udacity.com/#!/rubrics/432/view) as well as your description of how you addressed each point.  You should include a detailed description of the code used (with line-number references and code snippets where necessary), and links to other supporting documents or external references.  You should include images in your writeup to demonstrate how your code works with examples.  

All that said, please be concise!  We're not looking for you to write a book here, just a brief description of how you passed each rubric point, and references to the relevant code :).

You're not required to use markdown for your writeup.  If you use another method please just submit a pdf of your writeup.

The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

The following resources can be found in this github repository:
* drive.py
* video.py
* writeup_template.md

The simulator can be downloaded from the classroom. In the classroom, we have also provided sample data that you can optionally use to help train your model.

## Details About Files In This Directory

### `drive.py`

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```sh
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument, `run1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

```sh
ls run1

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
...
```

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run1
```

Creates a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.

#### Why create a video

1. It's been noted the simulator might perform differently based on the hardware. So if your model drives succesfully on your machine it might not on another machine (your reviewer). Saving a video is a solid backup in case this happens.
2. You could slightly alter the code in `drive.py` and/or `video.py` to create a video of what your model sees after the image is processed (may be helpful for debugging).

### Tips
- Please keep in mind that training images are loaded in BGR colorspace using cv2 while drive.py load images in RGB to predict the steering angles.

## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).
