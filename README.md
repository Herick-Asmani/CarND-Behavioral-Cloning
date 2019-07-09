# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/architecture.png "Model Visualization"
[image2]: ./examples/center.jpg "Center image"
[image3]: ./examples/left.jpg "Recovery Image 1"
[image4]: ./examples/right.jpg "Recovery Image 2"
[image5]: ./examples/center_track2.jpg "Track 2 Image"
[image6]: ./examples/center_forward.jpg "Normal driving Image"
[image7]: ./examples/center_backward.jpg "Backward driving Image"

---
### Files Submitted & Code Quality

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to reduce loss on validation set and prevent overfitting such that the car stays on track for autonomous mode.

My first step was to use a Convolutional neural network model similar to the traffic sign classifier project i.e. modified LeNet-5 architecture. I thought this model might be appropriate because it included dropout layers which helps the model to prevent overfitting.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set (80:20). I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. Also, I used this model and ran the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track

To combat this problem and overfitting, I then used Nvidia Architecture of Convolutional Neural Network for Autonomous Driving and added dropout layer to help the model generalize.

Then I used Udacity's simulator to create my own data to improve the driving behavior in cases where car went off-track and combined it with the Udacity's provided data. 

And finally I used this model and ran the simulator to see how well the car was driving around track one. The vehicle was able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (NVIDIA Architecture) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture

![Model Visualization][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded three laps on track one using center lane driving. Here is an example image of center lane driving:

![Center image][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to get back to the center These images show what a recovery looks like:

![Recovery Image 1][image3]
![Recovery Image 2][image4]

Then I repeated this process on track two in order to get more data points.

![Track 2 Image][image5]

To augment the data set, instead of flipping images and angles, I drived the car in the backwards direction thinking that this would help model generalize better at turns and different environment. For example, here is a normal image and backward driven image:

![Normal driving Image][image6]
![Backward driving Image][image7]

After the collection process, I had 75,171 number of data points. I then preprocessed this data by removing those data points whose speed was Zero. So, After removal of those data points, I had 75,036 data points.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 25 as evidenced by the code. I used an adam optimizer so that manually training the learning rate wasn't necessary.
