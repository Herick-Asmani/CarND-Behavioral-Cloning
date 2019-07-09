import zipfile
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sklearn
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.models import model_from_json
from keras.optimizers import Adam
from keras.layers.advanced_activations import ELU
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers import Convolution2D, Cropping2D, Conv2D, BatchNormalization, Activation
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

# zip_ref = zipfile.ZipFile('./MyData.zip', 'r')
# zip_ref.extractall('/home/workspace/CarND-Behavioral-Cloning-P3')
# zip_ref.close()

# Obtain Udacity's Data
lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
# Obtain My Data generated using Udacity's simulator
with open('./Data/driving_log.csv') as csvfile2:
    reader2 = csv.reader(csvfile2)
    for line in reader2:
        lines.append(line)
print("Total number of data (including only single camera angle): ",len(lines))


print(lines[0])
print(lines[1])
print(lines[-1])


lines = lines[1:]
image_paths = []
steering_angles = []
# Gather data - image paths and angles for center, left, right cameras in each row
for line in lines:
    # skip it if ~0 speed - not representative of driving behavior
    if float(line[6]) < 0.1 :
        continue
    # get center image path and angle
    image_paths.append(line[0])
    steering_angles.append(float(line[3]))
    # Create adjusted steering measurements for the side camera images
    correction = 0.25
    # get left image path and angle
    image_paths.append(line[1])
    steering_angles.append(float(line[3]) + correction)
    # get right image path and angle
    image_paths.append(line[2])
    steering_angles.append(float(line[3]) - correction)

image_paths = np.array(image_paths)
steering_angles = np.array(steering_angles)
print("Total number of data (including all three camera angles): ", len(image_paths))
print("Total number of steering angles (including only single camera angle): ", len(steering_angles))


train_samples_path, validation_samples_path,train_steering_angles, validation_steering_angles = train_test_split(image_paths, steering_angles, test_size=0.2)

print("Total number of train samples path: ", len(train_samples_path))
print("Total number of train steering angles: ", len(train_steering_angles))
print("Total number of validation samples path: ", len(validation_samples_path))
print("Total number of validation steering angles: ",len(validation_steering_angles))


# Define Generator to obtain data in batches
def generator(image_samples_path, steering_angles, batch_size = 64):   
    num_samples = len(image_samples_path)
    
    while True:
        image_samples_path, steering_angles = sklearn.utils.shuffle(image_samples_path, steering_angles)
    
        for offset in range(0, num_samples, batch_size):
            batch_samples = image_samples_path[offset:offset+batch_size]
            batch_steer_angles = steering_angles[offset:offset+batch_size]
            images = []
            measurements = []

            for batch_sample, batch_steer_angle in zip(batch_samples, batch_steer_angles):
            # Get images from all camera angles
                if batch_sample.split('/')[1] != 'root':    # To obtain Udacity's Data
                    filename = batch_sample.split('/')[-1]
                    current_path = './data/IMG/' + filename
                    image = mpimg.imread(current_path)
                    images.append(image)
                    measurements.append(batch_steer_angle)
                else:                                      # To obtain My Data
                    filename = batch_sample.split('/')[-1]
                    current_path = './Data/IMG/' + filename
                    image = mpimg.imread(current_path)
                    images.append(image)
                    measurements.append(batch_steer_angle)
        
            X = np.array(images)
            Y = np.array(measurements)
            yield sklearn.utils.shuffle(X, Y)

            
# Set our batch size
batch_size = 64

# Obtain train and validation data using the generator function
train_generator = generator(train_samples_path, train_steering_angles, batch_size=batch_size)
validation_generator = generator(validation_samples_path, validation_steering_angles, batch_size=batch_size)


# Define Model Architecture (NVIDIA Architecture for End to End Self Driving)
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))   # Normalization Layer
model.add(Cropping2D(cropping=((70,25),(0,0))))      # Crop unnecessary regions of interest

model.add(Convolution2D(24, (5, 5), strides=(2, 2), padding="valid", activation='relu'))
# model.add(BatchNormalization())
model.add(Convolution2D(36, (5, 5), strides=(2, 2), padding="valid", activation='relu'))
# model.add(BatchNormalization())
model.add(Convolution2D(48, (5, 5), strides=(2, 2), padding="valid", activation='relu'))
#model.add(BatchNormalization())
model.add(Convolution2D(64, (3, 3), padding="valid", activation='relu'))
#model.add(BatchNormalization())
model.add(Convolution2D(64, (3, 3), padding="valid", activation='relu'))
#model.add(BatchNormalization())
        
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

print('Model Done')
model.summary()


# Compile and Train the model
model.compile(loss='mse',optimizer='adam')

checkpointer = ModelCheckpoint(filepath = './model.h5', monitor = 'val_loss', verbose = 1, save_best_only = True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0)

hist = model.fit_generator(train_generator, steps_per_epoch=np.ceil(len(train_samples_path)/batch_size), validation_data=validation_generator,                          validation_steps=np.ceil(len(validation_samples_path)/batch_size), epochs=25, verbose=1, callbacks = [checkpointer, reduce_lr])