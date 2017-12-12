import os
import csv
import cv2
import numpy as np
import sklearn

from sklearn.model_selection import train_test_split

from keras.models import Sequential, Model
from keras.layers import Lambda, Dense, Flatten, Cropping2D, MaxPooling2D, Convolution2D

MIN_ANGLE = 0.022

def get_samples(driving_style, min_angle=0):
    '''Helper generator to yield a line from the dataset.
       Optionally filter lines based on steering angle.'''
    with open('../{}/driving_log.csv'.format(driving_style), 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for line in reader:
            if abs(float(line[3].strip())) < min_angle:
                continue
            yield line


def adjust_angle(camera_position, angle):
    '''Adjusts angle according to camera position'''
    if camera_position == 'left':
        new_angle = angle + 0.2
    elif camera_position == 'right':
        new_angle = angle - 0.2
    else:
        new_angle = angle
    return new_angle


def get_filename(sample_line, camera_position='center'):
    '''Returns filename relative to the path of the current file'''
    index = {'center': 0, 'left': 1, 'right': 2}[camera_position]
    relative_path = sample_line[index].split('/')[-3:]
    filename = os.path.join('..', *relative_path)
    if not os.path.exists(filename):
        filename = os.path.join('../udacity_data', *
                                [part.strip() for part in relative_path])
    return filename


def crop_image(image, from_top=50, from_bottom=20):
    '''Crops image: removes a bar from the top and from the bottom.
       Helper function for visualization purposes, not used in the model or for image processing.'''
    image[:from_top] = 0
    image[len(image)-from_bottom:] = 0
    return image


def flatten(nested_list):
    '''Helper function to flatten a nested iterable'''
    return [item for sublist in nested_list for item in sublist]


def add_image_and_angle_to_dataset(image, angle, images, angles):
    '''Adds the image and angle to the corresponding datasets,
       as well as the mirrored image and with the sign of the angle changed.'''
    images.append(image)
    angles.append(angle)

    image_flipped = np.fliplr(image)
    angle_flipped = -angle

    images.append(image_flipped)
    angles.append(angle_flipped)

def generate_images(samples, batch_size=32):
    '''Generator that yields bathches of training data and their corresponding labels'''
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images, angles = [], []

            for batch_sample in batch_samples:
                for index, camera_position in enumerate(['center', 'left', 'right']): #
                    angle = float(batch_sample[3].strip())

                    image = cv2.imread(get_filename(batch_sample, camera_position))
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # converting BGR color channel to RGB

                    angle = adjust_angle(camera_position, angle)
                    add_image_and_angle_to_dataset(image, angle, images, angles)

            X_train, y_train = np.array(images), np.array(angles)

            yield X_train, y_train

# 'udacity_data' is the dataset provided by udacity
# 'data' folder contains 'basic' driving data from a smooth driving session
# 'correction_driving' contains samples from a corrective driving session
#       (driving the card back to the center of the road from the edges of the road )
# 'curves_and_sandy_edges' represents difficult areas of the track (sandy parts, curves)
driving_styles = ['udacity_data'] + 2*['correction_driving'] + 3*['data'] + 4*['curves_and_sandy_edges']

samples = flatten([get_samples(driving_style, min_angle=MIN_ANGLE)
                   for driving_style in driving_styles])[:10000]

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

train_generator = generate_images(train_samples, batch_size=32)
validation_generator = generate_images(validation_samples, batch_size=32)


def get_model():
    '''Defines the CNN architecture: layers, dimensions, loss function and optimizer'''
    ch, row, col = 3, 90, 320  # Trimmed image format

    model = Sequential()
    # Crop top and bottom of image
    model.add(Cropping2D(cropping=((50, 40), (0, 0)), input_shape=(160, 320, 3)))

    # Preprocess incoming data, centered around zero with small standard deviation
    model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=(90, 320, 3)))
    model.add(Convolution2D(6, (5, 5), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten(input_shape=(ch, row, col)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')

    return model

def train_model(model, train_generator, train_samples, validation_generator, validation_samples):
    '''Trains the model'''
    history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples),
                                         validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=4)


    model.save('model.h5_exclude_angles_below_{}_samplesize_{}'.format(MIN_ANGLE, len(train_samples)))
    print('number of images used for the training: {}'.format(len(train_samples)))


model = get_model()

train_model(model, train_generator, train_samples, validation_generator, validation_samples)
