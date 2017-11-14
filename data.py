# Import packages
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

# define path
data_path = '../Simulated Data Old/'
data_augmentation = True
use_multiple_cameras = True
correction_factor = 0.2

def load_data():
    images,measurements = collect_data()

    # Define training data
    X_train = np.array(images)
    y_train = np.array(measurements)

    print('Shape image data: ', X_train.shape)
    print('Shape measurement data: ', y_train.shape)

    return X_train,y_train

def collect_data():
    # init csv file buffer
    lines = []

    # open csv file from simulation data
    with open(data_path + 'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
            
    # init image and measurement buffer
    images = []
    measurements = []

    # loop through frames and store images and measurements
    for line in lines:
        source_path = line[0]
        file_name = source_path.split('/')[-1]
        current_path = data_path + 'IMG/' + file_name
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)
        if use_multiple_cameras:
            if not line[1] == "":
                source_path = line[1]
                file_name = source_path.split('/')[-1]
                current_path = data_path + 'IMG/' + file_name
                image = cv2.imread(current_path)
                images.append(image)
                measurement = float(line[3]) + correction_factor
                measurements.append(measurement)
            if not line[2] == "":
                source_path = line[2]
                file_name = source_path.split('/')[-1]
                current_path = data_path + 'IMG/' + file_name
                image = cv2.imread(current_path)
                images.append(image)
                measurement = float(line[3]) - correction_factor
                measurements.append(measurement)

                

    # data augmentation
    if data_augmentation:
        extend_with_flipped_images(images,measurements)
        
    return images,measurements


def extend_with_flipped_images(images,measurements):

    print("Before data augmentation = " + str(len(images)))
    # determine
    number_of_images = len(images)
    for i in range(number_of_images):
        # flipped data
        images.append(cv2.flip(images[i],1))
        measurements.append(measurements[i]*-1.0)
        
    print("After data augmentation = " + str(len(images)))

def draw_data(X,y):
    n, bins, patches = plt.hist(y, 29, facecolor='g', alpha=0.75)
    plt.title('Histogram of steering angles')
    plt.grid(True)
    plt.show()


