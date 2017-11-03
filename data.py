# Import packages
import csv
import cv2
import numpy as np

# define path
data_path = '../Simulated Data/'


def load_data():
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

    # Define training data
    X_train = np.array(images)
    y_train = np.array(measurements)

    print('Shape image data: ', X_train.shape)
    print('Shape measurement data: ', y_train.shape)

    return X_train,y_train
