import os
import numpy as np
import cv2

DIR = os.getcwd() + "/stop/validation_dataset_sigma_crop/"

def crop(img, fromTopPx, filename):
    cv2.imwrite(filename, img[fromTopPx:np.shape(img)[0], :, :])

fs = os.scandir(DIR)
for file in fs:
    if ".jpg" in file.name:
        crop(cv2.imread(DIR + file.name), 40, DIR + file.name)