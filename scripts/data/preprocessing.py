from sklearn.preprocessing import StandardScaler
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
import seaborn as sns
from PIL import Image

def prepareimages(folder_benign_path,folder_malignant_path):
    """ 
    This function reads images from the specified folders
    Input: Two strings, each is a folder path to benign and malignant: train and test folders
    Output: Four Numpy arrays split into X and y for benign and malignant images each
    """

    read = lambda imgname: np.asarray(Image.open(imgname).convert("RGB"))     #lambda function to convert an image to RGB(if necessary) and returns a numpy array of the same
    # Load in pictures 
    imgs_benign = [read(os.path.join(folder_benign_path, filename)) for filename in os.listdir(folder_benign_path)]
    X_benign = np.array(imgs_benign, dtype='uint8')
    imgs_malignant = [read(os.path.join(folder_malignant_path, filename)) for filename in os.listdir(folder_malignant_path)]
    X_malignant = np.array(imgs_malignant, dtype='uint8')

    # Create labels: benign are labelled as 0 and malignant are labelled  as 1
    y_benign = np.zeros(X_benign.shape[0])
    y_malignant = np.ones(X_malignant.shape[0])

    return X_benign,X_malignant,y_benign,y_malignant

def shuffledata(X_benign,X_malignant, y_benign, y_malignant):
    """
    This function shuffles the numpy array to avoid the ordering of images after concatenation
    Input: 4 Numpy arrays, each for benign and malignant for X and y
    Output: 2 Numpy arrays, after shuffling the data
    """
    
    X = np.concatenate((X_benign, X_malignant), axis = 0)
    y = np.concatenate((y_benign, y_malignant), axis = 0)

    s = np.arange(X.shape[0])     #returns an array of numbers from 0 to X.shape[0]
    np.random.shuffle(s)                #shuffles the array in place
    X = X[s]
    y = y[s]
    return X,y
