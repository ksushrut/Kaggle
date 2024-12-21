from sklearn.preprocessing import StandardScaler
import os
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import pandas as pd
import seaborn as sns
from PIL import Image
from torchvision import transforms

# Function to perform data augmentation with Horizontal Flip
def augment_images_in_folder(input_folder, output_folder, num_augmentations=5, image_size=(224, 224)):
    """
    Augments images in a folder by applying horizontal flip and other transformations.
    
    Parameters:
    - input_folder: Path to the input folder containing images.
    - output_folder: Path to the output folder where augmented images will be saved.
    - num_augmentations: Number of augmented images per input image.
    - image_size: The target size of the images (default is 224x224).
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Define transformations (augmentation techniques)
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=1),  # Apply horizontal flip with probability 1 (always flip)
        transforms.RandomRotation(20),  # Random rotation by 20 degrees
        transforms.RandomResizedCrop(image_size[0]),  # Random resized crop
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Random color jitter
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),  # Random affine transformation
        transforms.ToTensor()  # Convert to tensor
    ])

    # List all image files in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Iterate through the images and apply augmentations
    for img_idx, img_name in enumerate(image_files):
        # Open the image
        img_path = os.path.join(input_folder, img_name)
        img = Image.open(img_path).convert('RGB')  # Ensure it's in RGB mode
        
        # Apply the augmentations multiple times per image
        for aug_idx in range(num_augmentations):
            augmented_img = transform(img)
            augmented_img = augmented_img.permute(1, 2, 0).numpy()  # Convert to [H, W, C] format
            
            # Denormalize the image back to [0, 255] range
            augmented_img = (augmented_img * 255).astype(np.uint8)
            
            # Save the augmented image
            augmented_img_pil = Image.fromarray(augmented_img)  # Convert back to PIL image
            save_path = os.path.join(output_folder, f"{os.path.splitext(img_name)[0]}_aug_{aug_idx+1}.jpg")
            augmented_img_pil.save(save_path)

def prepareimages(folder_benign_path,folder_malignant_path, size = (224,224)):
    """ 
    This function reads images from the specified folders
    Input: 
        - Two strings, each is a folder path to benign and malignant: train and test folders
        - Required image size (default = 224,224 across all images in the dataset)
    Output: Four Numpy arrays split into X and y for benign and malignant images each
    """

    def read_and_resize(imgname):
        """
            Helper Function to resize and normalize images
        """
        img = np.asarray(Image.open(imgname).convert("RGB").resize(size), dtype = 'float32')
        return img
    
    # Load in pictures 
    imgs_benign = [read_and_resize(os.path.join(folder_benign_path, filename)) for filename in os.listdir(folder_benign_path)]
    X_benign = np.array(imgs_benign, dtype='uint8')
    imgs_malignant = [read_and_resize(os.path.join(folder_malignant_path, filename)) for filename in os.listdir(folder_malignant_path)]
    X_malignant = np.array(imgs_malignant, dtype='uint8')

    # Create labels: benign are labelled as 0 and malignant are labelled  as 1
    y_benign = np.zeros(X_benign.shape[0])
    y_malignant = np.ones(X_malignant.shape[0])

    return X_benign,X_malignant,y_benign,y_malignant

def shuffledata(X_benign,X_benign_aug,X_malignant,X_malignant_aug, y_benign,y_benign_aug, y_malignant,y_malignant_aug):

    """
    This function shuffles the numpy array to avoid the ordering of images after concatenation
    Input: 4 Numpy arrays, each for benign and malignant for X and y
    Output: 2 Numpy arrays, after shuffling the data
    """
    
    X = np.concatenate((X_benign,X_benign_aug, X_malignant,X_malignant_aug), axis = 0)
    y = np.concatenate((y_benign,y_benign_aug, y_malignant,y_malignant_aug), axis = 0)

    s = np.arange(X.shape[0])     #returns an array of numbers from 0 to X.shape[0]
    np.random.shuffle(s)                #shuffles the array in place
    X = X[s]
    y = y[s]
    return X,y

# Data Augmentation Steps

