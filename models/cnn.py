import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../scripts/data'))
from preprocessing import augment_images_in_folder
from preprocessing import prepareimages
from preprocessing import shuffledata

folder_benign_train_path= 'images/data/train/benign'
folder_malignant_train_path = 'images/data/train/malignant'

folder_benign_test_path= 'images/data/test/benign'
folder_malignant_test_path= 'images/data/test/malignant'

output_folder = folder_benign_train_path+"_augmented"  # Folder to save augmented images
augment_images_in_folder(folder_benign_train_path, output_folder, num_augmentations=1)

output_folder = folder_malignant_train_path+"_augmented"  # Folder to save augmented images
augment_images_in_folder(folder_malignant_train_path, output_folder, num_augmentations=1)

output_folder = folder_benign_test_path+"_augmented"  # Folder to save augmented images
augment_images_in_folder(folder_benign_test_path, output_folder, num_augmentations=1)

output_folder = folder_malignant_test_path+"_augmented"  # Folder to save augmented images
augment_images_in_folder(folder_benign_test_path, output_folder, num_augmentations=1)

X_benign,X_malignant,y_benign,y_malignant=prepareimages(folder_benign_train_path+"_augmented",folder_malignant_train_path+"_augmented")
X_benign_test,X_malignant_test,y_benign_test,y_malignant_test=prepareimages(folder_benign_test_path+"_augmented",folder_malignant_test_path+"_augmented")

X_benign_aug,X_malignant_aug,y_benign_aug,y_malignant_aug=prepareimages(folder_benign_train_path,folder_malignant_train_path)
X_benign_test_aug,X_malignant_test_aug,y_benign_test_aug,y_malignant_test_aug=prepareimages(folder_benign_test_path,folder_malignant_test_path)

X_train,y_train=shuffledata(X_benign,X_benign_aug,X_malignant,X_malignant_aug,y_benign,y_benign_aug,y_malignant,y_malignant_aug)
X_test,y_test=shuffledata(X_benign_test,X_benign_test_aug,X_malignant_test,X_malignant_test_aug,y_benign_test,y_benign_test_aug,y_malignant_test,y_malignant_test_aug)

X_train=X_train/255.0
X_test=X_test/255.0

