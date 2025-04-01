#import os
#import numpy as np
#import kagglehub
#from sklearn.datasets import load_files

# Download the dataset from Kaggle
#path1 = kagglehub.dataset_download("nizarcharrada/khattarabic")

# Show the path of the downloaded dataset
#print("Dataset downloaded to:", path1)

# Path to the dataset
#dataset_path = os.path.join(path1, "train_deskewed")

# Load data (if it's text-based)
#data = load_files(dataset_path, encoding="utf-8", decode_error="replace")

#x_train = np.array(data.data)[:100]
#y_train = np.array(data.target)[:100].reshape(-1, 1)

#print(x_train.shape)
#print(y_train.shape)
#print(data.target_names)

import os
import numpy as np
import kagglehub
from PIL import Image
from sklearn.model_selection import train_test_split

# Download dataset
path1 = kagglehub.dataset_download("nizarcharrada/khattarabic")
print("Dataset downloaded to:", path1)

# Path to the training data
dataset_path = os.path.join(path1, "train_deskewed")
# Load images and labels
images = []
labels = []

# Loop through all folders (each folder is a different Arabic script style)
for script_style in os.listdir(dataset_path):
    script_path = os.path.join(dataset_path, script_style)
    
    # Loop through all images in each style folder
    for image_file in os.listdir(script_path):
        image_path = os.path.join(script_path, image_file)
        
        # Open image
        image = Image.open(image_path).convert('L')  # Convert to grayscale
        
        # Resize to a fixed size (مثلا 64x64)
        image = image.resize((64, 64))
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Append to the list
        images.append(image_array)
        labels.append(script_style)

images = np.array(images)
labels = np.array(labels)
print(images.shape)  # مثلا (1000, 64, 64)
print(labels.shape)  # (1000,)
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
