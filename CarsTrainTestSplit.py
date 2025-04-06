import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from skimage.feature import hog, match_template
from skimage import data, exposure
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from time import time
from shutil import copyfile
print("Packages Loaded")

##############################################################################################
#######################  Load Images  ########################################################
##############################################################################################
tic = time()

# File locations
excel_file = r"C:\Users\mhurth\REPO\MIDS281\mids-281-final-project-cars\cars_classes.xlsx"
train_folder = r"C:\Users\mhurth\REPO\MIDS281\cars_train\cars_train"
test_folder = r"C:\Users\mhurth\REPO\MIDS281\cars_test\cars_test"
output_folder = r"C:\Users\mhurth\REPO\MIDS281\cars_dataset"  # Folder for the combined dataset
excel_out = r"C:\Users\mhurth\REPO\MIDS281\mids-281-final-project-cars\cars_classes_split.xlsx"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Load training data
train_df = pd.read_excel(excel_file, sheet_name='train')

# Load testing data
test_df = pd.read_excel(excel_file, sheet_name='test')

# Function to label and copy images
def label_and_copy(df, image_folder, output_folder):
    for index, row in df.iterrows():
        image_name = row['image']  # Assuming 'image_name' column exists
        true_class_name = row['true_class_name']
        class_name = true_class_name.split()[0]
        source_path = os.path.join(image_folder, image_name)
        destination_path = os.path.join(output_folder, class_name, image_name)
        os.makedirs(os.path.join(output_folder, class_name), exist_ok=True)  # Create class folders
        copyfile(source_path, destination_path)

label_and_copy(train_df, train_folder, output_folder)# Label and copy training images
label_and_copy(test_df, test_folder, output_folder)# Label and copy testing images

def create_filtered_dataframe(dataset_path, brands):
  """
  Creates a Pandas DataFrame from a dataset folder,
  filtering for specific brands.

  Args:
    dataset_path: Path to the dataset folder.
    brands: List of brands to include.

  Returns:
    A Pandas DataFrame with image paths and labels.
  """

  data = []
  for brand in brands:
    brand_folder = os.path.join(dataset_path, brand)
    if os.path.isdir(brand_folder):
      for image_name in os.listdir(brand_folder):
        image_path = os.path.join(brand_folder, image_name)
        # image = cv2.imread(image_path)
        data.append({'image_path': image_path, 'brand': brand}) # 'image':image
  return pd.DataFrame(data)

# Specify the dataset path and desired brands
selected_brands = ['Chevrolet', 'BMW', 'Dodge', 'Audi']

# Create the filtered DataFrame
filtered_df = create_filtered_dataframe(output_folder, selected_brands)

# Display the DataFrame (optional)
filtered_df.head()

# Split the DataFrame into training and testing sets
train_df, test_df = train_test_split(filtered_df, test_size=0.3, random_state=42,stratify=filtered_df['brand'])

# Save the New DataFrames to the same xlsx files with two sheets
with pd.ExcelWriter(excel_out) as writer:
    train_df.to_excel(writer, sheet_name='train', index=False)
    test_df.to_excel(writer, sheet_name='test', index=False)

print("Images Copied and Split in: ", time()-tic)