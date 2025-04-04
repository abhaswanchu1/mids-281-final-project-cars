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
print("Packages Loaded")

##############################################################################################
#######################  Load Images  ########################################################
##############################################################################################
tic = time()

# Function to gather images from both directories
dir_1 = r'C:\Users\mhurth\REPO\MIDS281\train'
dir_2 = r'C:\Users\mhurth\REPO\MIDS281\test'

# For each directory, look at subdirectories and gather image names if the subdirectory has "Chevrolet",
# "Dodge", "Audi", or "BMW" in the directory Name
def gather_images(dir_1, dir_2):
    images = []
    for dir_path in [dir_1, dir_2]:
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file.endswith('.jpg'):
                    if any(x in root for x in ['Chevrolet', 'Dodge', 'Audi', 'BMW']):
                        image = {}
                        image["path"] = os.path.join(root, file) # store path
                        image["class"] = os.path.basename(root)
                        image["make"] = image["class"].split()[0] # store make
                        images.append(image)
    image_df = pd.DataFrame(images)
    return image_df

image_paths = gather_images(dir_1, dir_2)

# Assign each row to train/test 70/30 split but stratify by make
train_df, test_df = train_test_split(image_paths, test_size=0.3, stratify=image_paths["make"], random_state=42)
train_df.to_csv("train_images_split.csv", index=False)
test_df.to_csv("test_images_split.csv", index=False)

# For each image in train_df load in the image and resize the image to 224x224
# Convert the image to a numpy array in RGP format
def load_jpg(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Crop image so that it is square (DONT CHANGE ASPECT RATIO)
    h, w, _ = img.shape
    if h > w:
        img = img[(h-w)//2:(h+w)//2, :]
    else:
        img = img[:, (w-h)//2:(w+h)//2]
    img = cv2.resize(img, (224, 224))
    return img

def load_images(df):
    images = []
    Y = []
    for index, row in df.iterrows():
        img = load_jpg(row["path"])
        images.append(img)
        Y.append(row["make"])
    return images, Y

train_images, targets = load_images(train_df[0:10])

# Plot the Images and their labels
plt.figure(figsize=(20,10))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(train_images[i])
    plt.title(targets[i])
    plt.axis('off')
plt.show()

print("Images Loaded in: ", time()-tic)
##############################################################################################
#######################  Augment Images  #####################################################
##############################################################################################
tic = time()

def augment_images(images, targets):
    # OKAY TO FLIP IMAGES, but rotations and shearing will cause misalignment and probably not help blind
    # Missaligned edges (black space) a problem downstream? YES
    augmented_images = []
    augmented_target = []
    for i in range(len(images)):
        img = images[i]
        img = cv2.flip(img, 1)
        ### Consider Adding Contrast Normalization if the training data isn't large enough
        augmented_images.append(img)
        augmented_target.append(targets[i])
    return augmented_images, augmented_target

# # Plot the Augmented Images and their labels
augmented_images, augmented_targets = augment_images(train_images, targets)

plt.figure(figsize=(20,10))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(augmented_images[i])
    plt.title(augmented_targets[i])
    plt.axis('off')
plt.show()

train_images.extend(augmented_images)
targets.extend(augmented_targets)

print("Images Augmented in: ", time()-tic)
##############################################################################################
#######################  Feature Building ####################################################
##############################################################################################
'''
HOG transform, Forrier Transform, and Canny Edge Detect on images, save features as seperate Arrays
'''
tic = time()

def hog_features(images):
    hog_features = []
    for img in images:
        fd, hog_feature = hog(img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1),
                          visualize=True, channel_axis=-1)
        hog_image_rescaled = exposure.rescale_intensity(hog_feature, in_range=(0, 10))
        hog_features.append(hog_image_rescaled)
    return hog_features

def fourier_features(images):
    fourier_features = []
    [ydim, xdim, zdim] = images[0].shape
    win = np.outer(np.hanning(ydim), np.hanning(xdim))
    win = win / np.mean(win)
    for img in images:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        F = np.fft.fftshift(np.fft.fft2(gray_img*win))
        Fmag = np.abs(F)
        fourier_features.append(Fmag)
    return fourier_features

def canny_features(images):
    canny_features = []
    for img in images:
        canny = cv2.Canny(img, 100, 200)
        canny_features.append(canny)
    return canny_features

hog = hog_features(train_images)
fourier = fourier_features(train_images)
canny = canny_features(train_images)

for i in range(20):
    # plot the image, hog, fourier, and canny
    plt.figure(figsize=(20,20))
    plt.subplot(2, 2, 1)
    plt.imshow(train_images[i])
    plt.title("Image")
    plt.axis('off')
    plt.subplot(2, 2, 2)
    plt.imshow(hog[i], cmap='gray')
    plt.title("HOG")
    plt.axis('off')
    plt.subplot(2, 2, 3)
    plt.imshow(np.log(fourier[i]), cmap='gray')
    plt.title("Fourier")
    plt.axis('off')
    plt.subplot(2, 2, 4)
    plt.imshow(canny[i], cmap='gray')
    plt.title("Canny")
    plt.axis('off')
    plt.show()

print("Features Built in: ", time()-tic)
##############################################################################################
############################ Flatten and Stack Features Images ###############################
##############################################################################################
# Flatten all features into a single row
tic = time()
def stack_rows(features):
    '''
    Efficiently stacks multiple lists of image features row-wise.
    For each image i, flattens all features and concatenates them into a single row.
    Returns a 2D numpy array.
    '''
    stacked = np.array([
        np.concatenate([feature[i].flatten() for feature in features])
        for i in range(len(features[0]))
    ])
    return stacked


stacked_features = stack_rows([hog, fourier, canny])

print("Features Stacked in: ", time()-tic)
##############################################################################################
################################ Principal Component Analysis ################################
##############################################################################################
tic = time()

# Normalize the stacked features
from sklearn.preprocessing import StandardScaler
stacked_features = StandardScaler().fit_transform(stacked_features)

# Run PCA on the stacked features
pca = PCA(n_components=0.95)
pca.fit(stacked_features)
X = pca.transform(stacked_features)

# Plot the explained variance
plt.figure(figsize=(20,10))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance')
plt.title('Explained Variance vs. Number of Components')
plt.grid()
plt.show()


print("PCA Completed in: ", time()-tic)





