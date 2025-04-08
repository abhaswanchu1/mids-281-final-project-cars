# Import Packages
import cv2
import numpy as np
import pandas as pd
from skimage.feature import hog, match_template
from skimage import data, exposure
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from time import time

from sklearn.preprocessing import StandardScaler, LabelBinarizer
import pickle

print("Packages Loaded")

##############################################################################################
#######################  Load Images  ########################################################
##############################################################################################
tic = time()

train_df = pd.read_excel(r"C:\Users\mhurth\REPO\MIDS281\mids-281-final-project-cars\cars_classes_split.xlsx",
                         sheet_name='train')
test_df = pd.read_excel(r"C:\Users\mhurth\REPO\MIDS281\mids-281-final-project-cars\cars_classes_split.xlsx",
                        sheet_name='test')


# For each image in train_df load in the image and resize the image to 224x224
# Convert the image to a numpy array in RGP format
def load_jpg(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Crop image so that it is square (DONT CHANGE ASPECT RATIO)
    h, w, _ = img.shape
    if h > w:
        img = img[(h - w) // 2:(h + w) // 2, :]
    else:
        img = img[:, (w - h) // 2:(w + h) // 2]
    img = cv2.resize(img, (224, 224))
    return img


def load_images(df):
    images = []
    Y = []
    for index, row in df.iterrows():
        img = load_jpg(row["image_path"])
        images.append(img)
        Y.append(row["brand"])
    return images, Y


train_images, train_targets = load_images(train_df)
test_images, test_targets = load_images(test_df)

# Plot the Images and their labels
plt.figure(figsize=(20, 10))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(train_images[i])
    plt.title(train_targets[i])
    plt.axis('off')
plt.show()

print("Images Loaded in: ", time() - tic)

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
augmented_images, augmented_targets = augment_images(train_images, train_targets)
augmented_images_test, augmented_targets_test = augment_images(test_images, test_targets)

plt.figure(figsize=(20, 10))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(augmented_images[i])
    plt.title(augmented_targets[i])
    plt.axis('off')
plt.show()

train_images.extend(augmented_images)
train_targets.extend(augmented_targets)
test_images.extend(augmented_images_test)
test_targets.extend(augmented_targets_test)

del (augmented_images, augmented_targets, augmented_images_test, augmented_targets_test)

print("Images Augmented in: ", time() - tic)

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
        # fd, hog_feature = hog(img, orientations=8, pixels_per_cell=(16, 16), visualize=True,
        #                   cells_per_block=(1, 1), channel_axis=-1) # Uncomment to Plot HOG feature
        # hog_image_rescaled = exposure.rescale_intensity(hog_feature, in_range=(0, 10))
        # hog_features.append(hog_image_rescaled)
        hog_feature = hog(img, orientations=8, pixels_per_cell=(16, 16),
                           cells_per_block=(1, 1), channel_axis=-1)
        hog_features.append(hog_feature)
    return hog_features


def fourier_features(images):
    fourier_features = []
    [ydim, xdim, zdim] = images[0].shape
    win = np.outer(np.hanning(ydim), np.hanning(xdim))
    win = win / np.mean(win)
    for img in images:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        F = np.fft.fftshift(np.fft.fft2(gray_img * win))
        Fmag = np.abs(F)
        fourier_features.append(Fmag)
    return fourier_features


def canny_features(images):
    canny_features = []
    for img in images:
        canny = cv2.Canny(img, 100, 200)
        canny_features.append(canny)
    return canny_features


# Train Features
hog_train = hog_features(train_images)
fourier_train = fourier_features(train_images)
canny_train = canny_features(train_images)

# Test Features
hog_test = hog_features(test_images)
fourier_test = fourier_features(test_images)
canny_test = canny_features(test_images)

# for i in range(5):
#     # plot the image, hog, fourier, and canny
#     plt.figure(figsize=(20, 20))
#     plt.subplot(2, 2, 1)
#     plt.imshow(train_images[i])
#     plt.title("Image")
#     plt.axis('off')
#     plt.subplot(2, 2, 2)
#     plt.imshow(hog_train[i], cmap='gray')
#     plt.title("HOG")
#     plt.axis('off')
#     plt.subplot(2, 2, 3)
#     plt.imshow(np.log(fourier_train[i]), cmap='gray')
#     plt.title("Fourier")
#     plt.axis('off')
#     plt.subplot(2, 2, 4)
#     plt.imshow(canny_train[i], cmap='gray')
#     plt.title("Canny")
#     plt.axis('off')
#     plt.show()

print("Features Built in: ", time() - tic)

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


stacked_features = stack_rows([hog_train, fourier_train, canny_train])
stacked_features_test = stack_rows([hog_test, fourier_test, canny_test])

del (hog_train, fourier_train, canny_train, hog_test, fourier_test, canny_test)

print("Features Stacked in: ", time() - tic)

##############################################################################################
################################ Principal Component Analysis ################################
##############################################################################################
tic = time()

# Normalize the stacked features
scaler = StandardScaler()
scaler.fit(stacked_features)
stacked_features = scaler.transform(stacked_features)
stacked_features_test = scaler.transform(stacked_features_test)

# # Run PCA on the stacked features
# pca = PCA(n_components=0.95)
# pca.fit(stacked_features)
#
# # Save out the PCA model
# with open(r"C:\Users\mhurth\REPO\MIDS281\mids-281-final-project-cars\PCA.pkl", "wb") as f:
#     pickle.dump(pca, f)

# # Plot the explained variance
# plt.figure(figsize=(20,10))
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel('Number of Components')
# plt.ylabel('Explained Variance')
# plt.title('Explained Variance vs. Number of Components')
# plt.grid()
# plt.show()

# Load the PCA model
with open(r"C:\Users\mhurth\REPO\MIDS281\mids-281-final-project-cars\PCA.pkl", "rb") as f:
    pca = pickle.load(f)

X = pca.transform(stacked_features)
X_test = pca.transform(stacked_features_test)

# Trim X to the number of components that explain 90% of the variance
n_components = 2000
X = X[:, :n_components]
X_test = X_test[:, :n_components]

print("PCA Completed in: ", time() - tic)

# Save out X, X_test, Targets, and Test_Targets as numpy arrays
np.save(r"C:\Users\mhurth\REPO\MIDS281\mids-281-final-project-cars\X.npy", X)
np.save(r"C:\Users\mhurth\REPO\MIDS281\mids-281-final-project-cars\X_test.npy", X_test)
np.save(r"C:\Users\mhurth\REPO\MIDS281\mids-281-final-project-cars\targets.npy", train_targets)
np.save(r"C:\Users\mhurth\REPO\MIDS281\mids-281-final-project-cars\test_targets.npy", test_targets)


