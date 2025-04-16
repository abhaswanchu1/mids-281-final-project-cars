# Import Packages
import cv2
import numpy as np
import pandas as pd
from skimage.feature import hog, match_template
from skimage import data, exposure
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from time import time
import torchvision.models as models
from torchvision import transforms
from torch import nn
from PIL import Image
import torch
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
    img = cv2.imread(image_path) # Load image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB
    # Crop image so that it is square (DONT CHANGE ASPECT RATIO)
    h, w, _ = img.shape
    # Crop the image to a square
    if h > w:
        img = img[(h - w) // 2:(h + w) // 2, :]
    else:
        img = img[:, (w - h) // 2:(w + h) // 2]
    img = cv2.resize(img, (224, 224)) # Resize the image to 224x224
    return img


def load_images(df):
    images = []
    Y = []
    for index, row in df.iterrows():
        # Load the image and convert to numpy array
        img = load_jpg(row["image_path"])
        images.append(img)
        Y.append(row["brand"])
    return images, Y


train_images, train_targets = load_images(train_df)
test_images, test_targets = load_images(test_df[:(len(test_df) // 2)])
val_images, val_targets = load_images(test_df[(len(test_df) // 2):])

# Plot Some Example Images and their labels
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
    # Consider Adding Contrast Normalization if the training data isn't large enough
    augmented_images_inner = []
    augmented_target = []
    for i in range(len(images)):
        img = images[i]
        img = cv2.flip(img, 1) # Flip the image
        augmented_images_inner.append(img)
        augmented_target.append(targets[i])
    return augmented_images_inner, augmented_target


# # Plot the Augmented Images and their labels
augmented_images, augmented_targets = augment_images(train_images, train_targets)

# Plot Some Example Augmented Images and their labels
plt.figure(figsize=(20, 10))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(augmented_images[i])
    plt.title(augmented_targets[i])
    plt.axis('off')
plt.show()

train_images.extend(augmented_images)
train_targets.extend(augmented_targets)

del (augmented_images, augmented_targets)

print("Images Augmented in: ", time() - tic)

##############################################################################################
#######################  Feature Building ####################################################
##############################################################################################
'''
HOG transform, Forrier Transform, and Canny Edge Detect on images, save features as seperate Arrays
'''
tic = time()


def hog_images(images):
    """
    Compute HOG features for a list of images.
    Use This Function if you want to actually visualize the HOG Image Outputs
    """
    hog_features_inner = []
    for img in images:
        fd, hog_feature = hog(img, orientations=8, pixels_per_cell=(16, 16), visualize=True,
                              cells_per_block=(1, 1), channel_axis=-1)  # Uncomment to Plot HOG feature
        hog_image_rescaled = exposure.rescale_intensity(hog_feature, in_range=(0, 10))
        hog_features_inner.append(hog_image_rescaled)
    return hog_features_inner


def hog_features(images):
    """
    Compute HOG features for a list of images.
    """
    hog_features_inner = []
    for img in images:
        hog_feature = hog(img, orientations=8, pixels_per_cell=(16, 16),
                          cells_per_block=(1, 1), channel_axis=-1)
        hog_features_inner.append(hog_feature)
    return hog_features_inner


def fourier_features(images):
    """
    Compute Fourier features for a list of images.
    """
    fourier_features = []
    [ydim, xdim, _] = images[0].shape
    win = np.outer(np.hanning(ydim), np.hanning(xdim))
    win = win / np.mean(win)
    for img in images:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Convert to grayscale
        f = np.fft.fftshift(np.fft.fft2(gray_img * win))
        fmag = np.abs(f)
        fourier_features.append(fmag)
    return fourier_features


def canny_features(images):
    """
    Compute Canny edges for a list of images.
    """
    canny_features_inner = []
    for img in images:
        canny = cv2.Canny(img, 100, 200)
        canny_features_inner.append(canny)
    return canny_features_inner


# Train Features
hog_train = hog_features(train_images)
fourier_train = fourier_features(train_images)
canny_train = canny_features(train_images)

# Export Train Features for Seperate PCA Analysis
np.save(r"C:\Users\mhurth\REPO\MIDS281\mids-281-final-project-cars\FeatureVectors\hog_train.npy", hog_train)
np.save(r"C:\Users\mhurth\REPO\MIDS281\mids-281-final-project-cars\FeatureVectors\fourier_train.npy", fourier_train)
np.save(r"C:\Users\mhurth\REPO\MIDS281\mids-281-final-project-cars\FeatureVectors\canny_train.npy", canny_train)

# Validation Features
hog_val = hog_features(val_images)
fourier_val = fourier_features(val_images)
canny_val = canny_features(val_images)

# Test Features
hog_test = hog_features(test_images)
fourier_test = fourier_features(test_images)
canny_test = canny_features(test_images)

# Plot Some Example Features for the First 5 Images
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
############################ Pretrained NN Feature Generation  ###############################
##############################################################################################
tic = time()

# Preprocess the images for ResNet Models
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Slice the Model to the penultimate layer
def slice_model(original_model, from_layer=None, to_layer=None):
    return nn.Sequential(*list(original_model.children())[from_layer:to_layer])


resnet101 = models.resnet101(pretrained=True) # Load the ResNet101 model
model_conv_features = slice_model(resnet101, to_layer=-1) # Remove the last layer
model_conv_features.eval() # Set the model to evaluation mode
print(f"ResNet101 Model Loaded")

def torch_process_image(in_images):
    """
    Preprocess a list of images for the ResNet model.
    """
    # Convert list of NumPy arrays to a batch of preprocessed tensors
    images = [preprocess(Image.fromarray(img)) for img in in_images]  # Preprocess all
    batch = torch.stack(images)  # Shape: (batch_size, 3, 224, 224)

    # Disable gradient tracking for inference
    with torch.no_grad():
        features = model_conv_features(batch)  # Shape: (batch_size, C, H, W)
        features = features.view(features.size(0), -1)  # Flatten each feature map
    # Return as a list of flattened NumPy arrays
    resnet_features = [features[i].numpy() for i in range(features.size(0))]
    return resnet_features


# Process the images through the model
resnet_train_embedding = torch_process_image(train_images)
resnet_val_embedding = torch_process_image(val_images)
resnet_test_embedding = torch_process_image(test_images)

# Export Train Features for Seperate PCA Analysis
np.save(r"C:\Users\mhurth\REPO\MIDS281\mids-281-final-project-cars\FeatureVectors\resnet_train_embedding.npy",
        resnet_train_embedding)

print("ResNet Features Built in: ", time() - tic)
##############################################################################################
############################ Flatten and Stack Features Images ###############################
##############################################################################################
# Flatten all features into a single row
tic = time()


def stack_rows(features):
    """
    Efficiently stacks multiple lists of image features row-wise.
    For each image i, flattens all features and concatenates them into a single row.
    Returns a 2D numpy array.
    """
    stacked = np.array([
        np.concatenate([feature[i].flatten() for feature in features]) for i in range(len(features[0]))])
    return stacked


stacked_features = stack_rows([hog_train, fourier_train, canny_train, resnet_train_embedding])
stacked_features_test = stack_rows([hog_test, fourier_test, canny_test, resnet_test_embedding])
stacked_features_val = stack_rows([hog_val, fourier_val, canny_val, resnet_val_embedding])

del (hog_train, fourier_train, canny_train, hog_test, fourier_test, canny_test, resnet_train_embedding,
     resnet_test_embedding, train_images, test_images)

print("Features Stacked in: ", time() - tic)

##############################################################################################
################################ Principal Component Analysis ################################
##############################################################################################
tic = time()

# Normalize the stacked features
scaler = StandardScaler()
scaler.fit(stacked_features) # Fit the scaler to the training data
# Apply Scalar to the training, validation, and test data
stacked_features = scaler.transform(stacked_features)
stacked_features_val = scaler.transform(stacked_features_val)
stacked_features_test = scaler.transform(stacked_features_test)

# Export Stacked Features for PCA vs. Model Performance Analysis
np.save(r"C:\Users\mhurth\REPO\MIDS281\mids-281-final-project-cars\FeatureVectors\stacked_features.npy",
        stacked_features)
np.save(r"C:\Users\mhurth\REPO\MIDS281\mids-281-final-project-cars\FeatureVectors\stacked_features_val.npy",
        stacked_features_val)
np.save(r"C:\Users\mhurth\REPO\MIDS281\mids-281-final-project-cars\FeatureVectors\stacked_features_test.npy",
        stacked_features_test)

# Run PCA on the stacked features
pca = PCA(n_components=0.95)
pca.fit(stacked_features) # Fit the PCA model to the training data

# Save out the PCA model
with open(r"C:\Users\mhurth\REPO\MIDS281\mids-281-final-project-cars\PCAWithResnetFeatures.pkl", "wb") as f:
    pickle.dump(pca, f)

# # # Load the PCA model
# with open(r"C:\Users\mhurth\REPO\MIDS281\mids-281-final-project-cars\PCAWithResnetFeatures.pkl", "rb") as f:
#     pca = pickle.load(f)

# Plot the explained variance
plt.figure(figsize=(20, 10))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance')
plt.title('Explained Variance vs. Number of Components')
plt.grid()
plt.show()

# Transform the Stacked Features
X = pca.transform(stacked_features)
X_test = pca.transform(stacked_features_test)
X_val = pca.transform(stacked_features_val)

# Trim X to the number of components that explain 90% of the variance
n_components = 2000
X = X[:, :n_components]
X_test = X_test[:, :n_components]
X_val = X_val[:, :n_components]

print("PCA Completed in: ", time() - tic)

##############################################################################################
################################ Exports #####################################################
##############################################################################################

# Save out X, X_test, Targets, and Test_Targets as numpy arrays
np.save(r"C:\Users\mhurth\REPO\MIDS281\mids-281-final-project-cars\XWithResNetFeatures2000PCA.npy", X)
np.save(r"C:\Users\mhurth\REPO\MIDS281\mids-281-final-project-cars\X_valWithResNetFeatures2000PCA.npy", X_val)
np.save(r"C:\Users\mhurth\REPO\MIDS281\mids-281-final-project-cars\X_testWithResNetFeatures2000PCA.npy", X_test)
np.save(r"C:\Users\mhurth\REPO\MIDS281\mids-281-final-project-cars\targets.npy", train_targets)
np.save(r"C:\Users\mhurth\REPO\MIDS281\mids-281-final-project-cars\val_targets.npy", val_targets)
np.save(r"C:\Users\mhurth\REPO\MIDS281\mids-281-final-project-cars\test_targets.npy", test_targets)
