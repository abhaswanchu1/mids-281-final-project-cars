'''
This Script Runs TSNE and PCA for each Feature

TSNE plots will be plotted individually and colored by the target variables

PCA curves will be plotted to a single image
'''

# Import Libraries
import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Import Feature Vectors and The Targets
hog_features = np.load(r"C:\Users\mhurth\REPO\MIDS281\mids-281-final-project-cars\FeatureVectors\hog_train.npy")
fourier_features = np.load(r"C:\Users\mhurth\REPO\MIDS281\mids-281-final-project-cars\FeatureVectors\fourier_train.npy")
canny_features = np.load(r"C:\Users\mhurth\REPO\MIDS281\mids-281-final-project-cars\FeatureVectors\canny_train.npy")
resnet_features = np.load(r"C:\Users\mhurth\REPO\MIDS281\mids-281-final-project-cars\FeatureVectors\resnet_train_embedding.npy")
targets = np.load(r"C:\Users\mhurth\REPO\MIDS281\mids-281-final-project-cars\targetsWithResNetFeatures2000PCA.npy")

# Standardize the features
scaler = StandardScaler()
hog_features = scaler.fit_transform(hog_features)
fourier_features = scaler.fit_transform(fourier_features.reshape(7060, 224*224))
canny_features = scaler.fit_transform(canny_features.reshape(7060, 224*224))
resnet_features = scaler.fit_transform(resnet_features)

#########################################################
##### Generate TSNE for hand and ANN features ###########
#########################################################

# Color Dictionary for TSNE plots
color_dict = {
    'Audi': '#0072B2',
    'BMW': '#D55E00',
    'Chevrolet': '#009E73',
    'Dodge': '#CC79A7',
}

def TSNE_PLOT(features, targets, title):
    '''
    function takes input vector and conducts TSNE on the features
    plots the results with the corresponding target labels
    '''
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    colors = [color_dict[label] for label in targets]
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=colors, alpha=0.6)

    # Legend with IBM colors
    legend_patches = [mpatches.Patch(color=color, label=label) for label, color in color_dict.items()]
    plt.legend(handles=legend_patches, title='Car Brands')

    plt.title(title)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.show()

# Plot t-SNE for each feature
TSNE_PLOT(hog_features, targets, 't-SNE of HOG Features')
TSNE_PLOT(fourier_features, targets, 't-SNE of Fourier Features')
TSNE_PLOT(canny_features, targets, 't-SNE of Canny Features')
TSNE_PLOT(resnet_features, targets, 't-SNE of ResNet Features')

#########################################################
##### Generate PCA for hand and ANN features ############
#########################################################

# Import PCA on All Features from Preprocessing
with open(r"C:\Users\mhurth\REPO\MIDS281\mids-281-final-project-cars\PCAWithResnetFeatures.pkl", "rb") as f:
    pca_all = pickle.load(f)

PCA_hog = PCA(n_components=0.95, random_state=42)
PCA_hog.fit(hog_features) # HOG PCA

PCA_fourier = PCA(n_components=0.95, random_state=42)
PCA_fourier.fit(fourier_features) # Fourier PCA

PCA_canny = PCA(n_components=0.95, random_state=42)
PCA_canny.fit(canny_features) # Canny PCA

PCA_resnet = PCA(n_components=0.95, random_state=42)
PCA_resnet.fit(resnet_features) # ResNet PCA

def plot_PCA(model, feature):
    '''
    function takes PCA model and plots the explained variance ratio
    '''
    plt.figure(figsize=(10, 8))
    plt.plot(np.cumsum(model.explained_variance_ratio_))
    plt.axhline(y=0.95, color='r', linestyle='--')
    plt.title(f'{feature} PCA Explained Variance Ratio')
    plt.xlabel('Principal Component Index')
    plt.ylabel('Fraction of Total Variance Explained')
    plt.show()

for model, feature in zip([PCA_hog, PCA_fourier, PCA_canny, PCA_resnet], ['HOG', 'Fourier', 'Canny', 'ResNet']):
    plot_PCA(model, feature)


# Plot PCA curves for features together
plt.figure(figsize=(10, 8))
plt.plot(np.cumsum(PCA_hog.explained_variance_ratio_), label='HOG Features')
plt.plot(np.cumsum(PCA_fourier.explained_variance_ratio_), label='Fourier Features')
plt.plot(np.cumsum(PCA_canny.explained_variance_ratio_), label='Canny Features')
plt.plot(np.cumsum(PCA_resnet.explained_variance_ratio_), label='ResNet Features')
plt.plot(np.cumsum(pca_all.explained_variance_ratio_), label='All Features')
plt.axhline(y=0.95, color='r', linestyle='--')
plt.title('PCA Explained Variance Ratio')
plt.xlabel('Principal Component Index')
plt.ylabel('Fraction of Total Variance Explained')
plt.legend()
plt.show()

