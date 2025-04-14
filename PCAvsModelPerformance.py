'''
This script will use the stacked features and correponding PCA feature

We will train an SVM at a varying number of components and capture the time of training, time of prediction,
and the accuracy of the model on the train and validation data, put this all into a table and export

'''
# Import Packages
import pickle
import numpy as np
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
import time
import pandas as pd
import matplotlib.pyplot as plt
print("imported packages")

# Import PCA Model
with open(r"C:\Users\mhurth\REPO\MIDS281\mids-281-final-project-cars\PCAWithResnetFeatures.pkl", "rb") as f:
    pca = pickle.load(f)
print("loaded PCA model")

# Import Feature and Target Vectors
standard_scaled_features = np.load(r"C:\Users\mhurth\REPO\MIDS281\mids-281-final-project-cars\FeatureVectors\stacked_features.npy")
standard_scaled_features_val = np.load(r"C:\Users\mhurth\REPO\MIDS281\mids-281-final-project-cars\FeatureVectors\stacked_features_val.npy")
targets = np.load(r"C:\Users\mhurth\REPO\MIDS281\mids-281-final-project-cars\targets.npy")
val_targets = np.load(r"C:\Users\mhurth\REPO\MIDS281\mids-281-final-project-cars\val_targets.npy")
pca_features = pca.transform(standard_scaled_features) # PCA Transform the Standard Scaled Features
pca_features_val = pca.transform(standard_scaled_features_val) # PCA Transform the Standard Scaled Features
print("loaded data")

# Set Up the AUC Score
label_binarizer = LabelBinarizer().fit(targets)
y_onehot_train = label_binarizer.transform(targets)
y_onehot_val = label_binarizer.transform(val_targets)
n_classes = y_onehot_train.shape[1]

# Iterate through the different PCA Components
results = []
comps = [3000, 2000, 1500, 1000, 500, 250, 100, 50, 25, 1]
for comp_num in comps:
    # Clip to N Components
    X = pca_features[:, :comp_num]
    print(X.shape)
    # Fit the SVM (and Time it)
    tic = time.time()
    # clf = LogisticRegression(max_iter=3000, multi_class='multinomial', solver='lbfgs') # Uncomment for LogReg
    clf = svm.SVC(kernel='linear', probability=True, max_iter=3000) # Uncomment for SVM
    clf.fit(X, targets)
    toc = time.time()
    train_time = toc - tic
    print(f"Trained SVM with {comp_num} components in {train_time:.2f} seconds")
    # Predict the SVM (and Time it)
    tic = time.time()
    preds = clf.predict(X)
    toc = time.time()
    pred_time = toc - tic
    tic = time.time()
    preds_val = clf.predict(pca_features_val[:, :comp_num])
    toc = time.time()
    val_pred_time = toc - tic
    ### Calculate the micro-averaged AUC and Average
    # Train
    train_score = clf.predict_proba(X)
    fpr, tpr, _ = roc_curve(y_onehot_train.ravel(), train_score.ravel())
    auc_train = auc(fpr, tpr)
    accuracy_train = clf.score(X, targets)
    # Val
    val_score = clf.predict_proba(pca_features_val[:, :comp_num])
    fpr, tpr, _ = roc_curve(y_onehot_val.ravel(), val_score.ravel())
    auc_val = auc(fpr, tpr)
    accuracy_val = clf.score(pca_features_val[:, :comp_num], val_targets)
    # Store the Results
    results.append({
        'Components': comp_num,
        'Train Time': train_time,
        'Pred Time': pred_time,
        'Val Pred Time': val_pred_time,
        'Train AUC': auc_train,
        'Train Accuracy': accuracy_train,
        'Val AUC': auc_val,
        'Val Accuracy': accuracy_val
    })

# Convert to DataFrame
results_df = pd.DataFrame(results)

# # Save to CSV
results_df.to_csv(r"C:\Users\mhurth\REPO\MIDS281\mids-281-final-project-cars\PCA_Component_Sensitivity_SVM.csv",
                  index=False)

# Plot the Results with Components on X-Axis, time on right Y-axis, and AUC on left Y-axis
fig, ax1 = plt.subplots(figsize=(10, 6))
# Plot Train AUC
ax1.plot(results_df['Components'], results_df['Train AUC'], label='Train AUC', color='#D55E00', marker='o')
ax1.plot(results_df['Components'], results_df['Val AUC'], label='Val AUC', color='#009E73', marker='o')
ax1.set_xlabel('Number of PCA Components')
ax1.set_ylabel('AUC Score')
ax1.set_title('PCA Component Sensitivity Analysis (SVM)')
ax1.legend(loc='upper left')
# Create a second y-axis for the time
ax2 = ax1.twinx()
# Plot Train Time
ax2.plot(results_df['Components'], results_df['Train Time'], label='Train Time', color="#CC79A7", marker='o')
ax2.plot(results_df['Components'], results_df['Pred Time'], label='Prediction Time', color='#0072B2', marker='o')
ax2.set_ylabel('Time (seconds)')
ax2.legend(loc='upper right')
plt.show()





