# Import Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelBinarizer

##############################################################################################
########################  Load Images  ########################################################
##############################################################################################
# Load the train and test data Without ResNet Features
X = np.load(r"C:\Users\mhurth\REPO\MIDS281\mids-281-final-project-cars\X.npy")
X_test = np.load(r"C:\Users\mhurth\REPO\MIDS281\mids-281-final-project-cars\X_test.npy")
Y = np.load(r"C:\Users\mhurth\REPO\MIDS281\mids-281-final-project-cars\targets.npy")
Y_test = np.load(r"C:\Users\mhurth\REPO\MIDS281\mids-281-final-project-cars\test_targets.npy")

# Load the train and test data
X = np.load(r"C:\Users\mhurth\REPO\MIDS281\mids-281-final-project-cars\XWithResNetFeatures.npy")
X_test = np.load(r"C:\Users\mhurth\REPO\MIDS281\mids-281-final-project-cars\X_testWithResNetFeatures.npy")
Y = np.load(r"C:\Users\mhurth\REPO\MIDS281\mids-281-final-project-cars\targetsWithResNetFeatures.npy")
Y_test = np.load(r"C:\Users\mhurth\REPO\MIDS281\mids-281-final-project-cars\test_targetsWithResNetFeatures.npy")


##############################################################################################
################################ Classification  #############################################
##############################################################################################
# tic = time()

# Run SVM on X with the targets
clf = svm.SVC(kernel='linear', probability=True)
clf.fit(X, Y)
preds = clf.predict(X)

# Display the Confusion Matrix on Training Data
cm = confusion_matrix(Y, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.show()

# Display the Confusion Matrix on Test Data
preds_test = clf.predict(X_test)
cm_test = confusion_matrix(Y_test, preds_test)
disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=clf.classes_)
disp_test.plot(cmap=plt.cm.Blues)
plt.show()

##############################################################################################
################################ ROC Displays  ###############################################
##############################################################################################

# Display the ROC Curve for each of the 4 classes
# Get the probabilities for each class
'''
Next 22 Lines of Code (ROC Curve Prep) modified from 
https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
'''
label_binarizer = LabelBinarizer().fit(Y)
y_onehot_test = label_binarizer.transform(Y_test)
y_score = clf.predict_proba(X_test)
n_classes = y_onehot_test.shape[1]
# store the fpr, tpr, and roc_auc for all averaging strategies
fpr, tpr, roc_auc = dict(), dict(), dict()
# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_onehot_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
# print(f"Micro-averaged One-vs-Rest ROC AUC score:\n{roc_auc['micro']:.2f}")
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_onehot_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
fpr_grid = np.linspace(0.0, 1.0, 1000)
# Interpolate all ROC curves at these points
mean_tpr = np.zeros_like(fpr_grid)
for i in range(n_classes):
    mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation
# Average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = fpr_grid
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
# print(f"Macro-averaged One-vs-Rest ROC AUC score:\n{roc_auc['macro']:.2f}")

# Plot ROC Curves for each label and the micro and macro averages
plt.figure(figsize=(10, 8))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f'ROC curve of class {i} (AUC = {roc_auc[i]:.2f})')
plt.plot(fpr["micro"], tpr["micro"], label=f'Micro-average ROC curve (AUC = {roc_auc["micro"]:.2f})',
            linestyle=':', linewidth=2)
plt.plot(fpr["macro"], tpr["macro"], label=f'Macro-average ROC curve (AUC = {roc_auc["macro"]:.2f})',
            linestyle='--', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
plt.xlim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()



