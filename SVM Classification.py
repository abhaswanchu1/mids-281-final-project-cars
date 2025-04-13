# Import Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, roc_auc_score
from sklearn.metrics import roc_curve, auc

##############################################################################################
########################  Load Images  ########################################################
##############################################################################################
# Load the train and test data
X = np.load(r"C:\Users\mhurth\REPO\MIDS281\mids-281-final-project-cars\X.npy")
X_test = np.load(r"C:\Users\mhurth\REPO\MIDS281\mids-281-final-project-cars\X_test.npy")
Y = np.load(r"C:\Users\mhurth\REPO\MIDS281\mids-281-final-project-cars\targets.npy")
Y_test = np.load(r"C:\Users\mhurth\REPO\MIDS281\mids-281-final-project-cars\test_targets.npy")


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

# Display the ROC Curve for each of the 4 classes
# Get the probabilities for each class
probs = clf.predict_proba(X_test)
probs # Make Binary
classes = clf.classes_
# Calculate the ROC for each class
for i in range(len(classes)):
    fpr, tpr, _ = roc_curve(Y_test, probs[:, i], pos_label=classes[i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{classes[i]} (AUC = {roc_auc:.2f})')
    # Add a legend
    plt.legend(loc='lower right')
# Add a title and labels
plt.title('Receiver Operating Characteristic')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.show()



