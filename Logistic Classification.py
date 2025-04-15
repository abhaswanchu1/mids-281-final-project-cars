# Import Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, roc_auc_score
from sklearn.metrics import roc_curve, auc, f1_score, accuracy_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import ParameterGrid
import pickle
from time import time

##############################################################################################
########################  Load Images  ########################################################
##############################################################################################
# # Load the train and test data Without ResNet Features
# X = np.load(r"C:\Users\mhurth\REPO\MIDS281\mids-281-final-project-cars\X.npy")
# X_test = np.load(r"C:\Users\mhurth\REPO\MIDS281\mids-281-final-project-cars\X_test.npy")
# Y = np.load(r"C:\Users\mhurth\REPO\MIDS281\mids-281-final-project-cars\targets.npy")
# Y_test = np.load(r"C:\Users\mhurth\REPO\MIDS281\mids-281-final-project-cars\test_targets.npy")

# Load the train and test data
X = np.load(r"C:\Users\mhurth\REPO\MIDS281\mids-281-final-project-cars\XWithResNetFeatures2000PCA.npy")
X_val = np.load(r"C:\Users\mhurth\REPO\MIDS281\mids-281-final-project-cars\X_valWithResNetFeatures2000PCA.npy")
X_test = np.load(r"C:\Users\mhurth\REPO\MIDS281\mids-281-final-project-cars\X_testWithResNetFeatures2000PCA.npy")
Y = np.load(r"C:\Users\mhurth\REPO\MIDS281\mids-281-final-project-cars\targets.npy")
Y_val = np.load(r"C:\Users\mhurth\REPO\MIDS281\mids-281-final-project-cars\val_targets.npy")
Y_test = np.load(r"C:\Users\mhurth\REPO\MIDS281\mids-281-final-project-cars\test_targets.npy")

# Filter Down to 500 Components
X = X[:, :500]
X_val = X_val[:, :500]
X_test = X_test[:, :500]

##############################################################################################
################################ HyperParam Grid Search  #####################################
##############################################################################################
# # # Set up the hyperparameter grid
# param_grid = {'max_iter': [100, 500, 1000, 2000, 4000],
#               'solver' : ['lbfgs', 'saga']}
#
# # # Initialize the SVM classifier
# clf = LogisticRegression(multi_class='multinomial', n_jobs=-1)
#
# # Prep Data for AUC Score
# label_binarizer = LabelBinarizer().fit(Y)
# y_onehot_train = label_binarizer.transform(Y)
# y_onehot_val = label_binarizer.transform(Y_val)
# n_classes = y_onehot_train.shape[1]
#
# # # Perform grid search with ParameterGrid and a loop to iterate through the parameters
# results = []
# for param in ParameterGrid(param_grid):
#     # Set the parameters for the classifier
#     clf.set_params(**param)
#     # Fit the model on the training data
#     clf.fit(X, Y)
#     # Calculate the AUC score, Accuracy, and Weighted F1 Score on the Training and Validation data
#     train_score = clf.predict_proba(X)
#     fpr, tpr, _ = roc_curve(y_onehot_train.ravel(), train_score.ravel())
#     auc_train = auc(fpr, tpr)
#     accuracy_train = accuracy_score(Y, clf.predict(X))
#     f1_train = f1_score(Y, clf.predict(X), average='weighted')
#     # Val
#     val_score = clf.predict_proba(X_val)
#     fpr, tpr, _ = roc_curve(y_onehot_val.ravel(), val_score.ravel())
#     auc_val = auc(fpr, tpr)
#     accuracy_val = accuracy_score(Y_val, clf.predict(X_val))
#     f1_val = f1_score(Y_val, clf.predict(X_val), average='weighted')
#     # Append the results to the list
#     results.append({
#         'max_iter': param['max_iter'],
#         'solver': param['solver'],
#         'Train AUC': auc_train,
#         'Train Accuracy': accuracy_train,
#         'Train F1 Score': f1_train,
#         'Val AUC': auc_val,
#         'Val Accuracy': accuracy_val,
#         'Val F1 Score': f1_val
#     })
#     # Print the results
#     print(f"Max Iter: {param['max_iter']}, Solver: {param['solver']}")
#     print(f"Train AUC: {auc_train:.2f}, Train Accuracy: {accuracy_train:.2f}, Train F1 Score: {f1_train:.2f}")
#     print(f"Val AUC: {auc_val:.2f}, Val Accuracy: {accuracy_val:.2f}, Val F1 Score: {f1_val:.2f}")
#
# # Convert the results to a DataFrame
# results_df = pd.DataFrame(results)
# results_df.to_csv(r"C:\Users\mhurth\REPO\MIDS281\mids-281-final-project-cars\LogReg_ParamGridSearch_Results.csv",
#                   index=False)

##############################################################################################
################################ Classification With Optimal Model  ##########################
##############################################################################################


# Run SVM on X with the targets
clf = LogisticRegression(multi_class='multinomial', n_jobs=-1, max_iter=1000, solver='saga')
tic = time()
clf.fit(X, Y)
train_time = time() - tic
tic = time()
preds = clf.predict(X)
pred_time = time() - tic
tic = time()
preds_val = clf.predict(X_val)
val_pred_time = time() - tic
tic = time()
preds_test = clf.predict(X_test)
preds_test_time = time() - tic

# Print the training time, prediction time, and validation prediction time
print(f"Trained SVM in {train_time} seconds")
print(f"Predicted SVM in {pred_time:.10f} seconds")
print(f"Predicted SVM on validation set in {val_pred_time:.10f} seconds")
print(f"Predicted SVM on test set in {preds_test_time:.10f} seconds")

# Print the Weighted F1 Score on the Training, Validation, and Test data
f1_train = f1_score(Y, preds, average='weighted')
f1_val = f1_score(Y_val, preds_val, average='weighted')
f1_test = f1_score(Y_test, preds_test, average='weighted')
print(f"Weighted F1 Score on Training Data: {f1_train:.2f}")
print(f"Weighted F1 Score on Validation Data: {f1_val:.2f}")
print(f"Weighted F1 Score on Test Data: {f1_test:.2f}")

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
    plt.plot(fpr[i], tpr[i], label=f'ROC curve of class {clf.classes_[i]} (AUC = {roc_auc[i]:.2f})')
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



