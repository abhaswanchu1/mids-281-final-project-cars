# MIDS 281 Final Project - Cars

Welcome to the MIDS 281 Final Project repository! This project uses hand crafted features, pretrained ResNet embeddings, Principal Component Analysis, TSNE, and simple models like LogReg and SVM to classify the Make of vehicles based on images in the Stanford Cars Dataset

## Scripts
- CarsTrainTestSplit.py gathers the images from the original data source train and test folder, pulls the car make, and splits the data into train and test
- PreProcessing.py Loads the images, splits into train-val-test, preprocesses (crop, resize), augements the training data, generated hand features (HOG, Fuorier, Canny Edges), generates ResNet101 Embedings, Stacks Features, Runs PCA, and Exports Vectors for model exploration
- TSNEbyFeature.py Shows TSNE and PCA analysis for each individual feature generated and combines all seperate PCA analyses into a single vector
- PCAvsModelPerformance.py Contains sensitivites of model perfomance for SVM and Logistic Classifiers for different number of input principal components.
- SVM Classification.py Conducts a HyperParameter GridSearch on the SVM Model, Fits a Model on the best parameters, and generates fit statistics, compute-time statistics, ROC-AUC plots, and confusion matrices for the test sets
- Logistic Classification.py Conducts a HyperParameter GridSearch on the Logistic Classifier Model, Fits a Model on the best parameters, and generates fit statistics, compute-time statistics, ROC-AUC plots, and confusion matrices for the test sets
- RandomForestClassification.py is an early exploration of classification performance with RandomForestClassifier with ROC-AUC Plots.
- KNN Classification.py is an an early exploration of KNN performance with RandomForestClassifier with ROC-AUC Plots.

