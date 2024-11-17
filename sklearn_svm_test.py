import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import awkward as ak
from utilities import *

# Generate synthetic data for demonstration
# Assume you have loaded data as `data` (features) and `labels` (binary: 0 for 'rejected', 1 for 'accepted')

# Example: Replace this with actual data loading
# np.random.seed(42)
# data = np.random.randn(10000, 99)  # 10,000 points in 99-dimensional space
# labels = np.random.choice([0, 1], size=(10000,))  # Binary labels: 0 or 1

DFs = import_data_files(["l1calo_hist_ZMUMU_extended.root", "l1calo_hist_EGZ_extended.root"])

accepted_numpy = ak.to_numpy(DFs[0]['SuperCell_ET'])
rejected_numpy = ak.to_numpy(DFs[1]['SuperCell_ET'])
print("loaded data, converting to numpy")
accepted_labels = np.zeros(accepted_numpy.shape[0])
rejected_labels = np.ones(rejected_numpy.shape[0])
print("converted to numpy, shuffling data")
data = np.concatenate((accepted_numpy,rejected_numpy),axis=0)
labels = np.concatenate((accepted_labels,rejected_labels),axis=0)
np.random.seed(42)
np.random.shuffle(data)
np.random.seed(42)
np.random.shuffle(labels)
print("data shape",data.shape,"labels shape",labels.shape,"\nsplitting training and testing data")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Initialize the SVM classifier
svm_classifier = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
print("training...")
# Train the classifier
svm_classifier.fit(X_train, y_train)
print("running on test data...")
# Predict on the test set
y_pred = svm_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
