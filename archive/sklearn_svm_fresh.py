import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.manifold import TSNE
import awkward as ak
from utilities import *
import matplotlib.pyplot as plt
import seaborn as sns

# Load and prepare the data
DFs = import_data_files(["data/l1calo_hist_ZMUMU_extended.root", "data/l1calo_hist_EGZ_extended.root"])

accepted_numpy = ak.to_numpy(DFs[0]['SuperCell_ET'])
rejected_numpy = ak.to_numpy(DFs[1]['SuperCell_ET'])
print("loaded data, converting to numpy")
accepted_labels = np.zeros(accepted_numpy.shape[0])
rejected_labels = np.ones(rejected_numpy.shape[0])
print("converted to numpy, shuffling data")
data = np.concatenate((accepted_numpy, rejected_numpy), axis=0)
labels = np.concatenate((accepted_labels, rejected_labels), axis=0)
np.random.seed(42)
np.random.shuffle(data)
np.random.seed(42)
np.random.shuffle(labels)
print("data shape", data.shape, "labels shape", labels.shape, "\nsplitting training and testing data")

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

# t-SNE Visualization
print("Applying t-SNE for visualization...")
tsne = TSNE(n_components=2, random_state=42)
# Combine train and test data for visualization
X_combined = np.concatenate((X_train, X_test), axis=0)
y_combined_true = np.concatenate((y_train, y_test), axis=0)
y_combined_pred = np.concatenate((svm_classifier.predict(X_train), y_pred), axis=0)

X_embedded = tsne.fit_transform(X_combined.reshape(-1, 1))  # Reshape needed if data is 1D

# Plot t-SNE results with true labels
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=y_combined_true, palette="coolwarm", alpha=0.7)
plt.title("t-SNE Visualization with True Labels")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.show()

# Plot t-SNE results with predicted labels
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=y_combined_pred, palette="coolwarm", alpha=0.7)
plt.title("t-SNE Visualization with Predicted Labels")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.show()

# Highlight misclassified points
misclassified = (y_combined_true != y_combined_pred)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=misclassified, palette={True: 'red', False: 'blue'}, alpha=0.7)
plt.title("t-SNE with Misclassified Points Highlighted")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.show()
