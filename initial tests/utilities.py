import os
import pickle
import uproot
import pandas as pd
import numpy as np
import uproot
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import plotly.express as px
import awkward as ak
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, roc_curve, auc



def import_data(file):
    File = uproot.open(file)
    Tree = File["tree_DMC"]
    DF = Tree.arrays(library="pd")
    return DF

def import_data_files(data_filenames):
    data_files = []
    for data_filename in data_filenames:
        data_files.append(import_data(data_filename))
    return data_files

def visualise_ROI(dataframe_entry, isdataframe=True):
    if isdataframe:
        entry = ak.to_numpy(dataframe_entry)

    map0 = [0,11,22,33,44,55,66,77,88]    
    map1 = [1,2,3,4,12,13,14,15,23,24,25,26,34,35,36,37,45,46,47,48,56,57,58,59,67,68,69,70,78,79,80,81,89,90,91,92]
    map2 = [5,6,7,8,16,17,18,19,27,28,29,30,38,39,40,41,49,50,51,52,60,61,62,63,71,72,73,74,82,83,84,85,93,94,95,96]    
    map3 = [9,20,31,42,53,64,75,86,97]    
    mapH = [10,21,32,43,54,65,76,87,98]    

    cell_values = pd.DataFrame(columns=['eta', 'phi', 'r', 'value'])

    for cell_number, value in enumerate(entry):

        if cell_number in map0:
            r=0
            eta = 4* (map0.index(cell_number) %3)+ 1.5
        elif cell_number in map1:
            r=1
            eta = map1.index(cell_number) %12
        elif cell_number in map2:
            r=2
            eta = map2.index(cell_number) %12
        elif cell_number in map3:
            r=3
            eta = 4* (map3.index(cell_number) %3) + 1.5
        elif cell_number in mapH:
            r=4
            eta = 4 * (mapH.index(cell_number) %3) + 1.5

        phi = cell_number // 33

        cell_values.loc[len(cell_values.index)] = [eta, phi, r,value] 

    fig = px.scatter_3d(cell_values, x="eta", y="phi", z="r", color="value")
    fig.show()

def prepare_data(test_size=0.2, accept_data_filename="l1calo_hist_EGZ_extended.root", 
                 reject_data_filename="l1calo_hist_ZMUMU_extended.root", 
                 save_path="prepared_data.npz"):
    accept_data_filename = os.path.join(os.path.pardir, "data", accept_data_filename)
    reject_data_filename = os.path.join(os.path.pardir, "data", reject_data_filename)
    save_path = os.path.join(os.path.pardir, "data", save_path)

    if not os.path.exists(save_path):
        print("Preparing data...")
        DFs = import_data_files([accept_data_filename, reject_data_filename])

        accepted_numpy = ak.to_numpy(DFs[0]['SuperCell_ET'])
        rejected_numpy = ak.to_numpy(DFs[1]['SuperCell_ET'])

        accepted_labels = np.ones(accepted_numpy.shape[0])
        rejected_labels = np.zeros(rejected_numpy.shape[0])

        data = np.concatenate((accepted_numpy, rejected_numpy), axis=0)
        labels = np.concatenate((accepted_labels, rejected_labels), axis=0)
        
        np.random.seed(42)
        np.random.shuffle(data)
        np.random.seed(42)
        np.random.shuffle(labels)
        
        print(f"Saving prepared data to {save_path}")
        np.savez(save_path, data=data, labels=labels)
    
    else:
            print(f"Loading prepared data from {save_path}")
        
    data = np.load(save_path)
        
    X_train, X_test, y_train, y_test = train_test_split(data['data'], data['labels'], test_size=test_size, random_state=42)

    return X_train, X_test, y_train, y_test

def plot_2D_TSNE(embedded_data, colour_var, title):
    plt.figure(figsize=(10, 6))
    plt.scatter(embedded_data[:, 0], embedded_data[:, 1], c=colour_var, cmap='viridis', s=5)
    plt.colorbar(label="1 - accepted, 0 - rejected")
    plt.title(title)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.show()

def plot_3D_TSNE(embedded_data, colour_var,point_size=2):
    fig = px.scatter_3d(x=embedded_data[:, 0], y=embedded_data[:, 1], z=embedded_data[:, 2], color=colour_var)
    fig.update_traces(marker=dict(line=dict(width=0),size=np.ones(colour_var.shape)*point_size))
    fig.show()

def evaluate_sklearn_model(y_test, y_pred):
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Classification Report:\n",classification_report(y_test, y_pred))
    print("Confusion Matrix:\n",confusion_matrix(y_test, y_pred))
    print("Mean Squared Error:\n",mean_squared_error(y_test, y_pred))

# XX Works for SVC and SGB so far
def compute_roc(model, X_test, y_test):
    y_scores = model.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc

def plot_roc(fpr, tpr, roc_auc):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()