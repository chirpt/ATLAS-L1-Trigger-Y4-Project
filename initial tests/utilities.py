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
import math
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, roc_curve, auc, PrecisionRecallDisplay
import plotly.graph_objects as go


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


def prepare_data(test_size=0.2, accept_data_filename="l1calo_hist_EGZ_extended.root", reject_data_filename="l1calo_hist_ZMUMU_extended.root", data_subdir="ZMUMU_EGZ_extended_np_pd", format_mode="SuperCell_ET", get_pT=True):
    save_path = os.path.join(os.path.pardir, "data", data_subdir)
    if os.path.exists(os.path.join(save_path,"np_data.npz")) and os.path.exists(os.path.join(save_path,"input_df.parquet")):
        print(f"found preprepared data in {save_path}")
        np_data = np.load(os.path.join(save_path,"np_data.npz"))
        input_np, labels_np = np_data["input_np"], np_data["labels_np"]
        input_df = pd.read_parquet(os.path.join(save_path,"input_df.parquet"))


    else:
        print(f"preprepared data in {save_path} is missing, preparing and saving here")
        accept_data_path= os.path.join(os.path.pardir, "data", accept_data_filename)
        reject_data_path= os.path.join(os.path.pardir, "data", reject_data_filename)
        DFs = import_data_files([accept_data_path, reject_data_path])

        accepted_labels = np.ones(DFs[0].shape[0])
        rejected_labels = np.zeros(DFs[1].shape[0])
        accepted_df = pd.DataFrame({'offline_ele_pt': DFs[0]['offline_ele_pt'],'Label': 1})
        rejected_df = pd.DataFrame({'offline_ele_pt': DFs[1]['offline_ele_pt'],'Label': 0})

        input_np = format_numpy_training_input(DFs,format_mode)
        input_df = pd.concat([accepted_df,rejected_df]).reset_index(drop=True)
        labels_np = np.concatenate((accepted_labels, rejected_labels), axis=0)
        
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        np.savez(os.path.join(save_path,"np_data.npz"), input_np=input_np,labels_np=labels_np)
        input_df.to_parquet(os.path.join(save_path,"input_df.parquet"), index=False)

    if get_pT == True:
        X_train, X_test, pd_passthrough_train, pd_passthrough_test, y_train, y_test = train_test_split(input_np, input_df, labels_np, test_size=test_size, random_state=42)
        return X_train, X_test, y_train, y_test, pd_passthrough_train, pd_passthrough_test
    else:
        X_train, X_test, y_train, y_test = train_test_split(input_np, labels_np, test_size=test_size, random_state=42)
        return X_train, X_test, y_train, y_test

def format_numpy_training_input(DFs,format_mode):
    if format_mode == "SuperCell_ET":
        accepted_numpy = ak.to_numpy(DFs[0]['SuperCell_ET'])
        rejected_numpy = ak.to_numpy(DFs[1]['SuperCell_ET'])
        
    elif format_mode == "iso_vars":
        columns = ["eFEX_ReC", "eFEX_ReE", "eFEX_RhE", "eFEX_RhH", "eFEX_WsN", "eFEX_WsD"]
        accepted_numpy = DFs[0][columns].to_numpy(dtype=np.float32)
        rejected_numpy = DFs[1][columns].to_numpy(dtype=np.float32)

    elif format_mode == "reduced_SuperCell_ET":
        full_accepted_numpy = ak.to_numpy(DFs[0]['SuperCell_ET'])
        full_rejected_numpy = ak.to_numpy(DFs[1]['SuperCell_ET'])
        accepted_seed_indices = np.argmax(full_accepted_numpy[:, 49:52 + 1], axis=1) + 49
        rejected_seed_indices = np.argmax(full_rejected_numpy[:, 49:52 + 1], axis=1) + 49

        delete_indices = {
        49: [24, 25, 26, 28, 29, 30, 57, 58, 59, 61, 62, 63, 90, 91, 92, 94, 95, 96],  # Indices to delete for 49
        50: [1, 5, 25, 26, 29, 30, 34, 38, 58, 59, 62, 63, 67, 71, 91, 92, 95, 96],  # Indices to delete for 50
        51: [1, 2, 5, 6, 26, 30, 34, 35, 38, 39, 59, 63, 67, 68, 71, 72, 92, 96],  # Indices to delete for 51
        52: [1, 2, 3, 5, 6, 7, 34, 35, 36, 38, 39, 40, 67, 68, 69, 71, 72, 73]}   # Indices to delete for 52

        accepted_numpy = np.array([np.delete(row, delete_indices[cond]) for row, cond in zip(full_accepted_numpy, accepted_seed_indices)])
        rejected_numpy = np.array([np.delete(row, delete_indices[cond]) for row, cond in zip(full_rejected_numpy, rejected_seed_indices)])
    
    elif format_mode == "topocluster_ET_boundaries":
        print("attempting to generate topo training data")
        accepted_numpy = generate_topocluster_ET_distribution(DFs[0])
        rejected_numpy = generate_topocluster_ET_distribution(DFs[1])


    return np.concatenate((accepted_numpy, rejected_numpy), axis=0)

def dataset_size_equaliser():
    size = max(DFs[0].shape, DFs[i].shape[1])
     

def generate_topocluster_ET_distribution(DF):
    distance_boundaries = [0.1,0.2,0.3,0.4]
    ET_distributions = np.empty((DF.shape[0],len(distance_boundaries)))
    for i in range(DF.shape[0]):
        entry = DF.loc[i]
        TopoCluster_ETs = entry["TopoCluster_ET"]
        TopoCluster_etas = entry["TopoCluster_eta"]
        TopoCluster_phis = entry["TopoCluster_phi"]
        if sum(TopoCluster_ETs) > 0:
            barycentre = calculate_topo_barycentre(TopoCluster_ETs,TopoCluster_etas,TopoCluster_phis)
            topocluster_distances = get_distances_to_barycentre(barycentre,TopoCluster_ETs,TopoCluster_etas,TopoCluster_phis)
            ET_distributions[i] = get_ET_distribution(distance_boundaries,topocluster_distances,TopoCluster_ETs)
        else:
            ET_distributions[i] = [0 for k in range(len(distance_boundaries))]
        if i % 1000 ==0:
            print(round(i/DF.shape[0]*100,1),"%")

    return ET_distributions

def calculate_topo_barycentre(TopoCluster_ETs,TopoCluster_etas,TopoCluster_phis):
    barycentre = [0,0]
    ET_total = sum(TopoCluster_ETs)
    barycentre[0] = sum(x * m for x, m in zip(TopoCluster_etas, TopoCluster_ETs)) / ET_total
    barycentre[1] = sum(y * m for y, m in zip(TopoCluster_phis, TopoCluster_ETs)) / ET_total
    return barycentre

def get_distances_to_barycentre(barycentre,TopoCluster_ETs,TopoCluster_etas,TopoCluster_phis):
    topocluster_distances = [0 for i in range(len(TopoCluster_ETs))]
    for i in range(len(TopoCluster_ETs)):
        topocluster_distance = math.sqrt((TopoCluster_etas[i]-barycentre[0])**2+(TopoCluster_phis[i]-barycentre[1])**2)
        topocluster_distances[i] = topocluster_distance
    return topocluster_distances

def get_ET_distribution(distance_boundaries,topocluster_distances,TopoCluster_ETs):
    lower_boundry = 0
    topocluster_ET_distribution = [0 for i in range(len(distance_boundaries))]
    for i, upper_boundry in enumerate(distance_boundaries):
        for j, topocluster_distance in enumerate(topocluster_distances):
            if topocluster_distance > lower_boundry and topocluster_distance <= upper_boundry:
                topocluster_ET_distribution[i] += TopoCluster_ETs[j]
        lower_boundry = upper_boundry
    return np.array(topocluster_ET_distribution)

def visualise_topocluster_ETs(DF,num_bins):
    all_topocluster_ET_distances = []
    for i in range(DF.shape[0]):
        entry = DF.loc[i]
        TopoCluster_ETs = entry["TopoCluster_ET"]
        TopoCluster_etas = entry["TopoCluster_eta"]
        TopoCluster_phis = entry["TopoCluster_phi"]
        if sum(TopoCluster_ETs) > 0:
            barycentre = calculate_topo_barycentre(TopoCluster_ETs,TopoCluster_etas,TopoCluster_phis)
            topocluster_distances = get_distances_to_barycentre(barycentre,TopoCluster_ETs,TopoCluster_etas,TopoCluster_phis)
            for j, topocluster_distance in enumerate(topocluster_distances):
                topocluster_ET = TopoCluster_ETs[j]
                all_topocluster_ET_distances.append([topocluster_distance,topocluster_ET])
        else:
            pass
        if i % 1000 ==0:
            print(round(i/DF.shape[0]*100,1),"%")

    all_topocluster_ET_distances_np = np.array(all_topocluster_ET_distances)
    hist, xedges, yedges = np.histogram2d(all_topocluster_ET_distances_np[:,0], all_topocluster_ET_distances_np[:,1], bins=num_bins)
    xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
    fig = go.Figure(data=[go.Surface(z=hist, x=xpos, y=ypos)])
    fig.update_layout(
    title="3D Histogram (30 bins)",
    scene=dict(
        xaxis_title='(Distance from Barycentre)',
        yaxis_title='(Topocluster ET)',
        zaxis_title='Frequency',
        xaxis=dict(range=[0, 0.5]),
        yaxis=dict(range=[0, 100_000])
        )
    )

    fig.show()





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

def evaluate_sklearn_model(y_test, y_pred,show_CR=True,show_MSE=True,model_name=None):
    if model_name != None:
        print("Evaluation of "+model_name)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    if show_CR:
        print("Classification Report:\n",classification_report(y_test, y_pred))
    print("Confusion Matrix:\n",confusion_matrix(y_test, y_pred))
    if show_MSE:
        print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.4f}\n")

def compute_roc(model, X_test, y_test):
    if hasattr(model, "decision_function"):  # For models like SVM or SGD
        y_scores = model.decision_function(X_test)
    elif hasattr(model, "predict_proba"):  # For models like XGBoost or other tree-based models
        y_scores = model.predict_proba(X_test)[:, 1]
    else:
        raise ValueError("Model must have either a decision_function or predict_proba method.")
    
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
    
def precision_recall(model, X_test, y_test):
    if hasattr(model, "decision_function"):  # For models like SVM or SGD
        y_scores = model.decision_function(X_test)
    elif hasattr(model, "predict_proba"):  # For models like XGBoost or other tree-based models
        y_scores = model.predict_proba(X_test)[:, 1]
    else:
        raise ValueError("Model must have either a decision_function or predict_proba method.")
    
    display = PrecisionRecallDisplay.from_predictions(
    y_test, y_scores, name=model, plot_chance_level=True)
    _ = display.ax_.set_title("2-class Precision-Recall curve")

def plot_efficiency_vs_ele_PT(X_test_all, et_Low = 20, et_High = 60,prediction_parameter="pred",title_string="model"):
    electrons_all,bins,_ = plt.hist( X_test_all.query("Label == 1")["offline_ele_pt"],bins=40,alpha=0.6,range=[et_Low,et_High])
    electrons_all_tagged,bins,_ = plt.hist( X_test_all.query(f"Label == 1 & {prediction_parameter} == 1")["offline_ele_pt"],bins=40,alpha=0.6,range=[et_Low,et_High])
    plt.clf()
    electrons_efficiency = electrons_all_tagged/electrons_all

    plt.bar(bins[:-1], electrons_efficiency, width = np.diff(bins), align='edge', edgecolor='black')
    plt.title(f'efficiency against electron PT for {title_string}')
    plt.xlabel('offline electron pt')
    plt.ylabel('Efficiency')
    plt.xlim(bins[0], bins[-1])
    plt.show()

def multi_roc(classifiers, X_test, y_test, classifiers_2=None, X_test_2=None, y_test_2=None):
    plt.figure(figsize=(10, 10))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    for classifier in classifiers:
        model = classifiers[classifier]
        fpr, tpr, roc_auc = compute_roc(model,X_test,y_test)
        plt.plot(fpr, tpr, lw=1, label=classifier+f" ROC (AUC = {roc_auc:.4f})")

    if classifiers_2 != None:
        for classifier in classifiers_2:
            model = classifiers_2[classifier]
            fpr, tpr, roc_auc = compute_roc(model,X_test_2,y_test_2)
            plt.plot(fpr, tpr, lw=1, label=classifier+f" ROC (AUC = {roc_auc:.4f})")

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig("multi_roc", dpi=400, bbox_inches='tight')
    plt.show()
