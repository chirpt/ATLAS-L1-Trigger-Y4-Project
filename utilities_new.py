import os
import csv
import json
import time
import math
import pickle
import uproot
import numpy as np
import pandas as pd
import awkward as ak
import multiprocessing

from tqdm.auto import tqdm
from datetime import datetime
from inspect import signature

import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from sklearn.base import ClassifierMixin
from sklearn.utils import all_estimators
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, roc_curve, auc, PrecisionRecallDisplay, recall_score, precision_score, f1_score, precision_recall_curve

from xgboost import XGBClassifier


#Import
def import_data(file):
    File = uproot.open(file)
    if "tree_DMC" not in File:
        raise KeyError(f"'tree_DMC' not found in {file}")
    Tree = File["tree_DMC"]
    DF = Tree.arrays(library="pd")
    return DF

def import_data_files(data_filenames):
    data_files = []
    for data_filename in data_filenames:
        data_files.append(import_data(data_filename))
    return data_files

def import_all_classifiers():
    all_estimators_list = all_estimators(type_filter="classifier")

    binary_classifiers = {}
    default_base_estimator = LogisticRegression()

    for name, Classifier in all_estimators_list:
        try:
            # Handle meta-estimators requiring base estimators or parameters
            if name in [
                "ClassifierChain",
                "FixedThresholdClassifier",
                "MultiOutputClassifier",
                "OneVsOneClassifier",
                "OneVsRestClassifier",
                "OutputCodeClassifier",
                "StackingClassifier",
                "TunedThresholdClassifierCV",
                "VotingClassifier",
            ]:
                if "ClassifierChain" == name:
                    clf = Classifier(base_estimator=default_base_estimator)
                elif name in [
                    "MultiOutputClassifier",
                    "OneVsOneClassifier",
                    "OneVsRestClassifier",
                    "OutputCodeClassifier",
                    "FixedThresholdClassifier",
                    "TunedThresholdClassifierCV",
                ]:
                    clf = Classifier(estimator=default_base_estimator)
                elif name == "StackingClassifier":
                    clf = Classifier(estimators=[("lr", LogisticRegression())])
                elif name == "VotingClassifier":
                    clf = Classifier(estimators=[("lr", LogisticRegression())])
            else:
                # Instantiate directly for non-meta classifiers
                clf = Classifier()

            if hasattr(clf, "predict"):  # Check for predict method
                binary_classifiers[name] = Classifier
                globals()[name] = Classifier  # Dynamically add to global namespace
        except Exception as e:
            print(f"Could not import {name}: {e}")

    # Manually add XGBoost classifier
    try:
        binary_classifiers["XGBClassifier"] = XGBClassifier
        globals()["XGBClassifier"] = XGBClassifier
    except Exception as e:
        print(f"Could not import XGBClassifier: {e}")

    # Verify imported classifiers
    print(f"Imported {len(binary_classifiers)} binary classifiers:")
    print(list(binary_classifiers.keys()))
    return binary_classifiers

def visualise_ROI(dataframe_entry, isdataframe=True):
    entry = ak.to_numpy(dataframe_entry) if isdataframe else dataframe_entry

    map0 = [0,11,22,33,44,55,66,77,88]    
    map1 = [1,2,3,4,12,13,14,15,23,24,25,26,34,35,36,37,45,46,47,48,56,57,58,59,67,68,69,70,78,79,80,81,89,90,91,92]
    map2 = [5,6,7,8,16,17,18,19,27,28,29,30,38,39,40,41,49,50,51,52,60,61,62,63,71,72,73,74,82,83,84,85,93,94,95,96]    
    map3 = [9,20,31,42,53,64,75,86,97]    
    mapH = [10,21,32,43,54,65,76,87,98]    

    cell_values = pd.DataFrame(columns=['eta', 'phi', 'r', 'value'])

    for cell_number, value in enumerate(entry):

        if cell_number in map0:
            r=0
            eta = 4* (map0.index(cell_number) %3) + 1.5
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

        cell_values.loc[len(cell_values.index)] = [eta, phi, r, value] 

    fig = px.scatter_3d(cell_values, x="eta", y="phi", z="r", color="value")
    fig.show()

#Prepare
def prepare_data(test_size=0.2, accept_data_filename="l1calo_hist_EGZ_extended.root", reject_data_filename="l1calo_hist_ZMUMU_extended.root",
                 data_subdir="ZMUMU_EGZ_extended_np_pd", format_mode="SuperCell_ET", get_pT=True, distance_boundaries=[0.1,0.2,0.3,0.4], equalised=False):
    log = {"data_prep_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
           "test_size": test_size, 
           "accept_data_filename": accept_data_filename, 
           "reject_data_filename": reject_data_filename, 
           "data_subdir": data_subdir, 
           "format_mode": format_mode,
           "distance_boundaries": distance_boundaries,
           "equalised": equalised
           }
    # if equalised:
    #     data_subdir += "_equalised"
    save_path = os.path.join(os.path.pardir, "data", data_subdir)
    if os.path.exists(os.path.join(save_path, "np_data.npz")) and os.path.exists(os.path.join(save_path, "input_df.parquet")):
        if not os.path.exists(os.path.join(save_path, "log.json")):
            create_log(log, save_path)
            
        print(f"found preprepared data in {save_path}")
        np_data = np.load(os.path.join(save_path, "np_data.npz"))
        input_np, labels_np = np_data["input_np"], np_data["labels_np"]
        input_df = pd.read_parquet(os.path.join(save_path, "input_df.parquet")) 

    else:
        create_log(log, save_path)
        print(f"preprepared data in {save_path} is missing, preparing and saving here")
        accept_data_path = os.path.join(os.path.pardir, "data", accept_data_filename)
        reject_data_path = os.path.join(os.path.pardir, "data", reject_data_filename)
        DFs = import_data_files([accept_data_path, reject_data_path])
        if equalised:
            DFs = equalise(DFs)


        accepted_labels = np.ones(DFs[0].shape[0])
        rejected_labels = np.zeros(DFs[1].shape[0])
        accepted_df = pd.DataFrame({'offline_ele_pt': DFs[0]['offline_ele_pt'],'Label': 1})
        rejected_df = pd.DataFrame({'offline_ele_pt': DFs[1]['offline_ele_pt'],'Label': 0})

        input_np = format_numpy_training_input(DFs, format_mode, distance_boundaries)
        input_df = pd.concat([accepted_df, rejected_df]).reset_index(drop=True)
        labels_np = np.concatenate((accepted_labels, rejected_labels), axis=0)
        
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        np.savez(os.path.join(save_path,"np_data.npz"), input_np=input_np, labels_np=labels_np)
        input_df.to_parquet(os.path.join(save_path, "input_df.parquet"), index=False)

    if get_pT == True:
        X_train, X_test, pd_passthrough_train, pd_passthrough_test, y_train, y_test = train_test_split(input_np, input_df, labels_np, test_size=test_size, random_state=42)
        return X_train, X_test, y_train, y_test, pd_passthrough_train, pd_passthrough_test
    else:
        X_train, X_test, y_train, y_test = train_test_split(input_np, labels_np, test_size=test_size, random_state=42)
        return X_train, X_test, y_train, y_test

def format_numpy_training_input(DFs,format_mode, distance_boundaries):
    if format_mode == "SuperCell_ET":
        print("running sup")
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
        accepted_numpy = generate_topocluster_ET_distribution(DFs[0], distance_boundaries)
        rejected_numpy = generate_topocluster_ET_distribution(DFs[1], distance_boundaries)
        
    elif format_mode == "iso_vars_topocluster_ET_boundaries":
        print("attempting to generate topo training data with iso vars")
        columns = ["eFEX_ReC", "eFEX_ReE", "eFEX_RhE", "eFEX_RhH", "eFEX_WsN", "eFEX_WsD"]
        accepted_numpy = DFs[0][columns].to_numpy(dtype=np.float32)
        rejected_numpy = DFs[1][columns].to_numpy(dtype=np.float32)
        accepted_topocluster_ETs = generate_topocluster_ET_distribution(DFs[0], distance_boundaries)
        rejected_topocluster_ETs = generate_topocluster_ET_distribution(DFs[1], distance_boundaries)
        accepted_numpy = np.concatenate((accepted_numpy, accepted_topocluster_ETs), axis=1)
        rejected_numpy = np.concatenate((rejected_numpy, rejected_topocluster_ETs), axis=1)



    return np.concatenate((accepted_numpy, rejected_numpy), axis=0)

def create_log(log, data_subdir):
    os.makedirs(data_subdir, exist_ok=True)
    
    filepath = os.path.join(data_subdir, "log.json")
    
    with open(filepath, "w") as file:
        json.dump(log, file, indent=4)
    
    print(f"Log saved to {filepath}")
    
def equalise(DFs):
    if DFs[0].shape[0] == DFs[1].shape[0]:
        return DFs

    min_size = min(DFs[0].shape[0], DFs[1].shape[0])
    return [df.sample(n=min_size, random_state=42).reset_index(drop=True) for df in DFs]

def generate_topocluster_ET_distribution(DF, distance_boundaries):
    ET_distributions = np.empty((DF.shape[0], len(distance_boundaries)))
    for i in tqdm(range(DF.shape[0])):
        entry = DF.loc[i]
        TopoCluster_ETs = entry["TopoCluster_ET"]
        TopoCluster_etas = entry["TopoCluster_eta"]
        TopoCluster_phis = entry["TopoCluster_phi"]
        if sum(TopoCluster_ETs) > 0:
            barycentre = calculate_topo_barycentre(TopoCluster_ETs, TopoCluster_etas, TopoCluster_phis)
            topocluster_distances = get_distances_to_barycentre(barycentre, TopoCluster_ETs, TopoCluster_etas, TopoCluster_phis)
            ET_distributions[i] = get_ET_distribution(distance_boundaries, topocluster_distances, TopoCluster_ETs)
        else:
            ET_distributions[i] = [0 for k in range(len(distance_boundaries))]

    return ET_distributions

def calculate_topo_barycentre(TopoCluster_ETs, TopoCluster_etas, TopoCluster_phis):
    barycentre = [0,0]
    ET_total = sum(TopoCluster_ETs)
    barycentre[0] = sum(x * m for x, m in zip(TopoCluster_etas, TopoCluster_ETs)) / ET_total
    barycentre[1] = sum(y * m for y, m in zip(TopoCluster_phis, TopoCluster_ETs)) / ET_total
    return barycentre

def get_distances_to_barycentre(barycentre, TopoCluster_ETs, TopoCluster_etas, TopoCluster_phis):
    topocluster_distances = [0 for i in range(len(TopoCluster_ETs))]
    for i in range(len(TopoCluster_ETs)):
        topocluster_distance = math.sqrt((TopoCluster_etas[i] - barycentre[0])**2 + (TopoCluster_phis[i] - barycentre[1])**2)
        topocluster_distances[i] = topocluster_distance
    return topocluster_distances

def get_ET_distribution(distance_boundaries, topocluster_distances, TopoCluster_ETs):
    lower_boundry = 0
    topocluster_ET_distribution = [0 for i in range(len(distance_boundaries))]
    for i, upper_boundry in enumerate(distance_boundaries):
        for j, topocluster_distance in enumerate(topocluster_distances):
            if topocluster_distance > lower_boundry and topocluster_distance <= upper_boundry:
                topocluster_ET_distribution[i] += TopoCluster_ETs[j]
        lower_boundry = upper_boundry
    return np.array(topocluster_ET_distribution)

def visualise_topocluster_ETs(DF, num_bins):
    all_topocluster_ET_distances = []
    for i in range(DF.shape[0]):
        entry = DF.loc[i]
        TopoCluster_ETs = entry["TopoCluster_ET"]
        TopoCluster_etas = entry["TopoCluster_eta"]
        TopoCluster_phis = entry["TopoCluster_phi"]
        if sum(TopoCluster_ETs) > 0:
            barycentre = calculate_topo_barycentre(TopoCluster_ETs, TopoCluster_etas, TopoCluster_phis)
            topocluster_distances = get_distances_to_barycentre(barycentre, TopoCluster_ETs, TopoCluster_etas, TopoCluster_phis)
            for j, topocluster_distance in enumerate(topocluster_distances):
                topocluster_ET = TopoCluster_ETs[j]
                all_topocluster_ET_distances.append([topocluster_distance, topocluster_ET])
        else:
            pass
        if i % 1000 ==0:
            print(round(i/DF.shape[0]*100,1), "%")

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

def train_with_timeout(clf, X_train, y_train, timeout=3):
    """Runs classifier fitting with a timeout."""
    def target():
        clf.fit(X_train, y_train)
    
    process = multiprocessing.Process(target=target)
    process.start()
    process.join(timeout)
    
    if process.is_alive():
        process.terminate()
        process.join()
        return False
    return True


def train_evaluate_all_classifiers(binary_classifiers, X_train, X_test, y_train, y_test, pd_passthrough_train, pd_passthrough_test, description, data_subdir):
    results = []
    log = load_log(data_subdir)
    log["Description"] = description
    log["Classifiers"] = list(binary_classifiers.keys())
    log["Model_train_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    create_log(log, data_subdir)

    for name, Classifier in tqdm(binary_classifiers.items()):
        # condition = os.path.exists(os.path.join(os.path.join(data_subdir, "efficiency_vs_ele_pt"), f"efficiency_vs_ele_pt_{description}_{f'{name}'}.json"))
        results_found = False
        csv_file = os.path.join(data_subdir, f"{data_subdir}_{description}_all.csv")
        if os.path.exists(csv_file):
            with open(csv_file, mode="r", newline="") as file:
                existing_results = list(csv.DictReader(file))
                if any(row["Classifier"] == name for row in existing_results):
                    results_found = True

        if not results_found:
            result_entry = {"Classifier": name}  # Store classifier results here

            try:
                # Initialize classifier
                if name in [
                    "ClassifierChain",
                    "FixedThresholdClassifier",
                    "MultiOutputClassifier",
                    "OneVsOneClassifier",
                    "OneVsRestClassifier",
                    "OutputCodeClassifier",
                    "StackingClassifier",
                    "TunedThresholdClassifierCV",
                    "VotingClassifier",
                ]:
                    if name == "ClassifierChain":
                        clf = Classifier(base_estimator=LogisticRegression())
                    elif name in [
                        "MultiOutputClassifier",
                        "OneVsOneClassifier",
                        "OneVsRestClassifier",
                        "OutputCodeClassifier",
                        "FixedThresholdClassifier",
                        "TunedThresholdClassifierCV",
                    ]:
                        clf = Classifier(estimator=LogisticRegression())
                    elif name in ["StackingClassifier", "VotingClassifier"]:
                        clf = Classifier(estimators=[("lr", LogisticRegression()), ("rf", RandomForestClassifier())])
                else:
                    params = signature(Classifier).parameters
                    clf = Classifier(random_state=42) if "random_state" in params else Classifier()

                print(f"Training {name}...")
                start = time.time()
                # Train model
                clf.fit(X_train, y_train)
                print("Took:", time.time() - start, "seconds\npredicting...")
                start = time.time()
                # Make predictions
                y_pred = clf.predict(X_test)
                print("Took:", time.time() - start, "seconds")
                pd_passthrough_test["pred"] = y_pred

            except Exception as e:
                print(f"Could not train or predict with {name}: {e}")
                result_entry.update({key: "NULL" for key in ["Accuracy", "TN", "FP", "FN", "TP", "Recall", "Precision", "F1", "MSE"]})
                results.append(result_entry)
                continue  # Skip to the next classifier

            # Compute evaluation metrics, wrapping each in try-except
            try:
                tn, fp, fn, tp, accuracy, recall, precision, f1, mse = evaluate_sklearn_model(y_test, y_pred, classifier_name=f'{name}')
                result_entry.update({"Accuracy": accuracy, "TN": tn, "FP": fp, "FN": fn, "TP": tp, "Recall": recall, "Precision": precision, "F1": f1, "MSE": mse})
            except Exception as e:
                print(f"Failed evaluation metrics for {name}: {e}")
                result_entry.update({"Accuracy": "NULL", "TN": "NULL", "FP": "NULL", "FN": "NULL", "TP": "NULL", "Recall": "NULL", "Precision": "NULL", "F1": "NULL", "MSE": "NULL"})

            try:
                fpr, tpr, roc_auc = compute_roc(clf, X_test, y_test)
                save_roc_data(fpr, tpr, roc_auc, description, f'{name}', data_subdir)
            except Exception as e:
                print(f"Failed ROC computation for {name}: {e}")

            try:
                precision_arr, recall_arr, pr_auc, chance_level = compute_precision_recall(clf, X_test, y_test)
                save_precision_recall_data(precision_arr, recall_arr, pr_auc, description, chance_level, f'{name}', data_subdir)
            except Exception as e:
                print(f"Failed Precision-Recall computation for {name}: {e}")

            try:
                bins, electrons_efficiency = compute_efficiency_vs_ele_PT(pd_passthrough_test, et_Low=20, et_High=60, prediction_parameter="pred")
                save_efficiency_vs_ele_PT(bins, electrons_efficiency, description, f'{name}', data_subdir)
            except Exception as e:
                print(f"Failed Efficiency vs Ele PT computation for {name}: {e}")

            results.append(result_entry)
        else:
            print(f"Results for {name} already exist in {data_subdir}")

    save_csv(description, results, data_subdir)
    return results


def load_log(data_subdir):
    log_file = os.path.join(os.path.pardir, "data", data_subdir, "log.json")
    
    if os.path.exists(log_file):
        print(f"found data preparation log in {log_file}")
    if not os.path.exists(log_file):
        raise FileNotFoundError(f"Log file not found: {log_file}")
    with open(log_file, "r") as file:
        log = json.load(file)
    return log

def save_csv(description, results, data_subdir):
    output_file = f"{data_subdir}/{data_subdir}_{description}_all.csv"
    if os.path.exists(output_file):
        with open(output_file, mode="r", newline="") as file:
            existing_results = list(csv.DictReader(file))
        
        # Convert to dictionary with classifier names as keys for easy updating
        existing_results_dict = {row["Classifier"]: row for row in existing_results}
        
        # Update existing results with new results
        for result in results:
            existing_results_dict[result["Classifier"]] = result
        
        # Convert back to list and sort by classifier name
        updated_results = sorted(existing_results_dict.values(), key=lambda x: x["Classifier"])
    else:
        updated_results = sorted(results, key=lambda x: x["Classifier"])

    with open(output_file, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["Classifier", "Accuracy", "TN", "FP", "FN", "TP", "Precision", "Recall", "F1", "MSE"])
        writer.writeheader()
        writer.writerows(updated_results)

    print(f"Results saved to {output_file}")

#Evaluate
def evaluate_sklearn_model(y_test, y_pred, get_recall=True, get_precision=True, get_f1=True, show_CR=True, show_MSE=True, classifier_name=None):
    recall = None
    precision = None
    f1 = None
    mse = None
    
    if classifier_name != None:
        print("\n\nEvaluation of " + classifier_name)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.8f}")
    
    if get_recall:
        recall = recall_score(y_test, y_pred)
        print(f"Recall: {recall:.8f}")
    if get_precision:
        precision = precision_score(y_test, y_pred)
        print(f"Precision: {precision:.8f}")
    if get_f1:
        f1 = f1_score(y_test, y_pred)
        print(f"F1 Score: {f1:.8f}")
    if show_CR:
        print("Classification Report:\n", classification_report(y_test, y_pred))
    
    if show_MSE:
        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error: {mse:.8f}\n")

    return tn, fp, fn, tp, accuracy, recall, precision, f1, mse

#Plot
def plot_2D_TSNE(embedded_data, colour_var, title):
    plt.figure(figsize=(10, 6))
    plt.scatter(embedded_data[:, 0], embedded_data[:, 1], c=colour_var, cmap='viridis', s=5)
    plt.colorbar(label="1 - accepted, 0 - rejected")
    plt.title(title)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(loc="lower right")
    plt.show()

def plot_3D_TSNE(embedded_data, colour_var,point_size=2):
    fig = px.scatter_3d(x=embedded_data[:, 0], y=embedded_data[:, 1], z=embedded_data[:, 2], color=colour_var)
    fig.update_traces(marker=dict(line=dict(width=0), size=np.ones(colour_var.shape)*point_size))
    fig.show()

def compute_roc(model, X_test, y_test):
    if hasattr(model, "decision_function"):  # For models like SVM or SGD
        y_scores = model.decision_function(X_test)
    elif hasattr(model, "predict_proba"):  # For models like XGBoost or other tree-based models
        probs = model.predict_proba(X_test)
        y_scores = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]

    else:
        raise ValueError("Model must have either a decision_function or predict_proba method.")
    
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc

def plot_roc(fpr, tpr, roc_auc, classifier_name, description):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (AUC = {roc_auc:.8f})")
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic {description} - {classifier_name}')
    plt.legend(loc="lower right")
    plt.show()
    
def compute_precision_recall(model, X_test, y_test):
    if hasattr(model, "decision_function"):  # For models like SVM or SGD
        y_scores = model.decision_function(X_test)
    elif hasattr(model, "predict_proba"):  # For models like XGBoost or other tree-based models
        y_scores = model.predict_proba(X_test)[:, 1]
    else:
        raise ValueError("Model must have either a decision_function or predict_proba method.")
    
    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    pr_auc = auc(recall, precision)  # AUC for PR curve
    chance_level = np.mean(y_test)  # Precompute chance level baseline
    
    return precision, recall, pr_auc, chance_level

def plot_precision_recall(precision, recall, pr_auc, classifier_name, description, chance_level=None, plot_chance_level=False):
    plt.figure()
    plt.plot(recall, precision, color='blue', lw=2, label=f"Precision-Recall curve (AUC = {pr_auc:.8f})")

    # Optionally plot the "chance level" baseline
    if plot_chance_level and chance_level is not None:
        plt.axhline(y=chance_level, color='gray', linestyle='--', label=f"Chance level ({chance_level:.4f})")
    plt.ylim(bottom=0)
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve {description} - {classifier_name}')
    plt.legend(loc="lower right")
    plt.show()
    
def compute_efficiency_vs_ele_PT(X_test_all, et_Low = 20, et_High = 60, prediction_parameter="pred"):
    electrons_all,bins, _ = plt.hist( X_test_all.query("Label == 1")["offline_ele_pt"], bins=40, alpha=0.6, range=[et_Low, et_High])
    electrons_all_tagged, bins,_ = plt.hist( X_test_all.query(f"Label == 1 & {prediction_parameter} == 1")["offline_ele_pt"], bins=40, alpha=0.6, range=[et_Low, et_High])
    electrons_efficiency = np.divide(
        electrons_all_tagged, electrons_all, 
        out=np.zeros_like(electrons_all, dtype=float), 
        where=electrons_all != 0
    )
    plt.clf()
    return bins, electrons_efficiency

def plot_efficiency_vs_ele_PT(bins, electrons_efficiency, classifier_name, description):
    plt.bar(bins[:-1], electrons_efficiency, width = np.diff(bins), align='edge', edgecolor='black')
    plt.title(f'Efficiency against electron PT {description} - {classifier_name}')
    plt.xlabel('Offline electron pt (GeV)')
    plt.ylabel('Efficiency')
    plt.xlim(bins[0], bins[-1])
    plt.show()

def multi_roc(classifiers, X_test, y_test, classifiers_2=None, X_test_2=None, y_test_2=None):
    plt.figure(figsize=(10, 10))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    for classifier in classifiers:
        model = classifiers[classifier]
        fpr, tpr, roc_auc = compute_roc(model,X_test,y_test)
        plt.plot(fpr, tpr, lw=1, label=classifier+f" ROC (AUC = {roc_auc:.8f})")

    if classifiers_2 != None:
        for classifier in classifiers_2:
            model = classifiers_2[classifier]
            fpr, tpr, roc_auc = compute_roc(model,X_test_2,y_test_2)
            plt.plot(fpr, tpr, lw=1, label=classifier+f" ROC (AUC = {roc_auc:.8f})")
            

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig("multi_roc", dpi=400, bbox_inches='tight')
    plt.show()

#Save
def multi_roc_multi_data_single_plot(classifiers, X_tests, y_tests,filename):
    plt.figure(figsize=(15, 15))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    for i, classifier in enumerate(classifiers):
        model = classifiers[classifier]
        fpr, tpr, roc_auc = compute_roc(model,X_tests[i],y_tests[i])
        plt.plot(fpr, tpr, lw=1, label=classifier+f" ROC (AUC = {roc_auc:.8f})")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(filename, dpi=400, bbox_inches='tight')
    plt.show()
    
def save_roc_data(fpr, tpr, roc_auc, description, classifier_name, data_subdir):
    data = {
        "False Positive Rate": fpr.tolist(),
        "True Positive Rate": tpr.tolist(),
        "ROC AUC": roc_auc
    }
    
    directory = os.path.join(data_subdir, "roc")
    os.makedirs(directory, exist_ok=True)
    
    filename = os.path.join(directory, f"roc_{description}_{classifier_name}.json")
    with open(filename, "w") as file:
        json.dump(data, file, indent=4)
    
    print(f"ROC data saved to {filename}")

def save_precision_recall_data(precision_arr, recall_arr, pr_auc, description, chance_level, classifier_name, data_subdir):
    data = {
        "Precision": precision_arr.tolist(),
        "Recall": recall_arr.tolist(),
        "Precision-Recall AUC": pr_auc,
        "Chance Level": chance_level
    }

    directory = os.path.join(data_subdir, "precision_recall")
    os.makedirs(directory, exist_ok=True)

    filename = os.path.join(directory, f"precision_recall_{description}_{classifier_name}.json")
    with open(filename, "w") as file:
        json.dump(data, file, indent=4)

    print(f"Precision-Recall data saved to {filename}")

def save_efficiency_vs_ele_PT(bins, electrons_efficiency, description, classifier_name, data_subdir):
    data = {
        "Bins": bins.tolist(),
        "Efficiency": electrons_efficiency.tolist()
    }

    directory = os.path.join(data_subdir, "efficiency_vs_ele_pt")
    os.makedirs(directory, exist_ok=True)

    filename = os.path.join(directory, f"efficiency_vs_ele_pt_{description}_{classifier_name}.json")
    with open(filename, "w") as file:
        json.dump(data, file, indent=4)

    print(f"Efficiency vs Electron PT data saved to {filename}")
    
#Read
def read_roc(classifier_name, description, data_subdir):
    filename = os.path.join(data_subdir, "roc", f"roc_{description}_{classifier_name}.json")
    try:
        with open(filename, "r") as file:
            data = json.load(file)
        return data["False Positive Rate"], data["True Positive Rate"], data["ROC AUC"]
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return None, None, None

def read_precision_recall(classifier_name, description, data_subdir):
    filename = os.path.join(data_subdir, "precision_recall", f"precision_recall_{description}_{classifier_name}.json")
    try:
        with open(filename, "r") as file:
            data = json.load(file)
        return data["Precision"], data["Recall"], data["Precision-Recall AUC"], data["Chance Level"]
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return None, None, None, None

def read_efficiency_vs_ele_PT(classifier_name, description, data_subdir):
    filename = os.path.join(data_subdir, "efficiency_vs_ele_PT", f"efficiency_vs_ele_pt_{description}_{classifier_name}.json")
    try:
        with open(filename, "r") as file:
            data = json.load(file)
        return data["Bins"], data["Efficiency"]
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return None, None
    
def plot_all_results(binary_classifier_names, description, data_subdir):
    for classifier_name in binary_classifier_names:
        fpr, tpr, roc_auc = read_roc(classifier_name, description, data_subdir)
        precision, recall, pr_auc, chance_level = read_precision_recall(classifier_name, description, data_subdir)
        bins, electrons_efficiency = read_efficiency_vs_ele_PT(classifier_name, description, data_subdir)
        
        if fpr is not None and tpr is not None and roc_auc is not None:
            plot_roc(fpr, tpr, roc_auc, classifier_name, description)
        
        if precision is not None and recall is not None and pr_auc is not None:
            plot_precision_recall(precision, recall, pr_auc, classifier_name, description, chance_level=chance_level, plot_chance_level=True)

        if bins is not None and electrons_efficiency is not None:
            plot_efficiency_vs_ele_PT(bins, electrons_efficiency, classifier_name, description)
        