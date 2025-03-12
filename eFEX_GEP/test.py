import numpy as np
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(parent_dir))
from utilities_new import *

name = "MLPClassifier"
data_subdir = "eFEX_GEP_test"
accept_data_filename = "l1calo_topocluster_test_Zee.root"
reject_data_filename = "ZMUMU_TopoCluster_Supercell_Large.root"
distance_boundaries = [0.00625,0.0125,0.025,0.05,0.1,0.2,0.4]


save_path = os.path.join("data", data_subdir)
# if os.path.exists(os.path.join(save_path, "np_data.npz")) and os.path.exists(os.path.join(save_path, "input_df.parquet")):
if os.path.exists(os.path.join(save_path, "np_data.npz")):
    print(f"found preprepared data in {save_path}")
    np_data = np.load(os.path.join(save_path, "np_data.npz"))
    input_np1, input_np2, labels_np = np_data["input_np1"], np_data["input_np2"], np_data["labels_np"]
    # input_df = pd.read_parquet(os.path.join(save_path, "input_df.parquet")) 

else:
    print(f"preprepared data in {save_path} is missing, preparing and saving here")
    accept_data_path = os.path.join(parent_dir, "data", accept_data_filename)
    reject_data_path = os.path.join(parent_dir, "data", reject_data_filename)
    DFs = import_data_files([accept_data_path, reject_data_path])
    accepted_numpy_eFEX = ak.to_numpy(DFs[0]['SuperCell_ET'])
    rejected_numpy_eFEX = ak.to_numpy(DFs[1]['SuperCell_ET'])
    print("attempting to generate topo training data")
    accepted_numpy_GEP = generate_topocluster_ET_distribution(DFs[0], distance_boundaries)
    rejected_numpy_GEP = generate_topocluster_ET_distribution(DFs[1], distance_boundaries)
    accepted_labels = np.ones(DFs[0].shape[0])
    rejected_labels = np.zeros(DFs[1].shape[0])
    # accepted_df = pd.DataFrame({'offline_ele_pt': DFs[0]['offline_ele_pt'], 'EventNumber': DFs[0]['EventNumber'], 'Label': 1})
    # rejected_df = pd.DataFrame({'offline_ele_pt': DFs[1]['offline_ele_pt'], 'EventNumber': DFs[0]['EventNumber'], 'Label': 0})
    
    input_np1 = np.concatenate((accepted_numpy_eFEX, rejected_numpy_eFEX), axis=0)
    input_np2 = np.concatenate((accepted_numpy_GEP, rejected_numpy_GEP), axis=0)
    labels_np = np.concatenate((accepted_labels, rejected_labels), axis=0)
    # input_df = pd.concat([accepted_df, rejected_df]).reset_index(drop=True)
    
    if not os.path.exists(save_path):
            os.mkdir(save_path)
    np.savez(os.path.join(save_path,"np_data.npz"), input_np1=input_np1, input_np2=input_np2, labels_np=labels_np)
    # input_df.to_parquet(os.path.join(save_path, "input_df.parquet"), index=False)
    

# Step 1: First train-test split (80% train, 20% test)
X_train, X_test, X_train_topo, X_test_topo, y_train, y_test = train_test_split(input_np1, input_np2, labels_np, test_size=0.2, random_state=42)

# Step 2: Train first classifier
print("Training first classifier")
Classifier = import_all_classifiers()[name]
clf = Classifier()
clf.fit(X_train, y_train)

# Step 3: Get predictions on test data


if hasattr(clf, "decision_function"):  # For models like SVM or SGD
    y_scores = clf.decision_function(X_test)
elif hasattr(clf, "predict_proba"):  # For models like XGBoost or other tree-based models
    probs = clf.predict_proba(X_test)
    y_scores = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]

# Additional features for the second classifier (e.g., original test features + first model predictions)
X_test_extended = np.hstack((X_test_topo, y_scores.reshape(-1, 1))) #not sure about reshaping

# Step 4: Second train-test split (80% of test data for training, 20% for testing)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_test_extended, y_test, test_size=0.2, random_state=42)

# Step 5: Train second classifier (can be a different model)
print("Training second classifier")
Classifier2 = import_all_classifiers()[name]
clf2 = Classifier()
clf2.fit(X_train2, y_train2)

# Step 6: Evaluate second classifier
y_pred1 = clf.predict(X_test2)
y_pred2 = clf2.predict(X_test2)
print("before second training with topo:")
tn, fp, fn, tp, accuracy, recall, precision, f1, mse = evaluate_sklearn_model(y_test, y_pred1, classifier_name=f'{name}')
print("after second training with topo:")
tn, fp, fn, tp, accuracy, recall, precision, f1, mse = evaluate_sklearn_model(y_test, y_pred2, classifier_name=f'{name}')
