import os
import math
import pandas as pd
import numpy as np
import uproot
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from utilities import *

# What ML algorithm is best suited to the problem?

def main():
    DFs = import_data_files(["l1calo_hist_ZMUMU_extended.root","l1calo_hist_EGZ_extended.root"])
    DFs = generate_clusters(DFs)
    training_data, testing_data = prepare_data(DFs)
    model_data = train_model(training_data)
    save_model(model_data)
    test_model(model_data, testing_data)

def prepare_data(DFs):
    DFs = extract_features(DFs)   
    DFs = split_data(DFs)
    return DFs

def extract_features(DFs):
    output_DFs = []

    for DF in DFs:
        output_DF = pd.DataFrame(columns=['PS_Clus', 'EM1_Clus', 'EM2_Clus','EM3_Clus'])
        output_DF["PS_Clus"]=DF["PS_Clus"]
        output_DF["EM1_Clus"]=DF["EM1_Clus"]
        output_DF["EM2_Clus"]=DF["EM2_Clus"]
        output_DF["EM3_Clus"]=DF["EM3_Clus"]

        # output_DF = output_DF.assign(PS_Clus=DF["PS_Clus"])
        # output_DF = output_DF.assign(EM1_Clus=DF["EM1_Clus"])
        # output_DF = output_DF.assign(EM2_Clus=DF["EM2_Clus"])
        # output_DF = output_DF.assign(EM3_Clus=DF["EM3_Clus"])

        output_DFs.append(output_DF)

    return output_DFs

def split_data(DFs):
    training_data = []
    testing_data = []
    
    for DF in DFs:
        split_index = int(len(DF)*(2/3))
        training_data.append(DF.iloc[:split_index])
        testing_data.append(DF.iloc[split_index:])

    return training_data, testing_data


def train_model(training_data):
    training_data_to_accept = training_data[0]
    training_data_to_reject = training_data[1]

    input("clusters generated, press enter to begin ML training...")
    model_data = None
    return model_data
    
def save_model(model_data):
    pass
    input("model training completed, model saved, press enter to begin model testing...")

def test_model(model_data, testing_data):
    pass

main()