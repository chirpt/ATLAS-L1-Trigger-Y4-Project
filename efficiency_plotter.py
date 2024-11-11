import os
import math
import pandas as pd
import numpy as np
import uproot
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def main():
    file = "l1calo_hist_EGZ_extended.root"
    DF = import_data(file)
    efficiency_data, PT_bins = extract_efficiency(DF)
    plot(efficiency_data, PT_bins)


def import_data(file):
    File = uproot.open(file)
    Tree = File["tree_DMC"]
    DF = Tree.arrays(library="pd")
    return DF


def extract_efficiency(DF):

    print(DF.info())

    binning_ratio = 1
    momentum_range = DF["offline_ele_pt"].max()
    energy_cutoff = 25
    number_of_bins = int(momentum_range/binning_ratio) + 1

    PT_bins = [i * binning_ratio for i in range(number_of_bins)]
    efficiency_data = [0 for i in range(number_of_bins)]
    TOB_ET_counts = [0 for i in range(number_of_bins)]
    offline_PT_counts = [0 for i in range(number_of_bins)]

    print("MOMENTUM RANGE: ", momentum_range)
    print("NUMBER OF BINS: ", number_of_bins)

    try:
        for i in range(DF.shape[0]):
            entry = DF.loc[i]
            PT = entry["offline_ele_pt"]
            
            if abs(entry["offline_ele_eta"]) < 1.375 or abs(entry["offline_ele_eta"]) > 1.52:
                offline_PT_counts[int(PT/binning_ratio)] += 1
                if entry["TOB_ET"] >= energy_cutoff:
                    TOB_ET_counts[int(PT/binning_ratio)] +=1
            
            if i % 2000 == 0:
                print(round(i/DF.shape[0]*100,2),"%")

    except Exception as e:
        print(e)
        print("ENTRY:",entry)
        print("BIN INDEX: ", int(PT/binning_ratio))
    

    
    for i in range(number_of_bins):
        if offline_PT_counts[i] != 0:
            efficiency_data[i] = TOB_ET_counts[i] / offline_PT_counts[i]
            
    return efficiency_data, PT_bins


def plot(efficiency_data,PT_bins):
    plt.plot(PT_bins,efficiency_data)
    plt.show()


main()