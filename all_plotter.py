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

"""
Dataframe columns
['TOB_ET', 'TOB_eta', 'TOB_ieta', 'TOB_ietabin', 'TOB_phi', 'offline_ele_pt', 'offline_ele_eta', 'offline_ele_eta_cal', 'offline_ele_phi', 'offline_ele_phi_cal', 'eFEX_ET', 'eFEX_PS_ET', 'eFEX_L1_ET', 'eFEX_L2_ET', 'eFEX_L3_ET', 'SuperCell_ET', 'eFEX_ReC', 'eFEX_ReE', 'eFEX_RhE', 'eFEX_RhH', 'eFEX_WsN', 'eFEX_WsD']"""

def main():
    Files= ["l1calo_hist_ZMUMU_extended.root","l1calo_hist_EGZ_extended.root","l1calo_hist_EGZ_extended_new.root"] # "extended" includes isolation vars and extended_new trigger decision
    for filestring in Files:
        DF = import_data(filestring)
        # print(list(DF))
        plot(DF,filestring)

def plot(DF,filestring):
    # DF["TOB_ET"].plot(kind="hist",bins=600)
    # plt.show()

    
    (DF["eFEX_ReE"]/DF["eFEX_ReC"]).plot(kind="hist",bins=300, range=[0,1])
    plt.title(filestring)
    plt.show()

main()