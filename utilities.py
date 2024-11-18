import os
import math
import pandas as pd
import numpy as np
import uproot
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import plotly.express as px
import awkward as ak



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

def VariableMaker(inputs):

    map0 = [0,11,22,3,44,55,66,77,88]    
    map1 = [1,2,3,4,12,13,14,15,23,24,25,26,34,35,36,37,45,46,47,48,56,57,58,59,67,68,69,70,78,79,80,81,89,90,91,92]
    map2 = [5,6,7,8,16,17,18,19,27,28,29,30,38,39,40,41,49,50,51,52,60,61,62,63,71,72,73,74,82,83,84,85,93,94,95,96]    
    map3 = [9,20,31,42,53,64,75,86,97]    
    mapH = [10,21,32,43,54,65,76,87,98]    
    
    seedCell = 0
    seeds = np.array([])
    seedETs = np.array([])

    for sc in range(5,9):
        cell = 44 + sc
        scM = sc - 1
        cellM = 44+scM if scM >= 5 else 41 # cell 8 of tower 3 (33+8)
        scP = sc + 1
        cellP = 44+scP if scP <= 8 else 60 # cell 5 of tower 5 (55+5)
        if inputs[cell] >= inputs[cellM] and inputs[cell] > inputs[cellP]:
            seeds = np.append(seeds,sc)
            seedETs = np.append(seedETs,inputs[cell])

    if seeds.size == 0 or seeds.size > 2:
        print("Error: invalid number of seeds found!")
        return variables
    elif seeds.size == 1:
        seedCell = int(seeds[0])
    else:
        if seedETs[0] > seedETs[1]:
            seedCell = int(seeds[0])
        else:
            seedCell = int(seeds[1])
            
    # Now apply UnD logic
    up =   int(77 + seedCell)
    down = int(11 + seedCell)
    UnD = 1 if inputs[up] >= inputs[down] else 0

    # Presampler & layer 3: cluster = central tower supercell plus tower above/below in phi
    clus0 = inputs[44] + (inputs[77] if UnD > 0 else inputs[11])
    clus3 = inputs[53] + (inputs[86] if UnD > 0 else inputs[20])

    clus1 = 0
    clus2 = 0
    for off in range(-1,2):
        cell = seedCell + off # cell in layer 2
        towerC = 44 # Start of central tower (tower 4 of 8)
        # 
        if cell < 5: # to the left of central tower
            towerC = 33 # previous tower
            cell   = 8  # right-hand cell in that tower
        elif cell > 8: # to the right of central tower
            towerC = 55 # next tower
            cell   = 5  # left-hand cell of next tower
        # Neighbouring tower in phi (above or below)
        towerN = towerC+33 if UnD > 0 else towerC - 33
        clus1 += inputs[towerC+cell-4] + inputs[towerN+cell-4] # cells from layer 1
        clus2 += inputs[towerC+cell]   + inputs[towerN+cell]   # cells from layer 2 
    
    return clus0, clus1, clus2, clus3

def generate_clusters(DFs):
    print("generating clusters...")
    i = 0
    for DF in DFs:
        i += 1
        scVec = DF['SuperCell_ET']

        cl0=np.zeros(scVec.size)
        cl1=np.zeros(scVec.size)
        cl2=np.zeros(scVec.size)
        cl3=np.zeros(scVec.size)

        for roi in range(scVec.size):
            if roi % 1000 == 0:
                print(round(100*roi/scVec.size,1),"% ,","("+str(i)+"/"+str(len(DFs))+")")
            inputs = scVec[roi]
            cl0[roi],cl1[roi],cl2[roi],cl3[roi] = VariableMaker(inputs)

        DF = DF.assign(PS_Clus=cl0)
        DF = DF.assign(EM1_Clus=cl1)
        DF = DF.assign(EM2_Clus=cl2)
        DF = DF.assign(EM3_Clus=cl3)

    return DFs

def visualise_ROI(dataframe_entry):
    entry = ak.to_numpy(dataframe_entry)

    map0 = [0,11,22,33,44,55,66,77,88]    
    map1 = [1,2,3,4,12,13,14,15,23,24,25,26,34,35,36,37,45,46,47,48,56,57,58,59,67,68,69,70,78,79,80,81,89,90,91,92]
    map2 = [5,6,7,8,16,17,18,19,27,28,29,30,38,39,40,41,49,50,51,52,60,61,62,63,71,72,73,74,82,83,84,85,93,94,95,96]    
    map3 = [9,20,31,42,53,64,75,86,97]    
    mapH = [10,21,32,43,54,65,76,87,98]    

    cell_values = pd.DataFrame(columns=['eta', 'phi', 'r','value'])


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

    fig = px.scatter_3d(cell_values,x="eta",y="phi",z="r",color="value")
    fig.show()
