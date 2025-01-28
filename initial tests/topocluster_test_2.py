from utilities import *

Files= ["l1calo_topocluster_test_ZB.root","l1calo_topocluster_test_Zee.root"] # "extended" includes isolation vars and extended_new trigger decision
DFs = []



for i in range (0,len(Files)):
  File = uproot.open(os.path.join("C:\\Users\\Me\Desktop\\uni temp\\Y5\\Project\\ATLAS2\\ATLAS-L1-Trigger-Y4-Project\\data", Files[i]))
  Tree = File["tree_DMC"]
  DFs.append(Tree.arrays(library="pd"))

visualise_topocluster_ETs(DFs[0],100)
visualise_topocluster_ETs(DFs[1],1000)