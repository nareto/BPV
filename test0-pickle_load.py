import BPV
import pattern_manipulation as pm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ipdb
import pickle

prbl_decV = pickle.load(open("test0-prbl_decV",'rb'))
data = pickle.load(open("test0-df","rb"))

print(prbl_decV.decgraph_best_value_node,"\n\n", data.df)
