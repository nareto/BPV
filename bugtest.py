import sys
sys.path.append('/home/renato/.conda/envs/clone_tesi/lib/python3.4/site-packages/')
import matplotlib
#%matplotlib inline
matplotlib.use('TkAgg')
import BPV
import pattern_manipulation as pm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = BPV.Data()
df.read_csv("p.delviva.csv",False)
df.df.sort_index(by="p",inplace=True,ascending=True)
df.df.set_index(pd.Index([j for j in range(len(df.df))]), inplace=True)
df = df.data_head(100)

N=50
W=0.05

prbl_pulp = BPV.BPV("pulp",df,N,W,time_solver=False)
prbl_pulp.solve()
prbl_pulp.pprint_solution()


prbl_decg = BPV.BPV("decgraphH",df,N,W,time_solver=False)
prbl_decg.solve()
prbl_decg.pprint_solution()
