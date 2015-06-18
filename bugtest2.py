import sys
import BPV
import pattern_manipulation as pm
import numpy as np
import pandas as pd

df = BPV.Data()
df.read_csv("p.delviva.csv",False)
df.df.sort_index(by="p",inplace=True,ascending=True)
df.df.set_index(pd.Index([j for j in range(len(df.df))]), inplace=True)
df = df.data_head(100)

N=15
W=0.0019

prbl_pulp = BPV.BPV("pulp",df,N,W,time_solver=False)
prbl_pulp.solve()
prbl_pulp.pprint_solution()


prbl_decg = BPV.BPV("decgraphH",df,N,W,time_solver=False)
prbl_decg.solve()
prbl_decg.pprint_solution()

prbl_decg = BPV.BPV("decgraphW",df,N,W,time_solver=False)
prbl_decg.solve()
prbl_decg.pprint_solution()
