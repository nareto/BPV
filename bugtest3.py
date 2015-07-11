import matplotlib
import BPV
import pattern_manipulation as pm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gc


n = 70

df = BPV.Data()
df.read_csv("data/pixel.dist.csv",False)
df.df.sort_index(by="p",inplace=True,ascending=True)
df.df.set_index(pd.Index([j for j in range(len(df.df))]), inplace=True)
dfh1 = df.data_head(n)
dfh1.df.set_index(pd.Index([j for j in range(len(dfh1.df))]), inplace=True)


W=0.001
N = 3

prbl_decH = BPV.BPV("decgraphH",dfh1,N,W,time_solver=True)
prbl_decH.solve()
prbl_decH.pprint_solution()
