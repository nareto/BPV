import BPV
import pattern_manipulation as pm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ipdb
import pickle

n = 20
N=5
W=0.1
epsilon = 0.9

df = BPV.Data()
df.read_csv("p.delviva.csv",False)
df.df.sort_index(by="p",inplace=True,ascending=False)
df.df.set_index(pd.Index([j for j in range(len(df.df))]), inplace=True)
df = df.data_head(n)


prbl_pulp = BPV.BPV("pulp",df,N,W,time_solver=False)
prbl_pulp.solve()
prbl_pulp.pprint_solution()

prbl_decV = BPV.BPV("decgraphV",df,N,W,time_solver=False,use_quantized_entropy=True)
prbl_decV.solve()
#cProfile.run('prbl.solve()',sort=1)
prbl_decV.pprint_solution()

pickle.dump(prbl_decV, open("test0-prbl_decV", "wb"))
pickle.dump(df,open("test0-df","wb"))
