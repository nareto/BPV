import BPV
import pattern_manipulation as pm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cProfile

n = 50
N=5
W=0.01  #rate

data = BPV.Data()
data.read_csv("p.delviva.csv",False)
data.df.sort_index(by="p",inplace=True,ascending=False)
data.df.set_index(pd.Index([j for j in range(len(data.df))]), inplace=True)
data_head = data.data_head(n)
#data.df.sort_index(by="p")

prbl_pulp = BPV.BPV("pulp",data_head,N,W,time_solver=False)
prbl_pulp.solve()
prbl_pulp.pprint_solution()

prbl_decV = BPV.BPV("decgraphV",data_head,N,W,time_solver=False)
prbl_decV.solve()
#cProfile.run('prbl_decV.solve()',sort=1)
prbl_decV.pprint_solution()

prbl_decW = BPV.BPV("decgraphW",data_head,N,W,time_solver=False)
prbl_decW.solve()
#cProfile.run('prbl_decW.solve()',sort=1)
prbl_decW.pprint_solution()

