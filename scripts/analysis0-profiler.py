import BPV
import pattern_manipulation as pm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cProfile

n = 100
N=20
W=0.03  #rate

data = BPV.Data()
data.read_csv("pixel.dist.csv",False)
data.df.sort_index(by="p",inplace=True,ascending=True)
data.df.set_index(pd.Index([j for j in range(len(data.df))]), inplace=True)
data_head = data.data_head(n)
#data.df.sort_index(by="p")

#prbl_pulp = BPV.BPV("pulp",data_head,N,W,time_solver=False)
#prbl_pulp.solve()
#prbl_pulp.pprint_solution()
#

prbl_decH = BPV.BPV("decgraphH",data_head,N,W,time_solver=True)
prbl_decH.solve()
#cProfile.run('prbl_decH.solve()',sort=1)
prbl_decH.pprint_solution()

#prbl_decW = BPV.BPV("decgraphW",data_head,N,W,time_solver=False)
#prbl_decW.solve()
##cProfile.run('prbl_decW.solve()',sort=1)
#prbl_decW.pprint_solution()
