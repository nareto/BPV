import BPV
import pattern_manipulation as pm
import numpy as np
import pandas as pd

N,W = 10, 0.1504

data = BPV.Data()
data.read_csv("pixel.dist.csv",False)
data.df.sort_index(by="p",inplace=True,ascending=True)
data.df.set_index(pd.Index([j for j in range(len(data.df))]), inplace=True)

prbl_pulp = BPV.BPV("pulp",data,N,W,time_solver=False)
prbl_pulp.solve()
prbl_pulp.pprint_solution()


prbl_pulp = BPV.BPV("heuristic",data,N,W,time_solver=False)
prbl_pulp.solve()
prbl_pulp.pprint_solution()
