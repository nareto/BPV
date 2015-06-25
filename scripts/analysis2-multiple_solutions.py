import BPV
import pattern_manipulation as pm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

n = 10
N=5
W=0.025
multiplier = 4

data = BPV.Data()
data.read_csv("p.delviva.csv",False)
data.df.sort_index(by="p",inplace=True,ascending=False)
data.df.set_index(pd.Index([j for j in range(len(data.df))]), inplace=True)
dftmp = data.data_head(n).df

p = np.zeros(multiplier*n)
for i in range(multiplier*n):
    p[i] = dftmp['p'][i%n]

p /= p.sum()
df = BPV.Data(pd.DataFrame(p, columns=['p']))
df.df['plog1onp'] = df.df['p']*np.log(1/df.df['p'])
df.df.sort_index(by='p',inplace=True,ascending=False)
df.df.set_index(pd.Index([j for j in range(len(df.df))]), inplace=True)

prbl_decH = BPV.BPV("decgraphH",df,N,W,time_solver=False)
prbl_decH.solve()
#cProfile.run('prbl.solve()',sort=1)
prbl_decH.pprint_solution()

#prbl = BPV.BPV("glpk",df,N,W,time_solver=False)
#prbl.solve()
#prbl.pprint_solution()

prbl = BPV.BPV("pulp",df,N,W,time_solver=False)
prbl.solve()
prbl.pprint_solution()

#print(df.df)
#print("\n\n", prbl_decH.multiple_solutions, "\n\n")
