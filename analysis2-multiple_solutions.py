import BPV
import pattern_manipulation as pm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

n = 10
N=5
W=0.025

data = BPV.Data()
data.read_csv("p.delviva.csv",False)
data.df.sort_index(by="p",inplace=True,ascending=False)
data.df.set_index(pd.Index([j for j in range(len(data.df))]), inplace=True)
dftmp = data.data_head(n).df

p = np.zeros(2*n)
for i in range(2*n):
    p[i] = dftmp['p'][i%n]

p /= p.sum()
df = BPV.Data(pd.DataFrame(p, columns=['p']))
df.df['plog1onp'] = df.df['p']*np.log(1/df.df['p'])
df.df.sort_index(by='p',inplace=True,ascending=False)
df.df.set_index(pd.Index([j for j in range(len(df.df))]), inplace=True)

prbl = BPV.BPV("decgraphV",df,N,W,time_solver=False)
prbl.solve()
#cProfile.run('prbl.solve()',sort=1)
prbl.pprint_solution()

prbl = BPV.BPV("pulp",df,N,W,time_solver=False)
prbl.solve()
prbl.pprint_solution()

print(df.df)
#print(df.df)
#
#prbl = BPV.BPV("euristic",df,N,W,time_solver=False)
#prbl.solve()
#prbl.pprint_solution()
#
#plt.figure()
#lvisitlist = prbl.decgraph_len_visitlist
#t = np.arange(0,len(lvisitlist),1)
##plt.plot(t,lvisitlist,'b',t,2**t,'r')
#plt.plot(t,lvisitlist,'b')
#plt.ylim(0,max(lvisitlist))
#plt.show()
