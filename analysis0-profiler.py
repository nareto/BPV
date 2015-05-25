import BPV
import pattern_manipulation as pm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cProfile



data = BPV.Data()
data.read_csv("p.delviva.csv",False)
data.df.sort_index(by="p",inplace=True,ascending=False)
data.df.set_index(pd.Index([j for j in range(len(data.df))]), inplace=True)
data_head = data.data_head(20)
#data.df.sort_index(by="p")
N=5
W=0.025  #rate

prbl = BPV.BPV("decgraphV",data_head,N,W,time_solver=False)
prbl.solve()
#cProfile.run('prbl.solve()',sort=1)
prbl.pprint_solution()

prbl = BPV.BPV("pulp",data_head,N,W,time_solver=False)
prbl.solve()
prbl.pprint_solution()

#print(data_head.df)
#
#prbl = BPV.BPV("euristic",data_head,N,W,time_solver=False)
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
