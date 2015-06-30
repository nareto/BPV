import BPV
import pattern_manipulation as pm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ipdb

n = 10
N=5
W=0.025
epsilon = 1000

df = BPV.Data()
df.read_csv("pixel.dist.csv",False)
df.df.sort_index(by="p",inplace=True,ascending=False)
df.df.set_index(pd.Index([j for j in range(len(df.df))]), inplace=True)
df = df.data_head(n)

#p = np.zeros(2*n)
#for i in range(2*n):
#    p[i] = df.df['p'][i%n]
#p /= p.sum()
#df = BPV.Data(pd.DataFrame(p, columns=['p']))
#df.df['plog1onp'] = df.df['p']*np.log(1/df.df['p'])
#df.df.sort_index(by='p',inplace=True,ascending=False)
#df.df.set_index(pd.Index([j for j in range(len(df.df))]), inplace=True)

min_entropy = df.df['plog1onp'].min()
c = 2*N/(epsilon*min_entropy)
print("\n c = ",c,"\n\n")
scaler = lambda x: (1/c)*(int(c*x) + 1)
df.df['quantized_plog1onp'] = df.df['plog1onp'].apply(scaler)

prbl_pulp = BPV.BPV("pulp",df,N,W,time_solver=False)
prbl_pulp.solve()
prbl_pulp.pprint_solution()

#prbl_decW = BPV.BPV("decgraphH",df,N,W,time_solver=False)#,use_quantized_entropy=True)
#prbl_decW.solve()
##cProfile.run('prbl.solve()',sort=1)
#prbl_decW.pprint_solution()

prbl_decW = BPV.BPV("decgraphW",df,N,W,time_solver=False,use_quantized_entropy=True)
prbl_decW.solve()
#cProfile.run('prbl.solve()',sort=1)
prbl_decW.pprint_solution()
##print("\n\n Relative Erorr = ",BPV.relative_error(prbl_decW,prbl_pulp))
print("\n\n\n", prbl_decW.multiple_solutions)
#
#sol_view = BPV.solution_non_zero_view(prbl_pulp, prbl_decW)
#print(sol_view)
print(df.df)
#pulp_indexes = df.df[df.df['pulp'] == True].index
#decgraphW_indexes = df.df[df.df['decgraphW'] == True].index
#symdiffidx = pulp_indexes.sym_diff(decgraphW_indexes)
#symdiff = df.df.ix[symdiffidx]
#print("\n",symdiff)
