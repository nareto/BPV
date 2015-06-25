#import sys
#sys.path.append('/usr/lib/python3.4/site-packages')
import matplotlib
#%matplotlib inline
matplotlib.use('TkAgg')
import BPV
import pattern_manipulation as pm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


df = BPV.Data()
df.read_csv("p.delviva.csv",False)
df.df.sort_index(by="p",inplace=True,ascending=True)
df.df.set_index(pd.Index([j for j in range(len(df.df))]), inplace=True)


N = 50
W = 0.025
variances = [0.00001,0.00006,0.0001,0.001]
l = len(variances)
in_errs = [0]*l
out_errs = [0]*l
for i in range(l):
    df_err = df.artificial_noise1(variances[i])

    pulp_correct = BPV.BPV("pulp",df,N,W,time_solver=False)
    pulp_correct.solve()
    pulp_correct.pprint_solution()

    pulp_error = BPV.BPV("pulp",df_err,N,W,time_solver=False)
    pulp_error.solve()
    pulp_error.pprint_solution()

    in_errs[i] = BPV.distance1(df.df['p'],df_err.df['p'])
    out_errs[i] = np.abs(pulp_correct.solution_entropy - pulp_error.solution_entropy)
    print(in_errs[i], out_errs[i])
    
    print("------------------")

    
