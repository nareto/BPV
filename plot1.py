import matplotlib
matplotlib.use('TkAgg')
import BPV
import pattern_manipulation as pm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import ipdb

pdict = {}
files ={(0,"img_dig/cvcl/street"): "p.cvcl-street.csv",\
        (1,"img_dig/cvcl/tallbuilding"): "p.cvcl-tallbuilding.csv",\
        (2,"img_dig/cvcl/highway"): "p.cvcl-highway.csv",\
        (3,"img_dig/cvcl/inside_city"): "p.cvcl-inside_city.csv",\
        (4,"img_dig/cvcl/coast"): "p.cvcl-coast.csv",\
        (5,"img_dig/cvcl/forest"): "p.cvcl-forest.csv",\
        (6,"img_dig/cvcl/mountain"): "p.cvcl-mountain.csv",\
        (7,"img_dig/cvcl/Opencountry"): "p.cvcl-opencountry.csv"}


#files = {(0,"tmp_dig/"):"tmp.csv"}
for imgdb,csv in sorted(files.items()):
    print(imgdb)
    if os.path.isfile(csv) == False:
        #ipdb.set_trace()
        dist = pm.distribution(imgdb[1],(3,3))
        dist.to_csv(csv)
    tag = imgdb[1].lstrip("img_dig/")
    pdict[(imgdb[0],tag)] = pd.read_csv(csv,header=None)

#pdict[(8,"MacGill Calibrated")] = BPV.read_distribution_csv("p.macgill.csv")

tot_patterns = 512
N=50
W=0.05  #rate

n_plots = len(pdict.keys())

fig = plt.plot()
f,axarr = plt.subplots(n_plots,1,sharex=True)
i = 0
for imgdb, p in sorted(pdict.items()):
    n = len(p)
    if n < tot_patterns:
        print("WARNING: probability vector's {2} length is {0}<{1}".format(n,tot_patterns,imgdb[1]))
    p_array = np.array(p.ix[:,1])
    exact_solver = BPV.BPV("exact",n,N,W,p_array)
    exact_solver.solve()
    sol=exact_solver.solution_indexes()
    axarr[i].set_title(imgdb[1])
    axarr[i].plot(p_array)
    for point in sol:
        axarr[i].plot(sol,[0]*len(sol),'or')
    i += 1
plt.show()
