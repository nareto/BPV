import matplotlib
matplotlib.use('TkAgg')
import BPV
import pattern_manipulation as pm
import numpy as np
import matplotlib.pyplot as plt
import os

pdict = {}
files ={(0,"img_dig/cvcl/street"): "p.cvcl-street.csv",\
        (1,"img_dig/cvcl/tallbuilding"): "p.cvcl-tallbuilding.csv",\
        (2,"img_dig/cvcl/highway"): "p.cvcl-highway.csv",\
        (3,"img_dig/cvcl/inside_city"): "p.cvcl-inside_city.csv",\
        (4,"img_dig/cvcl/coast"): "p.cvcl-coast.csv",\
        (5,"img_dig/cvcl/forest"): "p.cvcl-forest.csv",\
        (6,"img_dig/cvcl/mountain"): "p.cvcl-mountain.csv",\
        (7,"img_dig/cvcl/Opencountry"): "p.cvcl-opencountry.csv"}

for imgdb,csv in sorted(files.items()):
    print(imgdb)
    if os.path.isfile(csv) == False:
        out = open(csv,'w')
        dist = pm.distribution(imgdb,(3,3))
        sorted_dist = sorted(dist, key=dist.get, reverse=True)
        for k in sorted_dist:
            line = k + "," + str(dist[k]) + "\n"
            out.write(line)
            #print(line)
        out.close()
    tag = imgdb[1].lstrip("img_dig/")
    pdict[(imgdb[0],tag)] = BPV.read_distribution_csv(csv)

#pdict[(8,"MacGill Calibrated")] = BPV.read_distribution_csv("p.macgill.csv")
n=512  #n = |Q|, the total number of patterns
N=50  #N = |N|, the wanted number of patterns
W=0.1  #rate

n_plots = len(pdict.keys())
fig = plt.plot()
f,axarr = plt.subplots(n_plots,1,sharex=True)
i = 0
for imgdb, p in sorted(pdict.items()):
    exact_solver = BPV.BPV("exact",n,N,W,p)
    exact_solver.solve()
    sol=exact_solver.solution_indexes()
    axarr[i].set_title(imgdb[1])
    axarr[i].plot(p)
    for point in sol:
        axarr[i].plot(sol,[0]*len(sol),'or')
    i += 1
plt.show()
