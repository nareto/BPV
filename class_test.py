import pdb
import common
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    M=20  #M = |Q|, the total number of patterns
    n=5  #n = |N|, the wanted number of patterns
    W=0.05  #rate

    p = np.random.exponential(1,M)
    
    #normalize
    p = (p/p.sum())
    
    #order p decreasingly
    p.sort()
    p = p[::-1]

    exact_solver = common.BPV("exact",M,n,W,p)
    #print(exact_solver.tot_patterns)
    #exact_solver.solve()
    #exact_solver.print_solution_summary()
    #print(exact_solver.solution_feasibility())

    solv = common.BPV("dynprog", M,n,W,p)
    solv.solve()
    solv.print_solution_summary()
    
if __name__ == "__main__":
    main()
