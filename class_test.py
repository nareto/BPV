import BPV
import numpy as np
import matplotlib.pyplot as plt
import os
import timeit
import pdb

def main():
    M=500  #M = |Q|, the total number of patterns
    n=50  #n = |N|, the wanted number of patterns
    W=0.1  #rate
    sig_fig = 2
    
    p = np.random.exponential(1,M)
    
    #normalize
    p = (p/p.sum())
    
    #order p decreasingly
    p.sort()
    p = p[::-1]
    p.dump("class_test-p.dump")
    #p = np.load("class_test-p.dump")

    exact_solver = BPV.BPV("exact",M,n,W,p)
    ex_t = timeit.timeit(exact_solver.solve,number=1)
    #exact_solver.print_solution_summary()
    #print(exact_solver.solution_feasibility())

    solv = BPV.BPV("dynprog", M,n,W,p,dynprog_significant_figures=sig_fig)
    s_t = timeit.timeit(solv.solve, number=1)
    solv2 = BPV.BPV("euristic", M,n,W,p)
    s2_t = timeit.timeit(solv2.solve,number=1)
    print(":::TIMES:::\nExact: ", ex_t, "\nDynProg: ", s_t, "\nEuristic: ", s2_t)
    #solv.print_solution_summary()
    #print(exact_solver.solution_entropy(), solv.solution_entropy(), solv2.solution_entropy())
    BPV.print_comparison_table(exact_solver,solv,solv2)
    #print("exact indixes = ", exact_solver.__solution_indexes__)
    #print("euristic extremes = ", min(solv2.__solution_indexes__), max(solv2.__solution_indexes__))
    #pdb.set_trace()
if __name__ == "__main__":
    main()
