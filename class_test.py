import BPV
import numpy as np
import matplotlib.pyplot as plt
import os
import timeit
import pdb

def main():
    M=20  #M = |Q|, the total number of patterns
    n=10  #n = |N|, the wanted number of patterns
    W=0.05  #rate
    eps = 3e-2
    plot_dynprog_graph = 0
    load = 0
    if load == 1:
        p = np.load("class_test-p.dump")
    else:
        p = np.random.exponential(1,M)
        
        #normalize
        p = (p/p.sum())
        
        #order p decreasingly
        p.sort()
        p = p[::-1]
        p.dump("class_test-p.dump")

    exact_solver = BPV.BPV("exact",M,n,W,p)
    ex_t = timeit.timeit(exact_solver.solve,number=1)
    #exact_solver.print_solution_summary()
    #print(exact_solver.solution_feasibility())

    solv = BPV.BPV("dynprog", M,n,W,p)
    solv.dynprog_table_plot = plot_dynprog_graph
    def solve_dynprog():
        solv.solve(eps)
    s_t = timeit.timeit(solve_dynprog, number=1)
    

    s_ex_solver = BPV.BPV("scaled_exact", M,n,W,p)
    def solve_s_ex():
        s_ex_solver.solve(eps)
    s_ex_t = timeit.timeit(solve_s_ex, number=1)
    #solv2 = BPV.BPV("euristic", M,n,W,p)
    #s2_t = timeit.timeit(solv2.solve,number=1)
    print(":::TIMES:::\nExact: ", ex_t, "\nDynProg: ", s_t, "\nScaledExact: ", s_ex_t)


    BPV.print_comparison_table(exact_solver,s_ex_solver,solv)
    #print("exact indixes = ", exact_solver.__solution_indexes__)
    #print("euristic extremes = ", min(solv2.__solution_indexes__), max(solv2.__solution_indexes__))
    print("dynprog indixes = ", solv.__solution_indexes__)
    print("scaled exact indixes = ", s_ex_solver.__solution_indexes__)
    print("dynprog entopy = ", solv.__solution_entropy__)
    print("scaled_exact entopy = ", s_ex_solver.__solution_entropy__)
if __name__ == "__main__":
    main()
