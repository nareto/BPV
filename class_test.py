import BPV
import numpy as np
import matplotlib.pyplot as plt
import os
import timeit
import pdb
import cProfile

def main():
    n=20  #n = |Q|, the total number of patterns
    N=10  #N = |N|, the wanted number of patterns
    W=0.1  #rate
    eps = 1
    plot_dynprog_graph = 0
    load = 0
    if load == 1:
        #p = np.load("class_test-p.dump")
        p = np.load("bug1-p30.dump") #TODO
    else:
        p = np.random.exponential(1,n)
        
        #normalize
        p = (p/p.sum())
        
        #order p decreasingly
        p.sort()
        p = p[::-1]
        p.dump("class_test-p.dump")

    exact_solver = BPV.BPV("exact",n,N,W,p)
    ex_t = timeit.timeit(exact_solver.solve,number=1)
    #exact_solver.print_solution_summary()
    #print(exact_solver.solution_feasibility())

    solv = BPV.BPV("dynprog2", n,N,W,p)
    solv.decisiongraph_plot = plot_dynprog_graph
    def solve_dynprog():
        solv.solve(eps)
    s_t = timeit.timeit(solve_dynprog, number=1)
    

    #s_ex_solver = BPV.BPV("scaled_exact", n,N,W,p)
    #def solve_s_ex():
    #    s_ex_solver.solve(eps)
    #s_ex_t = timeit.timeit(solve_s_ex, number=1)
    #solv2 = BPV.BPV("euristic", n,N,W,p)
    #s2_t = timeit.timeit(solv2.solve,number=1)
    print(":::TIMES:::\nExact: ", ex_t, "\nDynProg2: ", s_t)#, "\nScaledExact: ", s_ex_t)


    BPV.print_comparison_table(exact_solver,solv)
    print("exact indixes = ", exact_solver.__solution_indexes__)
    #print("euristic extremes = ", min(solv2.__solution_indexes__), max(solv2.__solution_indexes__))
    print("dynprog indixes = ", solv.__solution_indexes__)
    #print("scaled exact indixes = ", s_ex_solver.__solution_indexes__)
if __name__ == "__main__":
    #cProfile.run('main()',sort=1)
    main()
