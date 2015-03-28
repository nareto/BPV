import pdb
import common
import M1exactsolver as m1e
import MVPeuristic as mvp
import M1dynprog as m1dp
import numpy as np
import matplotlib.pyplot as plt
import os

def print_comparison_table(M,n,W,p, exact_solution, mvp_solution, m1dp_solution):
    """Prints table comparing entropy of exact and euristic solution"""

    exact_entropy = common.solution_entropy(p,exact_solution)
    exact_cardinality = common.solution_cardinality(exact_solution)
    exact_rate = common.solution_rate(p,exact_solution)
    
    mvp_entropy = common.solution_entropy(p,mvp_solution)
    mvp_cardinality = common.solution_cardinality(mvp_solution)
    mvp_rate = common.solution_rate(p,mvp_solution)

    m1dp_entropy = common.solution_entropy(p,m1dp_solution)
    m1dp_cardinality = common.solution_cardinality(m1dp_solution)
    m1dp_rate = common.solution_rate(p,m1dp_solution)

    columns = ("", "Cardinality (M=%d,n=%d)" % (M,n), "Rate (W=%f)" % W, "Entropy")
    rows = (("Euristic solution", mvp_cardinality, mvp_rate, mvp_entropy), \
            ("Dynamic Programming", m1dp_cardinality, m1dp_rate, m1dp_entropy), \
            ("Exact solution", exact_cardinality, exact_rate, exact_entropy))
        
    #print("Exact entropy: %f \n Euristic entropy: %f \n 
    column_format = "|{:<20}|"+ "{:<25}|{:<20}|" + "{:<12}|"
    row_format = "|{:<20}|"+ "{:<25d}|{:<20.8f}|{:<12.8f}|"
    print(column_format.format(*columns))
    for r in rows:
        print(row_format.format(*r))
    print("|{:<20}|{:<25.12f}|".format("Euristic Error", (exact_entropy-mvp_entropy)/exact_entropy))
    print("|{:<20}|{:<25.12f}|".format("DynProg Error", (exact_entropy-m1dp_entropy)/exact_entropy))

def main():
    M=20  #M = |Q|, the total number of patterns
    n=5  #n = |N|, the wanted number of patterns
    #nsuM = 0.1
    W=0.05  #rate
    plot = 1
    savefig = 0
    savefig_path = "/home/renato/tesi/testo/img/"

    #n = int(M*nsuM)
    #generate p
    #p = np.random.normal(10,2,M)
    p = np.random.exponential(1,M)
    
    #normalize
    p = (p/p.sum())
    p.dump("M1compare_methods-p.dump")
    
    #p = np.load("M1compare_methods-p.dump")
    
    #order p decreasingly
    p.sort()
    p = p[::-1]

    exact_solver = common.BPVsolver("exact",M,n,W,p)
    exact_solver.solve()
    exact_solver.print_solution_summary()
    print(exact_solver.solution_feasibility())
        
if __name__ == "__main__":
    main()
