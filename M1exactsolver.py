#!/usr/bin/env python

import pulp
import numpy as np
import matplotlib.pyplot as plt
import common

def main():
    M = 50
    n=10
    W=0.1
    plot = 1
   
    #generate p
    p = np.random.normal(10,2,M)
    #p = np.exp(-(np.arange(0,5,1/M))**2)

    #normalize
    p = (p/p.sum())
    p.dump("M1exactsolver-p.dump")
    
    #p = np.load("M1exactsolver-p.dump")
    
    #order p decreasingly
    p.sort()
    p = p[::-1]

    Nset = M1exactsolver(M,n,W,p)

    if common.check_solution(M,n,W,p,Nset) == 1:
        print("Solution is not feasible, exiting")
        exit(1)

    #print(Nset)

    if plot:
        #plt.figure(1)
        #p_array = np.arange(0.0001,1/np.e,0.001)
        #plog1onp_plot = plt.plot(p_array,p_array*np.log(1/p_array),'b')
        #for i in Nset:
        #    plt.plot(i,0,'r')
        ##xpoints = np.log(np.array([Nset.min(),Nset.max()]))
        ##ypoints = np.array([f(p[xpoints[0]]),f(p[xpoints[0]])])
        #plt.show()
        plt.figure(1)
        indexes = np.arange(M)
        plt.plot(indexes,p,'-r',alpha=0.3)
        plt.plot(Nset,np.zeros(len(Nset)),'.r')
        #xpoints = np.log(np.array([Nset.min(),Nset.max()]))
        #ypoints = np.array([f(p[xpoints[0]]),f(p[xpoints[0]])])
        #plt.ylim(-0.1,max(p))
        plt.show()

def M1exactsolver(M,n,W,p):
    """Returns a tuple (exact_solution, cardinality, rate, entropy)"""
    # Create the 'M1' variable to contain the problem data
    M1 = pulp.LpProblem(" (M1) ",pulp.LpMaximize)
    
    x = []
    for i in range(M):
        x.append(pulp.LpVariable("x_%d" % i,0,1,pulp.LpInteger))
    
    plog1onp = p*np.log(1/p)
    cdotx = 0
    for i in range(M):
        cdotx += plog1onp[i]*x[i]
        
    M1 += cdotx, "Entropy of the solution"
    
    constraint_n = 0
    constraint_W = 0
    for i in range(M):
        constraint_n += x[i]
        constraint_W += p[i]*x[i]
    
    M1 += constraint_n <= n, "Cardinality constraint"
    M1 += constraint_W <= W, "Rate constraint"
    
    M1.solve()
    
    # The status of the solution is printed to the screen
    #print( "Status:", LpStatus[M1.status])

    Nset = []
    sum_xi = 0
    sum_pi = 0
    for i in range(M):
        if x[i].value() != 0:
            Nset.append(i)
            sum_xi += 1
            sum_pi += p[i]

    return (Nset, sum_xi, sum_pi, pulp.value(M1.objective))
    #print("Achieved Entropy", value(M1.objective))
  
if __name__ == "__main__":
    main()    
