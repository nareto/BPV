import pdb
import common
import M1exactsolver as m1e
import MVPeuristic as mvp
import numpy as np
import matplotlib.pyplot as plt
import os

def print_comparison_table(M,n,W,p, exact_solution, mvp_solution):
    """Prints table comparing entropy of exact and euristic solution"""

    exact_entropy = common.solution_entropy(p,exact_solution)
    exact_cardinality = common.solution_cardinality(exact_solution)
    exact_rate = common.solution_rate(p,exact_solution)
    
    mvp_entropy = common.solution_entropy(p,mvp_solution)
    mvp_cardinality = common.solution_cardinality(mvp_solution)
    mvp_rate = common.solution_rate(p,mvp_solution)

    columns = ("", "Cardinality (M=%d,n=%d)" % (M,n), "Rate (W=%f)" % W, "Entropy")
    rows = (("Exact solution", exact_cardinality, exact_rate, exact_entropy), \
            ("Euristic solution", mvp_cardinality, mvp_rate, mvp_entropy))
    #print("Exact entropy: %f \n Euristic entropy: %f \n 
    column_format = "|{:<20}|"+ "{:<25}|{:<20}|" + "{:<12}|"
    row_format = "|{:<20}|"+ "{:<25d}|{:<20.8f}|{:<12.8f}|"
    print(column_format.format(*columns))
    for r in rows:
        print(row_format.format(*r))
    print("|{:<20}|{:<25.12f}|".format("Relative Error", (exact_entropy-mvp_entropy)/exact_entropy))

def main():
    M=512  #M = |Q|, the total number of patterns
    n=50  #n = |N|, the wanted number of patterns
    #nsuM = 0.1
    W=0.1  #rate
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

    #define our price per unitary cost function
    def unitary_cost(p):
        num = -p*np.log(p)
        den = max(1/n, p/W)
        return num/den

    sampled = np.zeros(len(p))
    for i in range(len(p)):
        sampled[i] = unitary_cost(p[i])

    (Nset_mvp, card_mvp, rate_mvp) = mvp.MVPeuristic(M,n,W,p,sampled)
    (Nset_m1e, card_m1e, rate_m1e, entropy_m1e)  = m1e.M1exactsolver(M,n,W,p)
    #print(rate_m1e, rate_mvp)
    print_comparison_table(M,n,W,p,Nset_m1e, Nset_mvp)
    if plot:
        plt.figure(1)
        indexes = np.arange(M)
        plt.plot(indexes,20*p,'-r', alpha=0.2)
        plt.plot(indexes,sampled,'_b', alpha=1)

        xpoints = np.array([min(Nset_mvp),max(Nset_mvp)])
        ypoints = np.array([sampled[xpoints[0]],sampled[xpoints[1]]])
        plt.plot([xpoints[0],xpoints[0]],[0,sampled[xpoints[0]]], '--b', alpha=0.3)
        plt.plot([xpoints[1],xpoints[1]],[0,sampled[xpoints[1]]], '--b', alpha=0.3)
        
        #plt.plot(xpoints[0],0,xpoints[0],ypoints[0],xpoints[1],0,xpoints[1],ypoints[1],'-g')
        #plt.plot([xpoints[0], xpoints[0]], [0, ypoints[0]])
        
        plt.plot(Nset_m1e, np.zeros(len(Nset_m1e)), 'dr')
        plt.plot(Nset_mvp, np.zeros(len(Nset_mvp)), '.b', alpha=0.7)
        
        maximum = sampled[Nset_m1e[0]]
        plt.ylim(-1e-1*maximum,1.3*maximum)

        if savefig:
            file_to_save = savefig_path + "m1compare-M{0}n{1}W{2}.png".format(M,n,str(W).replace('.','-'))
            if os.path.isfile(file_to_save):
                ans = input("Overwrite {0}? y/N: ".format(file_to_save))
                if ans == 'y':
                    plt.savefig(file_to_save)
            else:
                plt.savefig(file_to_save)
        
        plt.show()
    
if __name__ == "__main__":
    main()
