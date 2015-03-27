#import scipy.optimize
import matplotlib.pyplot as plt
import numpy as np
import pdb


def main():
    M=1000  #M = |Q|, the total number of patterns
    n=100   #n = |N|, the wanted number of patterns
    W=0.07  #rate
    plot = 1
    
    #generate p
    p = np.random.normal(10,2,M)
    #p = np.exp(-(np.arange(0,5,1/M))**2)
    
    #normalize
    p = (p/p.sum())
    p.dump("MVPeuristica-p.dump")
    
    #p = np.load("MVPeuristica-p.dump")
    
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

    Nset = MVPeuristic(M,n,W,p,sampled)

    if plot:
        plt.figure(1)
        indexes = np.arange(M)
        plt.plot(indexes,200*p,'-r', alpha=0.2)
        plt.plot(indexes,sampled,'-b', alpha=0.5)
        
        xpoints = np.array([min(Nset),max(Nset)])
        ypoints = np.array([sampled[xpoints[0]],sampled[xpoints[1]]])
        plt.plot(xpoints,ypoints,'.g')
        
        plt.plot(Nset, np.zeros(len(Nset)), '.r')
        
        maximum = sampled[Nset[0]]
        plt.ylim(0,1.3*maximum)
        plt.show()
    
def MVPeuristic(M,n,W,p,smpl_opt_func):
    """Returns a tuple (mvp_euristic_solution, cardinality, rate)

    The solution Nset is defined as a level curve of smpl_opt_func, that is
    Nset = {x | smpl_opt_func(x) > c} for some c which is determined by the constraints.
    We use the trivial method, which is O(nM), to find the greatest values of f, checking
    on every iteration for the constraints to be respected. For the i-th greatest value of
    smpl_opt_func we store it's index in Nset[i], i.e. smpl_opt_func[Nset[i]] is the
    i-th greatest value of smpl_opt_func."""
    
    greatest_values = []
    Nset = []   #this will be the list of indexes in {1,...,M} that yield the solution
    sum_xi = 0  #we use this to keep track of how many patterns we're adding to Nset
    sum_pi = 0  #we use this to ensure that the so far chosen patterns don't exceed the maximum rate
    search_space = [j for j in range(M)]

    #pdb.set_trace()
    for i in range(n):
        greatest_values.append(search_space[0])
        for k in search_space:
            if smpl_opt_func[k] > smpl_opt_func[greatest_values[i]]:# and k not in greatest_values:
                greatest_values[i] = k
        arg_max = greatest_values[i] if greatest_values[i] != search_space[0] else search_space[0]
        search_space.remove(arg_max)
        sum_pi += p[arg_max]
        if sum_pi > W:
            break
        else:
            Nset.append(arg_max)

    if sum_pi > W:
        sum_pi -= p[arg_max]

    return (Nset,sum_xi,sum_pi)

if __name__ == "__main__":
    main()
