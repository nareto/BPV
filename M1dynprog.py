#import scipy.optimize
import matplotlib.pyplot as plt
import numpy as np
import queue
import pdb


def main():
    M=10  #M = |Q|, the total number of patterns
    n=2   #n = |N|, the wanted number of patterns
    W=0.2  #rate
    plot = 0
    
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

    ##define our price per unitary cost function
    #def unitary_cost(p):
    #    num = -p*np.log(p)
    #    den = max(1/n, p/W)
    #    return num/den
    #
    #sampled = np.zeros(len(p))
    #for i in range(len(p)):
    #    sampled[i] = unitary_cost(p[i])

    #Nset = M1dynprog(M,n,W,p)

    M1dynprog(M,n,W,p)
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


def is_valid_node(table,node):
    if len(table.shape) != 3:
        print("ERROR: valid_node only works for 3D matrices")
        return(0)
    x,y,z = table.shape
    i,j,k = node
    if any(c < 0 for c in node) or i >= x or j >= y or k >= z:
        return(0)
    else:
        return(1)
    
def M1dynprog(M,n,W,p,significant_figures=3):
    """Returns a tuple (exact_solution, cardinality, entropy)"""
    
    #greatest_values = []
    #Nset = []   #this will be the list of indexes in {1,...,M} that yield the solution
    #sum_xi = 0  #we use this to keep track of how many patterns we're adding to Nset
    #sum_pi = 0  #we use this to ensure that the so far chosen patterns don't exceed the maximum rate
    #search_space = [j for j in range(M)]
    #
    ##pdb.set_trace()
    #for i in range(n):
    #    greatest_values.append(search_space[0])
    #    for k in search_space:
    #        if smpl_opt_func[k] > smpl_opt_func[greatest_values[i]]:# and k not in greatest_values:
    #            greatest_values[i] = k
    #    arg_max = greatest_values[i] if greatest_values[i] != search_space[0] else search_space[0]
    #    search_space.remove(arg_max)
    #    sum_pi += p[arg_max]
    #    if sum_pi > W:
    #        break
    #    else:
    #        Nset.append(arg_max)
    #
    #if sum_pi > W:
    #    sum_pi -= p[arg_max]

    matrix_dim = (M,int(W*(10**significant_figures)),n)
    #table = np.zeros(matrix_dim[0]*matrix_dim[1]*matrix_dim[2])
    #table = table.reshape(matrix_dim)
    table = np.zeros(matrix_dim)
    #print(table.shape)
    node_queue = queue.Queue()
    root = (matrix_dim[0] - 1,matrix_dim[1] - 1,matrix_dim[2] - 1)
    node_queue.put(root)

    successor = {} 
    while(node_queue.empty() == False):
        node = node_queue.get(block=False)
        i,j,k = node
        #print(node)
        if i > 0:
            parents = []
            scaled_pi = int(p[i]*(10**significant_figures))
            #print(i, j, k, scaled_pi)
            if scaled_pi  > j:
                parent = (i - 1, j, k)
                node_queue.put(parent)
                #print(parent)
                table[parent] = table[node]
                #successor[node].append(parent)
                parents.append(parent)
            else:
                #pdb.set_trace()
                parent1 = (i - 1, j, k)
                parent2 = (i - 1, j - scaled_pi, k - 1)
                #print(parent1, parent2)
                if table[node] < table[parent1]:
                    table[parent1] = table[node]
                    #node_queue.put(parent1)
                    #successor[node].append(parent1)
                    parents.append(parent1)
                    
                if parent2[1] >= 0 and k >= 1 and table[node] - p[i]*np.log(1/p[i]) < table[parent2]:
                    table[parent2] = table[node] - p[i]*np.log(1/p[i])
                    #node_queue.put(parent2)
                    #successor[node].append(parent2)
                    parents.append(parent2)

            print(parents)
            for p in parents:
                successor[p] = (i,j,k)
            
    #print(table[0])
                    
    #or index,value in np.ndenumerate(table):
    #   if value != 0:
    #       print(index, " :: ", value)

    Nset = []
    sum_xi = 0
    sum_pi = 0
    max_index = np.argmin(table[0],0)
    max_index = (0, max_index[0], max_index[1])
    print(successor[max_index])
    #while:
        
    return table
    #return (Nset,sum_xi,sum_pi)

if __name__ == "__main__":
    main()
