#import scipy.optimize
import matplotlib.pyplot as plt
import numpy as np
import queue
import pdb


def main():
    M=20  #M = |Q|, the total number of patterns
    n=6   #n = |N|, the wanted number of patterns
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

    Nset, cardinality, rate, entropy = M1dynprog(M,n,W,p)
    
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
    
def M1dynprog(M,n,W,p,significant_figures=2):
    """Returns a tuple (exact_solution, cardinality, entropy)"""
    
    matrix_dim = (M,int(W*(10**significant_figures)),n)
    table = np.zeros(matrix_dim)
    node_queue = queue.Queue()
    root = (matrix_dim[0] - 1,matrix_dim[1] - 1,matrix_dim[2] - 1)
    node_queue.put(root)

    number_of_extractions = {}
    successor = {}

    print("table size: ", table.size)
    
    while(node_queue.empty() == False):
        node = node_queue.get(block=False)
        try:
            number_of_extractions[node] += 1
        except KeyError:
            number_of_extractions[node] = 1
        #print(node)
        i,j,k = node
        parents = []
        scaled_pi = int(p[i]*(10**significant_figures))
        if scaled_pi  > j:
            parent = (i - 1, j, k)
            if is_valid_node(table,parent):
                node_queue.put(parent)
                table[parent] = table[node]
                parents.append(parent)
        else:
            parent1 = (i - 1, j, k)
            parent2 = (i - 1, j - scaled_pi, k - 1)
            #if table[node] < table[parent1] and is_valid_node(table,parent1):
            if is_valid_node(table,parent1):
                node_queue.put(parent1)
                table[parent1] = table[node]
                parents.append(parent1)
                   
            #if table[node] - p[i]*np.log(1/p[i]) < table[parent2] and is_valid_node(table,parent2):
            if is_valid_node(table,parent2):
                node_queue.put(parent2)
                table[parent2] = table[node] - p[i]*np.log(1/p[i])
                parents.append(parent2)

        #print(parents)
        for par in parents:
            successor[par] = node
            
    #print(table[0])
                    
    #for index,value in np.ndenumerate(table):
    #   if value != 0:
    #       print(index, " :: ", value)

    max = 0
    for k,v in iter(number_of_extractions.items()):
        if v > max:
            argmax = k
            max = v
    print("maximum node extractions: ", argmax,max)
    Nset = []
    sum_xi = 0
    sum_pi = 0
    sum_entropy = 0
    max_index = np.argmin(table[0],0)
    node = (0, max_index[0], max_index[1])

    #while 1:
    #    try:
    #        s = successor[node]
    #    except KeyError:
    #        break
    #    print(s[0] - node[0], s[1] - node[1])
    #    node = s
        
    #print(node, successor[node])
    while 1:
        try:
            succ = successor[node]
        except KeyError:
            break
        if succ[1] != node[1]:
            Nset.append(succ[0])
            sum_xi += 1
            sum_pi += p[succ[0]]
            sum_entropy += p[succ[0]]*np.log(1/p[succ[0]])
        node = succ
        
    #return table
    return (Nset,sum_xi,sum_pi,sum_entropy)

if __name__ == "__main__":
    main()
