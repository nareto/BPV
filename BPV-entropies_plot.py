import BPV
import numpy as np
import timeit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import ipdb
import pickle

def main():
    tot_patterns=512
    min_card=10
    max_card=100
    min_rate=0.001
    max_rate=0.2
    card_nsteps = 10
    rate_nsteps = 30
    n_indexes = card_nsteps*rate_nsteps
    card_range = np.arange(min_card, max_card, (max_card-min_card)/card_nsteps)
    rate_range = np.arange(min_rate, max_rate, (max_rate-min_rate)/rate_nsteps)
    load = 1 #wether to load or calculate the graph data
    save_and_overwrite = 0 #wether to save (overwriting) the calculated graph data
    
    if load:
        p = np.load("BPV-entropies_plot-p.dump")
        f = open("BPV-entropies_plot-coords.dump", 'rb')
        coords = pickle.load(f)
        f.close()
    else:
        p = np.random.exponential(1,tot_patterns)
        #normalize
        p = (p/p.sum())
        #order p decreasingly
        p.sort()
        p = p[::-1]
        coords_x = np.zeros(n_indexes)
        coords_y = np.zeros(n_indexes)
        coords_z = np.zeros(n_indexes)
        coords = (coords_x,coords_y,coords_z)
        
        i = 0
        for card in card_range:
            for rate in rate_range:
                print("Solving ", i+1,"/",card_nsteps*rate_nsteps,"(cardinality = ", card," rate = ", rate, ")")
                exact_solver = BPV.BPV("exact",tot_patterns,int(card),rate,p)
                exact_solver.solve()
                e = exact_solver.solution_entropy()
                coords_x[i] = card
                coords_y[i] = rate
                coords_z[i] = e
                i+=1
                
        if save_and_overwrite:
            p.dump("BPV-entropies_plot-p.dump")
            f = open("BPV-entropies_plot-coords.dump", 'wb')
            pickle.dump(coords,f)
            f.close()
        
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for i in range(n_indexes):
        x,y,z = coords[0][i], coords[1][i], coords[2][i]
        ax.scatter(x,y,z,'.r')
        #ax.plot_surface(x,y,z)
        #ax.scatter(x**2,y**2,z**2,'-r')
    ax.set_xlabel('Maximum Cardinality')
    ax.set_ylabel('Maximum Rate')
    ax.set_zlabel('Achieved Entropy')
    plt.show()
    
if __name__ == "__main__":
    main()
