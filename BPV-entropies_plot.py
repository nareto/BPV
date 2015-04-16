import BPV
import numpy as np
import timeit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D



def main():
    tot_patterns=500
    min_card=10
    max_card=100
    min_rate=0.001
    max_rate=0.2
    card_nsteps = 5
    rate_nsteps = 3
    card_range = np.arange(min_card, max_card, (max_card-min_card)/card_nsteps)
    rate_range = np.arange(min_rate, max_rate, (max_rate-min_rate)/rate_nsteps)
    table = np.meshgrid
    #table = -np.ones((card_nsteps, rate_nsteps))
    #card_space = np.linspace(min_card,max_card,card_nsteps)
    #rate_space = np.linspace(min_rate,max_rate,rate_nsteps)
    p = np.random.exponential(1,tot_patterns)
    
    #normalize
    p = (p/p.sum())
    
    #order p decreasingly
    p.sort()
    p = p[::-1]
    p.dump("plot1-p.dump")
    #p = np.load("class_test-p.dump")

    i = 1
    for card in card_range:
        for rate in rate_range:
            print("Solving ", i,"/",card_nsteps*rate_nsteps,"(cardinality = ", card," rate = ", rate, ")")
            exact_solver = BPV.BPV("exact",tot_patterns,int(card),rate,p)
            exact_solver.solve()
            if exact_solver.solution_feasibility() != 0:
                print("ERROR: infeasible solution: ", exact_solver.solution_feasibility())
            else:
                #index = (card/card_nsteps,rate/rate_nsteps)
                #print(index)
                #table[index] = exact_solver.solution_entropy()
            i+=1

    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    ##plt.plot(table)
    #plot = ax.plot_surface(card_space, rate_space, table, rstride=1, cstride=1, linewidth=0, antialiased=False)
    #cb = fig.colorbar(plot, shrink=0.5)
    #plt.show()

if __name__ == "__main__":
    main()
