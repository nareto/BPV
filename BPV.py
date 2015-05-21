#import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pulp
import ipdb
import pdb
import timeit
from collections import deque

def dec_to_bin(number):
    return(bin(int(number))[2:])

class Data():
    def __init__(self,dataframe=None):
        try:
            l = len(dataframe)
            self.df = dataframe
        except TypeError:
            pass

    def copy(self):
        return(Data(self.df.copy()))
    
    def read_csv(self,csvfile, binary=True):
        self.df = pd.read_csv(csvfile,header=None,names=["pattern","p"])
        self.df["plog1onp"] = self.df["p"]*np.log(1/self.df["p"])
        if not binary:
            self.df["pattern"] = self.df["pattern"].map(dec_to_bin)
        #return(self.df)

    def data_head(self,rows=10):
        """Returns a DataFrame copy of the first rows rows of self.df, renormalizing vector p"""

        data_head = Data(pd.DataFrame(self.df[["pattern","p"]][:rows].copy()))
        sum  = data_head.df["p"].sum()
        data_head.df["p"] /= sum
        data_head.df["plog1onp"] = data_head.df["p"]*np.log(1/data_head.df["p"])
        return(data_head)

def relative_error(approximated_instance, exact_instance):
    if approximated_instance.solved() and exact_instance.solved():
        return abs((exact_instance.solution_entropy() - approximated_instance.solution_entropy())/exact_instance.solution_entropy())

def check_compatible_instances(*BPV_instances):
    """Returns True if all the BPV instances share the same (==) inputs, False otherwise"""

    attributes = [(x.tot_patterns,x.max_cardinality,x.max_rate,x.p) for x in BPV_instances]
    try:
         iterator = iter(attributes)
         first = next(iterator)
         return all(first == rest for rest in iterator)
    except StopIteration:
         return True

def print_comparison_table(*BPV_instances):
    """Prints table comparing solutions"""

    if check_compatible_instances(*BPV_instances) == False:
        print("ERROR: incompatible instances")
    else:
        exact_instance = None
        for instance in BPV_instances:
            if instance.solver() == "exact":
                exact_instance = instance
        tmpinst = BPV_instances[0]
        tot_patterns,max_cardinality,max_rate = (tmpinst.tot_patterns, tmpinst.max_cardinality, tmpinst.max_rate)
        print("tot_patterns = %d \nmax_cardinality = %d \nmax_rate = %f" % (tot_patterns,max_cardinality,max_rate))

        columns = ["", "Cardinality", "Rate", "Entropy"]
        if exact_instance != None:
            columns.append("Relative Error")
        rows = []
        for instance in BPV_instances:
            if instance.solved():
                row = [instance.solver(), instance.solution_cardinality(), instance.solution_rate(), instance.solution_entropy()]
                if exact_instance != None:
                    row.append(relative_error(instance, exact_instance))
                rows.append(row)

        column_format = "|{:<20}|{:<12}|{:<12}|{:<12}|"
        row_format = "|{:<20}|{:<12d}|{:<12.8f}|{:<12.8f}|"
        if exact_instance != None:
            column_format += "{:<25}"
            row_format += "{:<25.12f}"

        print(column_format.format(*columns))
        for r in rows:
            print(row_format.format(*r))

class BPV:
    def __init__(self,solver_name,data,max_cardinality,max_rate,epsilon=0.05,time_solver=False,):
        self.solved = False
        self.__all_solvers__ = {"pulp": self.pulp_solver, "euristic": self.euristic_solver,\
                                 "decgraph": self.decgraph_solver,\
                                "decgraph_epsilon": self.decgraph_solver_epsilon}
        self.data = data
        self.tot_patterns = len(data.df)
        self.max_cardinality = max_cardinality
        self.max_rate = max_rate
        self.epsilon = epsilon

        self.solver_name = solver_name
        self.set_solver()
        self.time_solver = time_solver


    def set_solver(self):
        if self.solver_name not in self.__all_solvers__.keys():
            err = ("solver_name has to be one of"+" \"%s\""*len(self.__all_solvers__) % tuple(self.__all_solvers__.keys()))
            raise RuntimeError(err)
        else:
            self.solver = self.__all_solvers__[self.solver_name]
                
    def solve(self, epsilon=0.05):
        if self.time_solver == True:
            self.solution_time = timeit.timeit(self.solver,number=1)
        else:
            self.solver()
        self.solved = True


    def pprint_solution(self):
        if self.solved == True:
            print("\nSolver = ", self.solver_name,\
                  "\nEntropy = ", self.solution_entropy, \
                  "\nCardinality = ", self.solution_cardinality,\
                  "\nRate = ", self.solution_rate,\
                  "\n\n", self.data.df[self.data.df[self.solver_name] == 1])
            
            if self.time_solver == True:
                print("Solution time in seconds = ", self.solution_time)
        else:
            print("Problem not solved")
                  
    def solution_interval_measure(self):
        """Returns a real number in [0,1], measuring by how much the solution is not an interval"""
        self.solution.sort_index(by="p",inplace=True,ascending=False)
        self.solution.sort_index(by="p",ascending=True,inplace=True)
        idx = pd.Index([j for j in range(len(self.solution))])
        self.solution.set_index(idx,inplace=True)

        minidx = self.solution.index.min()
        
    def pulp_solver(self):
        """Uses PuLP to calculate [one] pulp solution"""

        pulp_instance = pulp.LpProblem(" (BPV) ",pulp.LpMaximize)
        
        self.__pulp_variables__ = []
        for i in range(self.tot_patterns):
            self.__pulp_variables__.append(pulp.LpVariable("x_%d" % i,0,1,pulp.LpInteger))
        
        cdotp = 0
        for i in range(self.tot_patterns):
            cdotp += self.data.df["plog1onp"][i]*self.__pulp_variables__[i] #linear combination to optimize
            
        pulp_instance += cdotp, "Entropy of the solution"
        
        constraint_cardinality = 0
        constraint_rate = 0
        for i in range(self.tot_patterns):
            constraint_cardinality += self.__pulp_variables__[i]
            constraint_rate += self.data.df["p"][i]*self.__pulp_variables__[i]
        
        pulp_instance += constraint_cardinality <= self.max_cardinality, "Cardinality constraint"
        pulp_instance += constraint_rate <= self.max_rate, "Rate constraint"
        
        pulp_instance.solve()
        self.solution_entropy = pulp.value(pulp_instance.objective)
        indexes = []
        self.solution_cardinality = 0
        self.solution_rate = 0
        for i in range(self.tot_patterns):
            if self.__pulp_variables__[i].value() != 0:
                indexes.append(i)
                self.solution_cardinality += 1
                self.solution_rate += self.data.df["p"][i]
        self.solution = self.data.df.ix[indexes]
        index_series = np.zeros(self.tot_patterns)
        for i in indexes:
            index_series[i] = 1
        self.data.df['pulp'] = pd.Series(index_series)

        
    def euristic_solver(self):
        """The solution self.solution_indexes is defined as a level curve of sampled_euristic_cost, that is
        self.solution_indexes = {x | sampled_euristic_cost(x) > c} for some c which is determined by the constraints.
        We use the trivial method, which is O(self.max_cardinality*self.tot_patterns), to find the greatest values of f, checking
        on every iteration for the constraints to be respected. For the i-th greatest value of
        sampled_euristic_cost we store it's index in self.solution_indexes[i], i.e. sampled_euristic_cost[self.solution_indexes[i]] is the
        i-th greatest value of sampled_euristic_cost."""

        def euristic_unitary_cost(value):
            num = -value*np.log(value)
            den = max(1/self.max_cardinality, value/self.max_rate)
            return num/den

        df = self.data.df[['p','plog1onp']].copy()
        df['cost'] = np.NaN
        for i in df.index:
            df['cost'][i] = euristic_unitary_cost(df['p'][i])

        df.sort_index(by="cost",ascending=False,inplace=True)

        self.solution_cardinality = 0
        self.solution_rate = 0
        self.solution_entropy = 0
        indexes = []
        for i in df.index:
            p = df['p'][i]
            plog1onp = df['plog1onp'][i]
            if self.solution_cardinality + 1 <= self.max_cardinality\
               and self.solution_rate + p < self.max_rate:
                indexes.append(i)
                self.solution_cardinality += 1
                self.solution_rate += p
                self.solution_entropy += plog1onp
            else:
                break
        self.solution = self.data.df.ix[indexes]
        index_series = np.zeros(self.tot_patterns)
        for i in indexes:
            index_series[i] = 1
        self.data.df['euristic'] = pd.Series(index_series)
        
                 
    def decgraph_solver(self):
        """Calculates solution using decision graph"""

        df = self.data.df[['p','plog1onp']].copy()
        df.sort_index(by="p",ascending=True,inplace=True)
        idx = pd.Index([j for j in range(len(df))])
        mapper = pd.Series(df.index)
        df.set_index(idx,inplace=True)
        
        #p = df["p"]
        #plog1onp = df["plog1onp"]
        #indexing is much faster on a numpy array than on a pandas dataframe:
        p = np.array(df["p"])
        plog1onp = np.array(df["plog1onp"])

        alpha = {}
        predecessor = {}
        self.decgraph_best_value = -1
        self.decgraph_best_value_node = None
        root = (-1,0,0)
        alpha[root] = 0
        visitlist = deque()
        visitlist.append(root)
        next_visitlist = deque()

        #graph_dimensions = (self.tot_patterns, self.max_rate, self.max_cardinality)
        #leafs = []
        self.decgraph_len_visitlist = [1]
        
        reverse_cumulative_plog1onp = np.zeros(self.tot_patterns)
        reverse_cumulative_plog1onp[self.tot_patterns - 1] = plog1onp[self.tot_patterns - 1]
        for i in np.arange(self.tot_patterns - 2, -1, -1):
            reverse_cumulative_plog1onp[i] = reverse_cumulative_plog1onp[i+1] + plog1onp[i]

            
        def add_child(parent, child, candidate_new_entropy):
            "Looks at child and if feasible adds it to next_visitlist"
            add_child = 0
            add_to_next_visitlist = 0
            try:
                if candidate_new_entropy > alpha[child]: #Bellman condition
                    add_child = 1
            except KeyError:
                add_child = 1
                add_to_next_visitlist = 1
            if add_child == 1:
                predecessor[child] = parent
                alpha[child] = candidate_new_entropy
                if add_to_next_visitlist == 1:
                    next_visitlist.append(child)
                if alpha[child] > self.decgraph_best_value:
                    self.decgraph_best_value = alpha[child]
                    self.decgraph_best_value_node = child
                #if is_boundary_cell(child):
                #    leafs.append(child)
            
        def fchild1():
            pass
        
        def fchild2():
            pass
        
        #main loop
        while not not visitlist:
            cur = visitlist.pop()
            k,mu,nu = cur
            if k+1 < self.tot_patterns and alpha[cur] + reverse_cumulative_plog1onp[k] >= self.decgraph_best_value:
                child1 = (k+1,mu,nu)
                child2 = (k+1, mu+p[k+1], nu+1)
                if mu + p[k+1] <= self.max_rate:
                    add_child(cur, child1, alpha[cur])
                    #fchild1()
                    if nu + 1 <= self.max_cardinality:
                        add_child(cur, child2, alpha[cur] + plog1onp[k+1])
                        #fchild2()
            if not visitlist:
                self.decgraph_len_visitlist.append(len(next_visitlist))
                visitlist = next_visitlist
                next_visitlist = []

        print_taken_patterns=0
        cur = self.decgraph_best_value_node
        indexes = []
        self.solution_cardinality = 0
        self.solution_rate = 0
        self.solution_entropy = 0

        while 1:
            try:
                next = predecessor[cur]
            except KeyError:
                break
            if cur[1] != next[1]:
                k = cur[0]
                indexes.append(mapper[k])
                self.solution_cardinality += 1
                self.solution_rate += p[k]
                self.solution_entropy += plog1onp[k]
                if print_taken_patterns:
                    print("taken pattern ", k, ", p[k] = ", p[k], "plog1onp[k] = ", plog1onp[k])
            if next == root:
                break
            else:
                cur = next

        self.solution = self.data.df.ix[indexes]
        index_series = np.zeros(self.tot_patterns)
        for i in indexes:
            index_series[i] = 1
        self.data.df['decgraph'] = pd.Series(index_series)

        #self.decisiongraph_plot = 0
        #if self.decisiongraph_plot == 1:
        #    fig = plt.figure()
        #    ax = fig.gca(projection='3d')
        #    for coords,n in extraction_order_of_nodes.items():
        #        x,y,z = coords
        #        ax.scatter(x,y,z,'r')
        #        ax.text(x,y,z, str(n), fontsize=9)
        #        
        #    leafs.remove(self.decgraph_best_value_node)
        #    for l in [self.decgraph_best_value_node] + leafs:
        #        cur = l
        #        if cur == self.decgraph_best_value_node:
        #            linestyle='-ob'
        #        else:
        #            linestyle='-r'
        #        while 1:
        #            try:
        #                next = predecessor.pop(cur)
        #            except KeyError:
        #                break
        #            x = (cur[0], next[0])
        #            y = (cur[1], next[1])
        #            z = (cur[2], next[2])
        #            if cur[1] != next[1]:
        #                ax.plot(x,y,z,linestyle)
        #            else:
        #                ax.plot(x,y,z,'--g')
        #            if next == root:
        #                break
        #            else:
        #                cur = next
        #                #x =[2,5,4,7]
        #                #y=[1,6,6,7]
        #                #z=[7,2,45,6]
        #                #ax.plot(x,y,z, '--r')
        #    print(indexes)
        #    ax.set_xlim(0,self.tot_patterns)
        #    ax.set_xlabel('Indexes')
        #    ax.set_ylabel('Scaled Entropy')
        #    ax.set_zlabel('Cardinality')
        #    plt.show()

    def decgraph_solver_epsilon(self,epsilon):
        """Calculates an epsilon-solution using Dynamic Programming"""

        scaling_factor = 1#epsilon*self.plog1onp[-1]/self.tot_patterns
        scaled_plog1onp = self.plog1onp
        #scaled_plog1onp = np.zeros(self.tot_patterns)
        #for i in range(self.tot_patterns):
        #    scaled_plog1onp[i] = 1 + int(self.plog1onp[i]/scaling_factor)
        scaled_tot_entropy = scaled_plog1onp.sum()

        table = {}
        table_shape = (self.tot_patterns, scaled_tot_entropy, self.max_cardinality)
        heap = []
        heapq.heapify(heap)
        root = (-1,0,0)
        heapq.heappush(heap, root)
        table[root] = 0
        self.decgraph_best_value = 0
        self.decgraph_best_value_node = (-1,-1,-1)
        predecessor = {}
        leafs = []

        reverse_cumulative_plog1onp = np.zeros(self.tot_patterns)
        reverse_cumulative_plog1onp[self.tot_patterns - 1] = scaled_plog1onp[self.tot_patterns - 1]
        for i in np.arange(self.tot_patterns - 2, -1, -1):
            reverse_cumulative_plog1onp[i] = reverse_cumulative_plog1onp[i+1] + scaled_plog1onp[i]


        #table_shape = (self.tot_patterns, scaled_tot_entropy, self.max_cardinality)
        def is_valid_cell(cell):
            i,j,k = cell
            if i >= table_shape[0] or k > table_shape[2]:
                return(0)
            else:
                return(1)    

        def is_boundary_cell(cell):
            if is_valid_cell(cell) and any(cell[i] == table_shape[i] - 1 for i in [0,1,2]):
                return(1)
            else:
                return(0)
            
        def add_child(parent, child, arc_type):
            "Looks at child and if feasible adds it to queue"
            if arc_type not in [1,2]:
                raise RuntimeError("arc_type must be either 1 or 2")
            if arc_type == 1:
                candidate_new_rate = table[parent]
            else:
                candidate_new_rate = table[parent] + self.p[child[0]]
            add_child = 0
            add_to_heap = 0
            try:
                if candidate_new_rate < table[child]:
                    add_child = 1
            except KeyError:
                add_child = 1
                add_to_heap = 1
            if add_child == 1:
                predecessor[child] = parent
                table[child] = candidate_new_rate
                if add_to_heap == 1:
                    heapq.heappush(heap, child)
                if child[1] > self.decgraph_best_value:
                    self.decgraph_best_value = child[1]
                    self.decgraph_best_value_node = child
                if is_boundary_cell(child):
                    leafs.append(child)
            
        def check_path(coords, print_taken_patterns=0):
            cur = coords
            indexes = []
            cardinality = 0
            rate = 0
            entropy = 0
            while 1:
                try:
                    next = predecessor[cur]
                except KeyError:
                    break
                if cur[1] != next[1]:
                    i = cur[0]
                    indexes.append(i)
                    cardinality += 1
                    rate += self.p[i]
                    entropy += self.plog1onp[i]
                    if print_taken_patterns:
                        print("taken pattern ", i, ", p[i] = ", self.p[i], "scaled plog1onp[i] = ", scaled_plog1onp[i])
                if next == root:
                    break
                else:
                    cur = next
            return(indexes,entropy,rate,cardinality)

        counter = 0
        extracted_nodes = {}
        while heapq.nlargest(1,heap) != []: #main loop
            cur = heapq.heappop(heap)
            if cur in extracted_nodes:
                raise RuntimeError("node has been inserted more than one time in heap")
            counter += 1
            i,j,k = cur
            extracted_nodes[cur] = (counter, table[cur])
            #intable_rate = table[cur]
            if i + 1 < self.tot_patterns and j + reverse_cumulative_plog1onp[i] >= self.decgraph_best_value:
                child1 = (i+1,j,k)
                child2 = (i+1, j+scaled_plog1onp[i+1], k+1)
                if is_valid_cell(child1):
                    add_child(cur, (child1), 1)
                if is_valid_cell(child2) and table[cur] + self.p[i+1] <= self.max_rate:
                    add_child(cur, (child2),2)

        self.solution_indexes , self.solution_entropy,\
            self.solution_rate, self.solution_cardinality = check_path(self.decgraph_best_value_node,1)
        self.solution_indexes.sort()
            
        print("\nin table: ", self.decgraph_best_value , "  calculated (scaled): ", 1 + int(self.solution_entropy/scaling_factor),\
              " calculated: ", self.solution_entropy)

        if self.decisiongraph_plot == 1:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            for coords,n in extraction_order_of_nodes.items():
                x,y,z = coords
                ax.scatter(x,y,z,'r')
                ax.text(x,y,z, str(n), fontsize=9)
            
            leafs.remove(self.decgraph_best_value_node)
            for l in [self.decgraph_best_value_node] + leafs:
                cur = l
                if cur == self.decgraph_best_value_node:
                    linestyle='-ob'
                else:
                    linestyle='-r'
                while 1:
                    try:
                        next = predecessor.pop(cur)
                    except KeyError:
                        break
                    x = (cur[0], next[0])
                    y = (cur[1], next[1])
                    z = (cur[2], next[2])
                    if cur[1] != next[1]:
                        ax.plot(x,y,z,linestyle)
                    else:
                        ax.plot(x,y,z,'--g')
                    if next == root:
                        break
                    else:
                        cur = next
            #x =[2,5,4,7]
            #y=[1,6,6,7]
            #z=[7,2,45,6]
            #ax.plot(x,y,z, '--r')
            print(self.solution_indexes)
            ax.set_xlim(0,self.tot_patterns)
            ax.set_xlabel('Indexes')
            ax.set_ylabel('Scaled Entropy')
            ax.set_zlabel('Cardinality')
            plt.show()
            
        

