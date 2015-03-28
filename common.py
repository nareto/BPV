import numpy as np
import queue
import pdb
import pulp

def is_extreme_node(table,node):
    if len(table.shape) != 3:
        print("ERROR: is_extreme_node only works for 3D matrices")
        return(0)
    if is_valid_node(table,node) and any(c == 0 for c in node):
        return(1)
    else:
        return(0)
        
def is_valid_node(table,node):
    if len(table.shape) != 3:
        print("ERROR: is_valid_node only works for 3D matrices")
        return(0)
    x,y,z = table.shape
    i,j,k = node
    if any(c < 0 for c in node) or i >= x or j >= y or k >= z:
        return(0)
    else:
        return(1)
    
class BPV:
    def __init__(self,solver_type,tot_patterns,max_cardinality,max_rate,p,dynprog_significant_figures=3):
        if len(p) != tot_patterns:
            err = "len(p) == tot_patterns must hold"
            raise RuntimeError(err)
        else:
            self.__is_solved__ = 0
            self.__all_solvers__ = {"exact": self.exact_solver, "euristic": self.euristic_solver, "dynprog": self.dynprog_solver}
            self.__solver_type__ = None
            self.__dynprog_significant_figures__ = dynprog_significant_figures
            try:
                self.set_solver(solver_type)
            except:
                pass
            self.tot_patterns = tot_patterns
            self.max_cardinality = max_cardinality
            self.max_rate = max_rate
            self.p = p


    def set_solver(self, solver_type):
        #print(self.tot_patterns)
        print("pr")
        if solver_type not in self.__all_solvers__.keys():
            err = ("solver_type has to be one of"+" \"%s\""*len(solvers) % solvers.keys())
            raise RuntimeError(err)
        else:
            self.__solver_type__ = solver_type
            if solver_type == "euristic":
                #pdb.set_trace()
                self.__sampled_euristic_cost__ = np.zeros(self.tot_patterns)
                for i in range(self.tot_patterns):
                    self.__sampled_euristic_cost__[i] = self.euristic_unitary_cost(self.p[i])
                
    def euristic_unitary_cost(value):
        num = -value*np.log(value)
        den = max(1/self.max_cardinality, value/self.max_rate)
        return num/den

    def solver(self):
        return self.__solver_type__

    def solve(self):
        ans=None
        if self.__is_solved__ == 1:
            ans = input("Problem is allready solved, solve again? y/N")
        if self.__is_solved__ == 0 or ans == "y":
            if self.solver() == None:
                print("ERROR: set solver type with set_solver")
            else:
                #pdb.set_trace()
                self.__all_solvers__[self.solver()]()
                self.__is_solved__ = 1


    def print_solution_summary(self):
        if self.__is_solved__ == 1:
            print("Solver = ", self.solver(),\
                  "\nEntropy = ", self.__solution_entropy__, \
                  "\nCardinality = ", self.__solution_cardinality__,\
                  "\nRate = ", self.__solution_rate__)
        else:
            print("Problem not solved")
                  
    def solution_cardinality(self):
        if self.__is_solved__:
            return self.__solution__cardinality__
        else:
            return None
        
    def solution_entropy(self):
        if self.__is_solved__:
            return self.__solution__entropy__
        else:
            return None
    
    def solution_rate(self):
        if self.__is_solved__:
            return self.__solution__rate__
        else:
            return None
        
    def solution_feasibility(self):
        """Returns -1 if problem is not solved, 0 if solution is feasibile, 1 if it violates the cardinality constraint,\
        2 if it violates the rate constraint"""
        if self.__is_solved__ == 1:
            if self.__solution_cardinality__ > self.max_cardinality:
                ret = 1                
            if self.__solution_rate__ > self.max_rate:
                ret = 2
            else:
                ret = 0
        else:
            ret = -1
        return ret
    
    def exact_solver(self):
        """Uses PuLP to calculate [one] exact solution"""

        pulp_instance = pulp.LpProblem(" (BPV) ",pulp.LpMaximize)
        
        self.__pulp_variables__ = []
        for i in range(self.tot_patterns):
            self.__pulp_variables__.append(pulp.LpVariable("x_%d" % i,0,1,pulp.LpInteger))
        
        plog1onp = self.p*np.log(1/self.p)
        cdotp = 0
        for i in range(self.tot_patterns):
            cdotp += plog1onp[i]*self.__pulp_variables__[i] #linear combination to optimize
            
        pulp_instance += cdotp, "Entropy of the solution"
        
        constraint_cardinality = 0
        constraint_rate = 0
        for i in range(self.tot_patterns):
            constraint_cardinality += self.__pulp_variables__[i]
            constraint_rate += self.p[i]*self.__pulp_variables__[i]
        
        pulp_instance += constraint_cardinality <= self.max_cardinality, "Cardinality constraint"
        pulp_instance += constraint_rate <= self.max_rate, "Rate constraint"
        
        pulp_instance.solve()
        self.__solution_entropy__ = pulp.value(pulp_instance.objective)
        self.__solution_indexes__ = []
        self.__solution_cardinality__ = 0
        self.__solution_rate__ = 0
        for i in range(self.tot_patterns):
            if self.__pulp_variables__[i].value() != 0:
                self.__solution_indexes__.append(i)
                self.__solution_cardinality__ += 1
                self.__solution_rate__ += self.p[i]

    def euristic_solver(self):
        """The solution self.__solution_indexes__ is defined as a level curve of self.__sampled_euristic_cost__, that is
        self.__solution_indexes__ = {x | self.__sampled_euristic_cost__(x) > c} for some c which is determined by the constraints.
        We use the trivial method, which is O(nself.tot_patterns), to find the greatest values of f, checking
        on every iteration for the constraints to be respected. For the i-th greatest value of
        self.__sampled_euristic_cost__ we store it's index in self.__solution_indexes__[i], i.e. self.__sampled_euristic_cost__[self.__solution_indexes__[i]] is the
        i-th greatest value of self.__sampled_euristic_cost__."""
        
        greatest_values = []
        self.__solution_indexes__ = []   #this will be the list of indexes in {1,...,self.tot_patterns} that yield the solution
        self.__solution_cardinality__ = 0  #we use this to keep track of how many patterns we're adding to self.__solution_indexes__
        self.__solution_rate__ = 0  #we use this to ensure that the so far chosen patterns don't exceed the maximum rate
        search_space = [j for j in range(self.tot_patterns)]
    
        for i in range(self.max_cardinality):
            greatest_values.append(search_space[0])
            for k in search_space:
                if self.__sampled_euristic_cost__[k] > self.__sampled_euristic_cost__[greatest_values[i]]:# and k not in greatest_values:
                    greatest_values[i] = k
            arg_max = greatest_values[i] if greatest_values[i] != search_space[0] else search_space[0]
            search_space.remove(arg_max)
            self.__solution_rate__ += p[arg_max]
            if self.__solution_rate__ > self.max_rate:
                break
            else:
                self.__solution_indexes__.append(arg_max)
                self.__solution_cardinality__ += 1
    
        if self.__solution_rate__ > self.max_rate:
            self.__solution_rate__ -= p[arg_max]
    
        return (self.__solution_indexes__,self.__solution_cardinality__,self.__solution_rate__)

    def dynprog_solver(self):
        """Calculates the solution using Dynamic Programming and Bellman's shortest path algorithm"""
        
        self.dynprog_table_dim = (self.tot_patterns,int(self.max_rate*(10**self.__dynprog_significant_figures__)),self.max_cardinality)
        self.dynprog_table = np.zeros(self.dynprog_table_dim)
        #self.dynprog_table = dok_matrix() #dok_matrix is 2D only
        node_queue = queue.Queue()
        root = (self.dynprog_table_dim[0] - 1,self.dynprog_table_dim[1] - 1,self.dynprog_table_dim[2] - 1)
        node_queue.put(root)
    
        number_of_extractions = {}
        successor = {}
    
        while(node_queue.empty() == False):
            node = node_queue.get(block=False)
            try:
                number_of_extractions[node] += 1
            except KeyError:
                number_of_extractions[node] = 1
            i,j,k = node
            parents = []
            scaled_pi = int(self.p[i]*(10**self.__dynprog_significant_figures__))
            if scaled_pi  > j:
                parent = (i - 1, j, k)
                if is_valid_node(self.dynprog_table,parent):
                    node_queue.put(parent)
                    self.dynprog_table[parent] = self.dynprog_table[node]
                    parents.append(parent)
            else:
                parent1 = (i - 1, j, k)
                parent2 = (i - 1, j - scaled_pi, k - 1)
                if is_valid_node(self.dynprog_table,parent1):
                    node_queue.put(parent1)
                    self.dynprog_table[parent1] = self.dynprog_table[node]
                    parents.append(parent1)
                if is_valid_node(self.dynprog_table,parent2):
                    node_queue.put(parent2)
                    self.dynprog_table[parent2] = self.dynprog_table[node] - self.p[i]*np.log(1/self.p[i])
                    parents.append(parent2)

            for par in parents:
                successor[par] = node
                
        max = 0
        for k,v in iter(number_of_extractions.items()):
            if v > max:
                argmax = k
                max = v
        print("maximum node extractions: ", argmax,max)
    
        #TODO: the next for is very expensive - find a better way to obtain extreme_nodes
        extreme_nodes = []
        for (index,value) in np.ndenumerate(self.dynprog_table):
            if is_extreme_node(self.dynprog_table, index):
                extreme_nodes.append(index)
    
        min_value = 0
        argmin = None
        for index in extreme_nodes:
            if self.dynprog_table[index] < min_value:
                min_value = self.dynprog_table[index]
                argmin = index
        node = argmin        
    
        self.__solution_indexes__ = []
        self.__solution_cardinality__ = 0
        self.__solution_rate__ = 0
        self.__solution_entropy__ = 0
        
        print(len(successor))
        while 1:
            try:
                succ = successor[node]
            except KeyError:
                break
            if succ[1] != node[1]:
                self.__solution_indexes__.append(succ[0])
                self.__solution_cardinality__ += 1
                self.__solution_rate__ += self.p[succ[0]]
                self.__solution_entropy__ += self.p[succ[0]]*np.log(1/self.p[succ[0]])
            node = succ
            
        #return (self.__solution_indexes__,self.__solution_cardinality__,self.__solution_rate__,self.__solution_entropy__)
