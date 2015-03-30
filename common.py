import numpy as np
import queue
import pulp
import timeit
import pdb

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

def is_non_increasing_vector(vector):
    if len(vector.shape) != 1:
        print("ERROR: is_decreasing_vector works only for 1D arrays")
    else:
        ret = True
        for i in range(len(vector) - 1):
            if vector[i+1] > vector[i]:
                ret = False
                break
        return ret
            
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
            self.tot_patterns = tot_patterns
            self.max_cardinality = max_cardinality
            self.max_rate = max_rate
            self.p = p
            self.set_solver(solver_type)


    def set_solver(self, solver_type):
        #print(self.tot_patterns)
        if solver_type not in self.__all_solvers__.keys():
            err = ("solver_type has to be one of"+" \"%s\""*len(solvers) % solvers.keys())
            raise RuntimeError(err)
        else:
            if solver_type == "dynprog" and not is_non_increasing_vector(self.p):
                print("ERROR: dynamic programming requires the probability vector to be decreasing")
            else:
                self.__solver_type__ = solver_type
            if solver_type == "euristic":
                self.__sampled_euristic_cost__ = np.zeros(self.tot_patterns)
                for i in range(self.tot_patterns):
                    self.__sampled_euristic_cost__[i] = self.euristic_unitary_cost(self.p[i])
                
    def euristic_unitary_cost(self,value):
        num = -value*np.log(value)
        den = max(1/self.max_cardinality, value/self.max_rate)
        return num/den

    def solved(self):
        return self.__is_solved__
    
    def solver(self):
        return self.__solver_type__

    def solve(self):
        ans=None
        if self.solved() == 1:
            ans = input("Problem is allready solved, solve again? y/N")
        if self.solved() == 0 or ans == "y":
            if self.solver() == None:
                print("ERROR: set solver type with set_solver")
            else:
                self.__all_solvers__[self.solver()]()
                self.__is_solved__ = 1


    def print_solution_summary(self):
        if self.solved() == 1:
            print("Solver = ", self.solver(),\
                  "\nEntropy = ", self.__solution_entropy__, \
                  "\nCardinality = ", self.__solution_cardinality__,\
                  "\nRate = ", self.__solution_rate__)
        else:
            print("Problem not solved")
                  
    def solution_cardinality(self):
        if self.solved():
            return self.__solution_cardinality__
        else:
            return None
        
    def solution_entropy(self):
        if self.solved():
            return self.__solution_entropy__
        else:
            return None
    
    def solution_rate(self):
        if self.solved():
            return self.__solution_rate__
        else:
            return None
        
    def solution_feasibility(self):
        """Returns -1 if problem is not solved, 0 if solution is feasibile, 1 if it violates the cardinality constraint,\
        2 if it violates the rate constraint"""
        if self.solved() == 1:
            if self.__solution_cardinality__ > self.max_cardinality:
                ret = 1                
            if self.__solution_rate__ > self.max_rate:
                ret = 2
            else:
                ret = 0
        else:
            ret = -1
        return ret

    def calculate_rate(self, indexes):
        rate = 0
        for i in indexes:
            rate += self.p[i]
        return rate

    def calculate_entropy(self, indexes):
        entropy = 0
        for i in indexes:
            entropy += self.p[i]*np.log(1/p[i])
        return entropy

    def calculate_cardinality(self, indexes):
        return len(indexes)
    
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
        self.__solution_entropy__ = 0
        search_space = [j for j in range(self.tot_patterns)]
    
        for i in range(self.max_cardinality):
            greatest_values.append(search_space[0])
            for k in search_space:
                if self.__sampled_euristic_cost__[k] > self.__sampled_euristic_cost__[greatest_values[i]]:# and k not in greatest_values:
                    greatest_values[i] = k
            #TODO: why did I originally write this and not simply arg_max = greatest_values[i] ?
            #arg_max = greatest_values[i] if greatest_values[i] != search_space[0] else search_space[0]
            arg_max = greatest_values[i]
            search_space.remove(arg_max)
            if self.__solution_rate__ + self.p[arg_max] > self.max_rate:
                break
            else:
                self.__solution_rate__ += self.p[arg_max]
                self.__solution_entropy__ += self.p[arg_max]*np.log(1/self.p[arg_max])
                self.__solution_indexes__.append(arg_max)
                self.__solution_cardinality__ += 1

    def dynprog_solver(self):
        """Calculates the solution using Dynamic Programming and Bellman's shortest path algorithm"""

        self.dynprog_table_dim = (self.tot_patterns+1,int(self.max_rate*(10**self.__dynprog_significant_figures__))+1,self.max_cardinality+1)
        self.dynprog_table = np.zeros(self.dynprog_table_dim) #TODO: would it be better to implement the table as a dictionary?
        #self.dynprog_table = dok_matrix() #dok_matrix is 2D only
        self.dynprog_scaled_p = np.zeros(self.tot_patterns,dtype='int')
        for i in range(self.tot_patterns):
            self.dynprog_scaled_p[i] = int(self.p[i]*(10**self.__dynprog_significant_figures__)) #can't do int() on numpy array
        node_queue = queue.Queue()
        inqueue = set()
        root = (self.dynprog_table_dim[0]-1,self.dynprog_table_dim[1]-1,self.dynprog_table_dim[2]-1)
        node_queue.put(root)
        inqueue.add(root)
        
        predecessor = {}
        best_node_value = 0
        best_node_index = None
        
        while(node_queue.empty() == False):
            node = node_queue.get(block=False)
            inqueue.remove(node)
            i,j,k = node
            #scaled_pi = int(self.p[i-1]*(10**self.__dynprog_significant_figures__)) #worst case: I'm calculating the same value #nodes times = tot_patterns*max_cardinality*max_rate*10**sig_fig
            child1 = (i - 1, j, k)
            child2 = (i - 1, j - self.dynprog_scaled_p[i-1], k - 1)
            if self.dynprog_scaled_p[i-1]  > j:
                if is_valid_node(self.dynprog_table,child1):
                    predecessor[child1] = node
                    self.dynprog_table[child1] = self.dynprog_table[node]
            else:
                if is_valid_node(self.dynprog_table,child1) and self.dynprog_table[node] < self.dynprog_table[child1]:
                    predecessor[child1] = node
                    self.dynprog_table[child1] = self.dynprog_table[node]
                    if child1 not in inqueue: #this is probably uneccessary, as I don't think 'child1 in queue' can ever hold
                        node_queue.put(child1)
                        inqueue.add(child1)
                if is_valid_node(self.dynprog_table,child2) and self.dynprog_table[node] - self.p[i-1]*np.log(1/self.p[i-1]) < self.dynprog_table[child2]:
                    self.dynprog_table[child2] = self.dynprog_table[node] - self.p[i-1]*np.log(1/self.p[i-1])
                    predecessor[child2] = node
                    if child2 not in inqueue:
                        node_queue.put(child2) 
                        inqueue.add(child2)
                    if self.dynprog_table[child2] < best_node_value:
                        best_node_value = self.dynprog_table[child2]
                        best_node_index = child2
                        self.dynprog_print_path_on_table(predecessor,best_node_index)
                        print("--------\n")
                        #print("new best_node_index = ", child2, "with value = ", self.dynprog_table[child2])
                        


        #for i,v in np.ndenumerate(self.dynprog_table):
        #    if v != 0:
        #        print("index = ", i, " value = ", v)
        
        ##check to be sure best_node_index was calculated correctly
        #min = 0
        #argmin = None
        #for (index,value) in np.ndenumerate(self.dynprog_table):
        #    if is_extreme_node(self.dynprog_table, index) and self.dynprog_table[index] < min:
        #        min = self.dynprog_table[index]
        #        argmin = index
        #print("best_node_index = ", best_node_index, "argmin = ", argmin)
                
        node = best_node_index
        #print("bni: ", best_node_index, "\nsucc: ",  predecessor[best_node_index],"\n",self.p[0], self.p[1])
        self.__solution_indexes__ = []
        self.__solution_cardinality__ = 0
        self.__solution_rate__ = 0
        self.__solution_entropy__ = 0
        #count = 0
        #print("--------")        
        while 1:
            try:
                pred = predecessor[node]
            except KeyError:
                break
            if pred[1] != node[1]:
                print("taking pattern %d with probability %f" % (pred[0]-1, self.p[pred[0]-1]))
                self.__solution_indexes__.append(pred[0]-1)
                self.__solution_cardinality__ += 1
                self.__solution_rate__ += self.p[pred[0]-1]
                self.__solution_entropy__ += self.p[pred[0]-1]*np.log(1/self.p[pred[0]-1])
            node = pred

        print("\n-best_node_value = ", -best_node_value,"entropy = ",self.__solution_entropy__, " rate = ", self.__solution_rate__)

    def dynprog_print_path_on_table(self,predecessor_dictionary, starting_node):
        n = starting_node
        rate = 0
        card = 0
        entropy = 0
        indexes = []
        while 1:
            try:
                pred = predecessor_dictionary[n]
            except KeyError:
                break
            if pred[1] != n[1]:
                i = pred[0] - 1
                print("taking pattern %d with probability %f and entropy %f" % (i, self.p[i], self.p[i]*np.log(1/self.p[i])))
                indexes.append(i)
                card += 1
                rate += self.p[i]
                entropy += self.p[i]*np.log(1/self.p[i])
            n = pred
        print("indexes = ", indexes, "cardinality = %d rate = %f entropy = %f value on table = %f" % (card, rate, entropy, self.dynprog_table[starting_node]))

