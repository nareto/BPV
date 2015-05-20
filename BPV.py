#import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from pandas import Series
import numpy as np
import pulp
import ipdb
import pdb

#def read_distribution_csv(file):
#    f = open(file,'r')
#    lines = f.readlines()
#    n = len(lines)
#    p = np.zeros(n)
#    i = 0
#    for l in lines:
#        binary_code, probability = l.split(',')
#        p[i] = float(probability)
#        i += 1
#    f.close()
#    return(p)


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
            

class BPV:
    def __init__(self,solver_type,tot_patterns,max_cardinality,max_rate,p):
        if len(p) != tot_patterns:
            err = "len(p) == tot_patterns must hold"
            raise RuntimeError(err)
        else:
            self.__is_solved__ = 0
            self.__all_solvers__ = {"exact": self.exact_solver, "euristic": self.euristic_solver,\
                                    "scaled_exact": self.scaled_exact_solver, "dynprog": self.dynprog_solver,\
                                    "dynprog2": self.dynprog_solver2}
            self.__solver_type__ = None
            self.tot_patterns = tot_patterns
            self.max_cardinality = max_cardinality
            self.max_rate = max_rate
            self.p = p
            self.plog1onp = self.p*np.log(1/self.p)
            self.set_solver(solver_type)
            self.tot_entropy = self.plog1onp.sum()


    def set_solver(self, solver_type):
        if solver_type not in self.__all_solvers__.keys():
            err = ("solver_type has to be one of"+" \"%s\""*len(self.__all_solvers__) % tuple(self.__all_solvers__.keys()))
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

    def solve(self, epsilon=0.05):
        ans=None
        if self.solved() == 1:
            ans = input("Problem is allready solved, solve again? y/N")
        if self.solved() == 0 or ans == "y":
            if self.solver() == None:
                print("ERROR: set solver type with set_solver")
            else:
                if self.solver() == "dynprog":
                    self.__all_solvers__["dynprog"](epsilon)
                elif self.solver() == "scaled_exact":
                    self.__all_solvers__["scaled_exact"](epsilon)
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
                  
    def solution_indexes(self):
        if self.solved():
            return Series(self.__solution_indexes__)
        else:
            return None
            
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
        2 if it violates the rate constraint, 3 if it violates both"""
        ret = 0
        if self.solved() == 1:
            if self.__solution_cardinality__ > self.max_cardinality:
                ret += 1                
            if self.__solution_rate__ > self.max_rate:
                ret += 2
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
            entropy += self.plog1onp[i]
        return entropy

    def calculate_cardinality(self, indexes):
        return len(indexes)
    
    def exact_solver(self):
        """Uses PuLP to calculate [one] exact solution"""

        pulp_instance = pulp.LpProblem(" (BPV) ",pulp.LpMaximize)
        
        self.__pulp_variables__ = []
        for i in range(self.tot_patterns):
            self.__pulp_variables__.append(pulp.LpVariable("x_%d" % i,0,1,pulp.LpInteger))
        
        cdotp = 0
        for i in range(self.tot_patterns):
            cdotp += self.plog1onp[i]*self.__pulp_variables__[i] #linear combination to optimize
            
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

    def scaled_exact_solver(self,epsilon):
        scaling_factor = epsilon*self.plog1onp[-1]/self.tot_patterns
        scaled_plog1onp = np.zeros(self.tot_patterns)
        for i in range(self.tot_patterns):
            scaled_plog1onp[i] = 1 + int(self.plog1onp[i]/scaling_factor)

        pulp_instance = pulp.LpProblem(" (BPV) ",pulp.LpMaximize)
        
        self.__pulp_variables__ = []
        for i in range(self.tot_patterns):
            self.__pulp_variables__.append(pulp.LpVariable("x_%d" % i,0,1,pulp.LpInteger))
        
        cdotp = 0
        for i in range(self.tot_patterns):
            cdotp += scaled_plog1onp[i]*self.__pulp_variables__[i] #linear combination to optimize
            
        pulp_instance += cdotp, "Entropy of the solution"
        
        constraint_cardinality = 0
        constraint_rate = 0
        for i in range(self.tot_patterns):
            constraint_cardinality += self.__pulp_variables__[i]
            constraint_rate += self.p[i]*self.__pulp_variables__[i]
        
        pulp_instance += constraint_cardinality <= self.max_cardinality, "Cardinality constraint"
        pulp_instance += constraint_rate <= self.max_rate, "Rate constraint"
        
        pulp_instance.solve()
        self.__solution_indexes__ = []
        self.__solution_cardinality__ = 0
        self.__solution_rate__ = 0
        self.__solution_entropy__ = 0
        for i in range(self.tot_patterns):
            if self.__pulp_variables__[i].value() != 0:
                self.__solution_entropy__ += self.plog1onp[i]
                self.__solution_indexes__.append(i)
                self.__solution_cardinality__ += 1
                self.__solution_rate__ += self.p[i]

    def euristic_solver(self):
        """The solution self.__solution_indexes__ is defined as a level curve of self.__sampled_euristic_cost__, that is
        self.__solution_indexes__ = {x | self.__sampled_euristic_cost__(x) > c} for some c which is determined by the constraints.
        We use the trivial method, which is O(self.max_cardinality*self.tot_patterns), to find the greatest values of f, checking
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
                self.__solution_entropy__ += self.plog1onp[arg_max]#self.p[arg_max]*np.log(1/self.p[arg_max])
                self.__solution_indexes__.append(arg_max)
                self.__solution_cardinality__ += 1

    def dynprog_solver(self,epsilon):
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
        self.dynprog_best_value = 0
        self.dynprog_best_value_node = (-1,-1,-1)
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
                if child[1] > self.dynprog_best_value:
                    self.dynprog_best_value = child[1]
                    self.dynprog_best_value_node = child
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
            if i + 1 < self.tot_patterns and j + reverse_cumulative_plog1onp[i] >= self.dynprog_best_value:
                child1 = (i+1,j,k)
                child2 = (i+1, j+scaled_plog1onp[i+1], k+1)
                if is_valid_cell(child1):
                    add_child(cur, (child1), 1)
                if is_valid_cell(child2) and table[cur] + self.p[i+1] <= self.max_rate:
                    add_child(cur, (child2),2)

        self.__solution_indexes__ , self.__solution_entropy__,\
            self.__solution_rate__, self.__solution_cardinality__ = check_path(self.dynprog_best_value_node,1)
        self.__solution_indexes__.sort()
            
        print("\nin table: ", self.dynprog_best_value , "  calculated (scaled): ", 1 + int(self.__solution_entropy__/scaling_factor),\
              " calculated: ", self.__solution_entropy__)

        if self.decisiongraph_plot == 1:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            for coords,n in extraction_order_of_nodes.items():
                x,y,z = coords
                ax.scatter(x,y,z,'r')
                ax.text(x,y,z, str(n), fontsize=9)
            
            leafs.remove(self.dynprog_best_value_node)
            for l in [self.dynprog_best_value_node] + leafs:
                cur = l
                if cur == self.dynprog_best_value_node:
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
            print(self.__solution_indexes__)
            ax.set_xlim(0,self.tot_patterns)
            ax.set_xlabel('Indexes')
            ax.set_ylabel('Scaled Entropy')
            ax.set_zlabel('Cardinality')
            plt.show()

    def dynprog_solver2(self):
        """Calculates solution using decision graph"""

        #scaled_plog1onp = self.plog1onp
        #scaled_plog1onp = np.zeros(self.tot_patterns)
        #for i in range(self.tot_patterns):
        #    scaled_plog1onp[i] = 1 + int(self.plog1onp[i]/scaling_factor)
        #scaled_tot_entropy = scaled_plog1onp.sum()

        alpha = {}
        graph_dimensions = (self.tot_patterns, self.max_rate, self.max_cardinality)
        root = (-1,0,0)
        alpha[root] = 0
        self.dynprog_best_value = 0
        self.dynprog_best_value_node = (-1,-1,-1)
        predecessor = {}
        leafs = []
        visitlist = [root]
        next_visitlist = []
        
        reverse_cumulative_plog1onp = np.zeros(self.tot_patterns)
        reverse_cumulative_plog1onp[self.tot_patterns - 1] = self.plog1onp[self.tot_patterns - 1]
        for i in np.arange(self.tot_patterns - 2, -1, -1):
            reverse_cumulative_plog1onp[i] = reverse_cumulative_plog1onp[i+1] + self.plog1onp[i]


        #graph_dimensions = (self.tot_patterns, scaled_tot_entropy, self.max_cardinality)
        def is_valid_cell(cell):
            k,mu,nu = cell
            if k >= graph_dimensions[0] or mu > graph_dimensions[1] or nu > graph_dimensions[2]:
                return(0)
            else:
                return(1)    

        def is_boundary_cell(cell):
            if is_valid_cell(cell) and any(cell[i] == graph_dimensions[i] - 1 for i in [0,1,2]):
                return(1)
            else:
                return(0)
            
        def add_child(parent, child, arc_type):
            "Looks at child and if feasible adds it to queue"
            if arc_type == 1:
                candidate_new_entropy = alpha[parent]
            elif arc_type == 2:
                candidate_new_entropy = alpha[parent] + self.plog1onp[child[0]]
            else:
                raise RuntimeError("arc_type must be either 1 or 2")
            add_child = 0
            add_to_next_visitlist = 0
            try:
                if candidate_new_entropy > alpha[child]:
                    add_child = 1
            except KeyError:
                add_child = 1
                add_to_next_visitlist = 1
            if add_child == 1:
                predecessor[child] = parent
                alpha[child] = candidate_new_entropy
                if add_to_next_visitlist == 1:
                    next_visitlist.append(child)
                if alpha[child] > self.dynprog_best_value:
                    self.dynprog_best_value = alpha[child]
                    self.dynprog_best_value_node = child
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
                    k = cur[0]
                    indexes.append(k)
                    cardinality += 1
                    rate += self.p[k]
                    entropy += self.plog1onp[k]
                    if print_taken_patterns:
                        print("taken pattern ", k, ", p[k] = ", self.p[k], "scaled plog1onp[k] = ", self.plog1onp[k])
                if next == root:
                    break
                else:
                    cur = next
            return(indexes,entropy,rate,cardinality)

        counter = 0
        extracted_nodes = {}
        while len(visitlist) != 0: #main loop
            #ipdb.set_trace()
            cur = visitlist.pop()
            if cur in extracted_nodes:
                raise RuntimeError("node has been inserted more than one time in visitlist")
            extracted_nodes[cur] = (counter, alpha[cur])
            counter += 1
            k,mu,nu = cur
            #if True:
            #if k+1 < self.tot_patterns:
            if k+1 < self.tot_patterns and alpha[cur] + reverse_cumulative_plog1onp[k] >= self.dynprog_best_value:
                child1 = (k+1,mu,nu)
                child2 = (k+1, mu+self.p[k+1], nu+1)
                if is_valid_cell(child1):
                    add_child(cur, (child1), 1)
                if is_valid_cell(child2) and mu + self.p[k+1] <= self.max_rate:
                    add_child(cur, (child2),2)
            if len(visitlist) == 0:
                visitlist = next_visitlist
                next_visitlist = []

        self.__solution_indexes__ , self.__solution_entropy__,\
            self.__solution_rate__, self.__solution_cardinality__ = check_path(self.dynprog_best_value_node,1)
        self.__solution_indexes__.sort()
            
        print("\nin graph: ", self.dynprog_best_value , "  calculated : ", self.__solution_entropy__,)

        if self.decisiongraph_plot == 1:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            for coords,n in extraction_order_of_nodes.items():
                x,y,z = coords
                ax.scatter(x,y,z,'r')
                ax.text(x,y,z, str(n), fontsize=9)
            
            leafs.remove(self.dynprog_best_value_node)
            for l in [self.dynprog_best_value_node] + leafs:
                cur = l
                if cur == self.dynprog_best_value_node:
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
            print(self.__solution_indexes__)
            ax.set_xlim(0,self.tot_patterns)
            ax.set_xlabel('Indexes')
            ax.set_ylabel('Scaled Entropy')
            ax.set_zlabel('Cardinality')
            plt.show()
            
        

