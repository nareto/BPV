import numpy as np
import queue
import pulp
import timeit
import heapq
import ipdb
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
    return is_valid_node_shape(table.shape, node)

def is_valid_node_shape(shape,node):
    if len(shape) != 3:
        print("ERROR: is_valid_node_shape only works for 3D matrices")
        return(0)
    x,y,z = shape
    i,j,k = node
    if any(c < 0 for c in node) or i >= x or j >= y or k >= z:
        return(0)
    else:
        return(1)    

class dyn_prog_graph_node:
    def __init__(self,coords):
        self.coords = tuple(coords)

    def __eq__(self, other):
        return (self.coords == other.coords)
    
    def __lt__(self, other):
        if type(other) != type(self):
            raise NotImplementedError("Can only compare two dyn_prog_graph_node")
        if self.coords[0] == other.coords[0] and self.coords[1] == other.coords[1] and self.coords[2] != other.coords[2]:
            raise RuntimeError("Two nodes can't differ only in the last coordinate")
        else:
            selfinvcoords = (self.coords[0], -self.coords[1], self.coords[2])
            otherinvcoords = (other.coords[0], -other.coords[1], other.coords[2])
            return (selfinvcoords < otherinvcoords) #uses lexicographic ordering
            
class BPV:
    def __init__(self,solver_type,tot_patterns,max_cardinality,max_rate,p):
        if len(p) != tot_patterns:
            err = "len(p) == tot_patterns must hold"
            raise RuntimeError(err)
        else:
            self.__is_solved__ = 0
            self.__all_solvers__ = {"exact": self.exact_solver, "euristic": self.euristic_solver, "dynprog": self.dynprog_solver}
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
        """Calculates the solution using Dynamic Programming and Bellman's shortest path algorithm"""

        scaling_factor = epsilon*self.plog1onp[-1]/self.tot_patterns
        print("epsilon: ", epsilon, "scaling factor: ", scaling_factor)
        scaled_plog1onp = np.zeros(self.tot_patterns)
        for i in range(self.tot_patterns):
            scaled_plog1onp[i] = 1 + int(self.plog1onp[i]/scaling_factor)
        scaled_tot_entropy = scaled_plog1onp.sum()

        #pdb.set_trace()
        table = {}
        table_shape = (self.tot_patterns, scaled_tot_entropy, self.max_cardinality)
        heap = []
        heapq.heapify(heap)
        root = dyn_prog_graph_node((0,0,0))
        heapq.heappush(heap, root)
        table[root.coords] = 0
        self.dynprog_best_value = 0
        self.dynprog_best_value_node = dyn_prog_graph_node((-1,-1,-1))
        predecessor = {}
        
        def check_child(parent, child, arc_type):
            "Looks at child and if feasible ad.coordsds it to queue"
            if arc_type not in [1,2]:
                raise RuntimeError("arc_type must be either 1 or 2")
            updated = 0
            if child.coords in predecessor.keys():
                existent_rate = table[child.coords]
                if arc_type == 1:
                    candidate_new_rate = table[parent.coords]
                elif arc_type == 2:
                    candidate_new_rate = table[parent.coords] + self.p[child.coords[1] - 1]
                if candidate_new_rate < existent_rate:
                    predecessor[child.coords] = parent
                    table[child.coords] = candidate_new_rate
                    updated = 1
            else:
                predecessor[child.coords] = parent
                table[child.coords] = table[parent.coords]
                heapq.heappush(heap, child)
                updated = 1
                
            if updated == 1 and child.coords[1] > self.dynprog_best_value:
                    self.dynprog_best_value = child.coords[1]
                    self.dynprog_best_value_node = child
        
        while heapq.nlargest(1,heap) != []:
            cur = heapq.heappop(heap)
            i,j,k = cur.coords
            child1 = (i+1,j,k+1)
            child2 = (i+1, j+scaled_plog1onp[i], k+1)
            if is_valid_node_shape(table_shape,child1):
                check_child(cur, dyn_prog_graph_node(child1), 1)
            if is_valid_node_shape(table_shape,child2) and table[cur.coords] + self.p[i] <= self.max_rate:
                check_child(cur, dyn_prog_graph_node(child2),2)
    
        self.__solution_indexes__ = []
        self.__solution_cardinality__ = 0
        self.__solution_rate__ = 0
        self.__solution_entropy__ = 0
        intable_entropy = self.dynprog_best_value
        cur = self.dynprog_best_value_node

        while 1:
            try:
                next = predecessor[cur.coords]
            except KeyError:
                break
            if cur.coords[1] != next.coords[1]:
                i = cur.coords[0]
                self.__solution_indexes__.append(i-1)
                self.__solution_cardinality__ += 1
                self.__solution_rate__ += self.p[i-1]
                self.__solution_entropy__ += self.plog1onp[i-1]
            if next == root:
                break
            else:
                cur = next

        print("in table: ", intable_entropy , "  calculated (scaled): ", 1 + int(self.__solution_entropy__/scaling_factor),\
              " calculated: ", self.__solution_entropy__)
                
    def dynprog_print_path_on_table(self,predecessor_dictionary, starting_node):
        n = starting_node
        rate = 0
        card = 0
        entropy = 0
        indexes = []
        iteration = 1
        while 1:
            try:
                pred = predecessor_dictionary[n]
                print("node ", starting_node, "predecessor %d = " % iteration, pred)
            except KeyError:
                break
            if pred[2] != n[2]:
                i = pred[0] - 1
                print("taking pattern %d with probability = %f, rounded entropy = %f" % (i, self.p[i], self.dynprog_approx_plog1onp[i]))
                indexes.append(i)
                card += 1
                rate += self.p[i]
                entropy += self.p[i]*np.log(1/self.p[i])
            n = pred
            iteration += 1
        print("starting node = ", starting_node, "indexes = ", indexes, "cardinality = %d rate = %f entropy = %f value on table = %f" % (card, rate, entropy, self.dynprog_table[starting_node]))

