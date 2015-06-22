#import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pattern_manipulation as pm
import pandas as pd
import numpy as np
import random
import pulp
import timeit
from collections import deque
import ipdb
import pdb
import gc

def euclidean_distance(pdseries1,pdseries2):
    dist = 0
    for i in pdseries1.index:
        dist += (pdseries1[i] - pdseries2[i])**2
    return(np.sqrt(dist))

def distance1(pdseries1,pdseries2):
    sup = 0
    for i in pdseries1.index:
        den_sum = 1
        for j in pdseries2.index:
            den_sum += np.abs(pdseries1[i] - pdseries2[j])
        candidate = np.abs(pdseries1[i] -pdseries2[i])/den_sum
        if candidate > sup:
            sup = candidate
    return(sup)

def solution_strings(*BPV_instances):
    """Returns list of the names of boolean solution columns"""
    
    if check_compatible_instances(*BPV_instances):
        strings = []
        for instance in BPV_instances:
            if instance.multiple_solutions != None:
                for i in range(instance.multiple_solutions):
                    strings.append(instance.solver_name + str(i))
            else:
                strings.append(instance.solver_name)
        return(strings)
                
def solution_only_view(*BPV_instances):
    """Returns view on data with only boolean solution columns"""
    
    if check_compatible_instances(*BPV_instances):
        col_names = solution_strings(*BPV_instances)
        return(BPV_instances[0].data.df[col_names])

def solution_non_zero_view(pattern_data=True,*BPV_instances):
    """Returns view on data restricted to rows being part of some solution"""
    
    if check_compatible_instances(*BPV_instances):
        sol_col_names = solution_strings(*BPV_instances)
        min_idx = len(BPV_instances[0].data.df)
        max_idx = 0
        for col in sol_col_names:
            data = BPV_instances[0].data.df
            new_min = data.ix[data[col] == True].index.min()
            new_max = data.ix[data[col] == True].index.max()
            if new_min < min_idx:
                min_idx = new_min
            if new_max > max_idx:
                max_idx = new_max
        if pattern_data:
            view = data.ix[min_idx:max_idx]
        else:
            view = data[sol_col_names].ix[min_idx:max_idx]
        return(view)
    
def dec_to_bin(number):
    """Converts decimal number to binary string"""
    
    return(bin(int(number))[2:])

def bin_to_dec(bin_string):
    """Converts binary string to natural number"""

    ret = 0
    power = len(bin_string) - 1
    for i in bin_string:
        ret += 2**power * int(i)
        power -= 1
    return(ret)

class Data():
    """Class representing the data needed by the BPV class"""
    
    def __init__(self,dataframe=None):
        try:
            l = len(dataframe)
            self.df = dataframe
        except TypeError:
            pass

    def copy(self):
        """Returns a new Data instance with a copy of the DataFrame"""
        
        return(Data(self.df.copy()))

    def artificial_noise1(self, variance, recalculate_entropy=True):
        """Returns a new Data instance with a gaussian noise on the p column"""

        cols = list(self.df.columns)
        cols.remove('p')
        p = self.df['p']
        df_length = len(self.df)
        p_err = np.zeros(df_length)
        for i in range(df_length):
            p_err[i] = np.abs(p[i] + np.random.normal(0,variance))
        p_err = pd.Series(p_err/p_err.sum())
        #new_df = pd.DataFrame(index=self.index, columns=cols)
        new_df = self.df[cols].copy()
        new_df['p'] = p_err
        new_df = Data(new_df)
        if recalculate_entropy:
            cols_minus_entropy = list(new_df.df.columns)
            cols_minus_entropy.remove('plog1onp')
            new_df.df = new_df.df[cols_minus_entropy].copy()
            new_df.calculate_entropy()
        return(new_df)

    def artificial_noise2(self, variance_factor,recalculate_entropy=True):
        """Returns a new Data instance with a gaussian noise on the p column.

        Variance for each component p_i is variance_factor*p_i"""

        cols = list(self.df.columns)
        cols.remove('p')
        p = self.df['p']
        df_length = len(self.df)
        p_err = np.zeros(df_length)
        for i in range(df_length):
            p_err[i] = np.abs(p[i] + np.random.normal(0,variance_factor*p[i]))
        p_err = pd.Series(p_err/p_err.sum())
        #new_df = pd.DataFrame(index=self.index, columns=cols)
        new_df = self.df[cols].copy()
        new_df['p'] = p_err
        new_df = Data(new_df)
        if recalculate_entropy:
            cols_minus_entropy = list(new_df.df.columns)
            cols_minus_entropy.remove('plog1onp')
            new_df.df = new_df.df[cols_minus_entropy].copy()
            new_df.calculate_entropy()
        return(new_df)

    def calculate_entropy(self):
        self.df["plog1onp"] = self.df["p"]*np.log(1/self.df["p"])
        
    def read_csv(self,csvfile, binary=True):
        """Reads a CSV with columns: pattern-string,p\

        pattern-string can either be binary or decimal, in which case  'binary' must be set to False"""
        
        self.df = pd.read_csv(csvfile,header=None,names=["pattern-string","p"])
        self.calculate_entropy()
        if not binary:
            self.df["pattern-string"] = self.df["pattern-string"].map(dec_to_bin)
        s2p = lambda x: pm.string2pattern(x,(3,3))
        self.df['pattern-matrix'] = self.df['pattern-string'].apply(s2p)
        self.df = self.df.reindex_axis(['pattern-string','pattern-matrix','p','plog1onp'],axis=1)
        #return(self.df)

    def data_head(self,rows=10):
        """Returns a DataFrame copy of the first rows rows of self.df, renormalizing vector p"""

        data_head = Data(pd.DataFrame(self.df[["pattern-string","p"]][:rows].copy()))
        sum  = data_head.df["p"].sum()
        data_head.df["p"] /= sum
        data_head.df["plog1onp"] = data_head.df["p"]*np.log(1/data_head.df["p"])
        return(data_head)

            
        
def distance_solutions(sol1, sol2):
    """Returns a tuple (sup,argsup) of the solution distance.\

    sol1 and sol2 must be a boolean array, typically Data.df['solvername']"""
    
    sup = 0
    argsup = -1
    i = 0
    for value1 in sol1:
        num = sol1[i] - sol2[i]
        if num == 0:
            i+=1
            continue
        else:
            den = 1
            for value2 in sol2:
                den += value1 - value2
            value1dist = num/den
            if value1dist > sup:
                sup = value1dist
                argsup = i
            i+=1
    return(sup,argsup)
    
    
def relative_error(approximated_instance, exact_instance):
    """Returns the relative error made by approximated_instance with respect to exact_instance"""
    
    if approximated_instance.solved and exact_instance.solved:
        return abs((exact_instance.solution_entropy - approximated_instance.solution_entropy)/exact_instance.solution_entropy)

def check_compatible_instances(*BPV_instances):
    """Returns True if all the BPV instances share the same (==) inputs, False otherwise"""

    attributes = [(x.tot_patterns,x.max_cardinality,x.max_rate,x.data) for x in BPV_instances]
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
            if instance.solved:
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
    """Class representing an instance of BPV. There are various solver methods:\
    
    pulp: uses Pulp (python universal linear programming) library to calculate an exact solution
    heuristic: uses the heuristic approximation given by Bruni, Punzi and del Viva
    decgraphH: uses the decision graph algorithm with V_{k,\mu,\nu} subproblems
    decgraphW: uses the decision graph algorithm with W_{k,v,\nu} subproblems"""
    
    def __init__(self,solver_name,data,max_cardinality,max_rate,time_solver=False,use_quantized_entropy=False):
        self.solved = False
        self.__all_solvers__ = {"pulp": self.pulp_solver, "heuristic": self.heuristic_solver,\
                                 "decgraphH": self.decgraphH_solver,"decgraphW": self.decgraphW_solver}
        self.multiple_solutions = None
        self.selected_solution = None
        self.data = data
        self.tot_patterns = len(data.df)
        self.max_cardinality = max_cardinality
        self.max_rate = max_rate

        self.solver_name = solver_name
        self.set_solver()
        self.time_solver = time_solver
        self.use_quantized_entropy = use_quantized_entropy

    def set_solver(self):
        if self.solver_name not in self.__all_solvers__.keys():
            err = ("solver_name has to be one of"+" \"%s\""*len(self.__all_solvers__) % tuple(self.__all_solvers__.keys()))
            raise RuntimeError(err)
        else:
            self.solver = self.__all_solvers__[self.solver_name]
                
    def solve(self):
        if self.time_solver == True:
            self.solution_time = timeit.timeit(self.solver,number=1)
        else:
            self.solver()



    def pprint_solution(self):
        if self.solved == True:
            print("\nSolver = ", self.solver_name,\
                  "\nEntropy = ", self.solution_entropy, \
                  "\nCardinality = ", self.solution_cardinality,\
                  "\nRate = ", self.solution_rate)
            if self.multiple_solutions != None:
                print("\n{0} equivalent solutions found".format(self.multiple_solutions))
                i = 0
                for sol in self.solution:
                    self.selected_solution = i
                    print("\n Solution {0} Interval Measure = ".format(i),\
                          self.solution_interval_measure())
                    i+=1
            #else:
            #    print("\n\n", self.data.df[self.data.df[self.solver_name] == 1])
            if self.time_solver == True:
                print("Solution time in seconds = ", self.solution_time)
        else:
            print("Problem not solved")
                  
    def solution_interval_measure(self):
        """Returns a real number in [0,1], measuring by how much the solution is not an interval"""
        
        index = self.selected_solution
        if index != None:
            sol = self.solution[index]
            name = self.solver_name + str(index)
        else:
            sol = self.solution
            name = self.solver_name
        sol.sort_index(by="p",inplace=True,ascending=False)
        sol.sort_index(by="p",ascending=True,inplace=True)
        idx = pd.Index([j for j in range(len(sol))])
        sol.set_index(idx,inplace=True)

        solution = self.data.df[self.data.df[name] != 0]
        minidx = solution.index.min()
        maxidx = solution.index.max()
        holes = self.data.df[self.data.df[name] == 0].ix[minidx:maxidx]
        return(0.5*len(holes)/(maxidx - minidx))

    def paths_list(self,starting_node,rec_level=0,solutions_list=None,first_choice=None,multiple_solutions = None,sol_list_index=0):
        """Returns list of paths that end in node"""

        if rec_level == 0:
            solutions_list = [[]]
            indexes = solutions_list[0]
        indexes = solutions_list[sol_list_index]
        cur_node = starting_node
        while 1:
            if first_choice == 0:
                next_node = self.predecessor[cur_node][0]
            elif first_choice == 1:
                next_node = self.predecessor[cur_node][1]
            elif cur_node != self.decgraph_root and len(self.predecessor[cur_node]) > 1:
                if multiple_solutions == None:
                    multiple_solutions = 2
                else:
                    multiple_solutions += 1
                indexes2 = indexes.copy()
                solutions_list.append(indexes2)
                indexes2_index = len(solutions_list) - 1 
                self.paths_list(cur_node,rec_level+1,solutions_list,0,multiple_solutions,sol_list_index)
                self.paths_list(cur_node,rec_level+1,solutions_list, 1,multiple_solutions,indexes2_index)
                if rec_level > 0:
                    break
                else:
                    return(solutions_list)
            elif cur_node != self.decgraph_root:
                next_node = self.predecessor[cur_node][0]
            if cur_node[1] > 0:
                if cur_node[1] != next_node[1]:
                    indexes.append(self.decgraph_index_mapper[cur_node[0]])
                first_choice = None
                cur_node = next_node
            else: #cur_node == self.decgraph_root or cur_node[1] == 0
                if multiple_solutions == None:
                    return(solutions_list)
                else:
                    break

    def write_cplex_lp(self,filepath):
        """Uses PuLP .writeLP method to write the method to filepath in CPLEX LP format"""

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

        pulp_instance.writeLP(filepath)


        
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
            index_series[i] = True
        self.data.df['pulp'] = pd.Series(index_series,dtype='bool')
        self.selected_solution = 0
        self.solved = True
        
    def heuristic_solver(self):
        """The solution self.solution_indexes is defined as a level curve of sampled_heuristic_cost, that is
        self.solution_indexes = {x | sampled_heuristic_cost(x) > c} for some c which is determined by the constraints.
        We use the trivial method, which is O(self.max_cardinality*self.tot_patterns), to find the greatest values of f, checking
        on every iteration for the constraints to be respected. For the i-th greatest value of
        sampled_heuristic_cost we store it's index in self.solution_indexes[i], i.e. sampled_heuristic_cost[self.solution_indexes[i]] is the
        i-th greatest value of sampled_heuristic_cost."""

        def heuristic_unitary_cost(value):
            num = -value*np.log(value)
            den = max(1/self.max_cardinality, value/self.max_rate)
            return num/den

        df = self.data.df[['p','plog1onp']].copy()
        df['cost'] = np.NaN
        for i in df.index:
            df['cost'][i] = heuristic_unitary_cost(df['p'][i])
        self.heuristic_cost = df['cost'].copy()
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
            index_series[i] = True
        self.data.df['heuristic'] = pd.Series(index_series,dtype='bool')
        self.selected_solution = 0        
        self.solved = True
        
    def decgraphH_solver(self):
        """Calculates solution using decision graph for V_{k,\mu,\nu} subproblems"""

        self.nnodes = 1
        df = self.data.df[['p','plog1onp']].copy()
        df.sort_index(by="p",ascending=True,inplace=True)
        idx = pd.Index([j for j in range(len(df))])
        self.decgraph_index_mapper = pd.Series(df.index)
        df.set_index(idx,inplace=True)
        
        #p = df["p"]
        #plog1onp = df["plog1onp"]
        #indexing is much faster on a numpy array than on a pandas dataframe:
        p = np.array(df["p"])
        plog1onp = np.array(df["plog1onp"])

        self.alpha = {}
        self.predecessor = {}
        self.decgraph_best_value = -1
        self.decgraph_best_value_node = None
        self.decgraph_root = (-1,0,0)
        self.cur_nodes_dict = {}
        self.next_nodes_dict = {}
        self.cur_nodes_dict[self.decgraph_root] = 0
        self.decgraph_len_visitlist = [1]
        
        reverse_cumulative_plog1onp = np.zeros(self.tot_patterns)
        reverse_cumulative_plog1onp[self.tot_patterns - 1] = plog1onp[self.tot_patterns - 1]
        for i in np.arange(self.tot_patterns - 2, -1, -1):
            reverse_cumulative_plog1onp[i] = reverse_cumulative_plog1onp[i+1] + plog1onp[i]

        def add_child(parent, arc_type, alpha_cur):
            "Looks at child and if feasible adds it to next_visitlist"
            
            k,mu,nu = parent
            if arc_type == 1:
                child = (k+1,mu,nu)
                candidate_new_entropy = alpha_cur
            else:
                child = (k+1, mu+p[k+1], nu+1)
                candidate_new_entropy = alpha_cur + plog1onp[k+1]
            add_child = True
            equivalent_paths = False
            if child in self.next_nodes_dict.keys():
                if candidate_new_entropy < self.next_nodes_dict[child]:
                    add_child = False
                elif candidate_new_entropy == self.next_nodes_dict[child]:
                    equivalent_paths = True
            if add_child == 1:
                if equivalent_paths == True:
                    if arc_type == 1:
                        self.predecessor[child] = [parent] + self.predecessor[child]
                    elif arc_type == 2:
                        self.predecessor[child] = self.predecessor[child] + [parent]
                else:
                    self.predecessor[child] = [parent]
                    self.next_nodes_dict[child] = candidate_new_entropy
                if self.next_nodes_dict[child] > self.decgraph_best_value:
                    self.decgraph_best_value = self.next_nodes_dict[child]
                    self.decgraph_best_value_node = child

        #main loop
        while self.cur_nodes_dict.keys():
            cur, alpha = self.cur_nodes_dict.popitem()
            k,mu,nu = cur
            if k+1 < self.tot_patterns and\
               alpha + reverse_cumulative_plog1onp[k] >= self.decgraph_best_value and\
               mu + p[k+1] <= self.max_rate and nu + 1 <= self.max_cardinality:
                add_child(cur, 1, alpha)
                add_child(cur, 2, alpha)
            if not self.cur_nodes_dict.keys() and k < self.tot_patterns -1:
                l = len(self.next_nodes_dict.keys())
                self.decgraph_len_visitlist.append(l)
                self.nnodes += l
                self.cur_nodes_dict = self.next_nodes_dict
                self.next_nodes_dict = {}
                gc.collect()

        self.selected_solution = 0        
        if self.decgraph_best_value_node == None:
            self.solved = False
            self.solution = [None]
        else:
            self.solved = True
            self.solutions_indexes_list = self.paths_list(self.decgraph_best_value_node)
            self.solution_cardinality = 0
            self.solution_rate = 0
            self.solution_entropy = 0
            for idx in self.solutions_indexes_list[0]:
                self.solution_cardinality += 1
                self.solution_rate += self.data.df['p'][idx]
                self.solution_entropy += self.data.df['plog1onp'][idx]

            if self.multiple_solutions == None:
                idx = self.solutions_indexes_list[0]
                self.solution = self.data.df.ix[idx]
                index_series = np.zeros(self.tot_patterns)
                for j in idx:
                    index_series[j] = 1
                self.data.df['decgraphH'] = pd.Series(index_series,dtype='bool')
            else:
                i = 0
                self.solution = []
                for sol in self.solutions_indexes_list:
                    self.solution.append(self.data.df.ix[sol])
                    index_series = np.zeros(self.tot_patterns)
                    for j in sol:
                        index_series[j] = 1
                    self.data.df['decgraphH' + str(i)] = pd.Series(index_series,dtype='bool')
                    i += 1

    def decgraphW_solver(self):
        """Calculates solution using decision graph for W_{k,v,\nu} subproblems"""

        self.nnodes = 1
        if self.use_quantized_entropy:
            df = self.data.df[['p','quantized_plog1onp']].copy()
        else:
            df = self.data.df[['p','plog1onp']].copy()
        df.sort_index(by="p",ascending=True,inplace=True)
        idx = pd.Index([j for j in range(len(df))])
        self.decgraph_index_mapper = pd.Series(df.index)
        df.set_index(idx,inplace=True)
        #indexing is much faster on a numpy array than on a pandas dataframe:
        p = np.array(df["p"])
        if self.use_quantized_entropy:
            plog1onp = np.array(df["quantized_plog1onp"])
        else:
            plog1onp = np.array(df["plog1onp"])
        self.alpha = {}
        self.predecessor = {}
        self.decgraph_best_value = -1
        self.decgraph_best_value_node = None
        self.decgraph_root = (-1,0,0)
        self.cur_nodes_dict = {}
        self.next_nodes_dict = {}
        self.cur_nodes_dict[self.decgraph_root] = 0

        self.decgraph_len_visitlist = [1]
        
        reverse_cumulative_plog1onp = np.zeros(self.tot_patterns)
        reverse_cumulative_plog1onp[self.tot_patterns - 1] = plog1onp[self.tot_patterns - 1]
        for i in np.arange(self.tot_patterns - 2, -1, -1):
            reverse_cumulative_plog1onp[i] = reverse_cumulative_plog1onp[i+1] + plog1onp[i]

        def add_child(parent, arc_type, alpha_cur):
            "Looks at child and if feasible adds it to next_visitlist"
            
            k,v,nu = parent
            if arc_type == 1:
                child = (k+1,v,nu)
                candidate_new_rate = alpha_cur
            else:
                child = (k+1, v+plog1onp[k+1], nu+1)
                candidate_new_rate = alpha_cur + p[k+1]
            add_child = True
            equivalent_paths = False
            if child in self.next_nodes_dict.keys():
                if candidate_new_rate > self.next_nodes_dict[child]:
                    add_child = False
                elif candidate_new_rate == self.next_nodes_dict[child]:
                    equivalent_paths = True
            if add_child == 1:
                if equivalent_paths == True:
                    if arc_type == 1:
                        self.predecessor[child] = [parent] + self.predecessor[child]
                    elif arc_type == 2:
                        self.predecessor[child] = self.predecessor[child] + [parent]
                else:
                    self.predecessor[child] = [parent]
                    self.next_nodes_dict[child] = candidate_new_rate
                if child[1] > self.decgraph_best_value:
                    self.decgraph_best_value = child[1]
                    self.decgraph_best_value_node = child
            
        #main loop
        while self.cur_nodes_dict.keys():
            cur,alpha = self.cur_nodes_dict.popitem()
            k,v,nu = cur
            if k+1 < self.tot_patterns and\
               v + reverse_cumulative_plog1onp[k] >= self.decgraph_best_value and\
               alpha + p[k+1] <= self.max_rate and nu + 1 <= self.max_cardinality:
                add_child(cur, 1, alpha)
                add_child(cur, 2, alpha)
            if not self.cur_nodes_dict.keys() and k < self.tot_patterns -1:
                l = len(self.next_nodes_dict.keys())
                self.nnodes += l
                self.decgraph_len_visitlist.append(l)
                self.cur_nodes_dict = self.next_nodes_dict
                self.next_nodes_dict = {}
                gc.collect()

        self.selected_solution = 0        

        if self.decgraph_best_value_node == None:
            self.solved = False
            self.solution = [None]
        else:
            self.solved = True
            self.solutions_indexes_list = self.paths_list(self.decgraph_best_value_node)
            self.solution_cardinality = 0
            self.solution_rate = 0
            self.solution_entropy = 0
            for idx in self.solutions_indexes_list[0]:
                self.solution_cardinality += 1
                self.solution_rate += self.data.df['p'][idx]
                self.solution_entropy += self.data.df['plog1onp'][idx]


            if self.multiple_solutions == None:
                idx = self.solutions_indexes_list[0]
                self.solution = self.data.df.ix[idx]
                index_series = np.zeros(self.tot_patterns)
                for j in idx:
                    index_series[j] = 1
                self.data.df['decgraphW'] = pd.Series(index_series,dtype='bool')
            else:
                i = 0
                self.solution = []
                for sol in self.solutions_indexes_list:
                    self.solution.append(self.data.df.ix[sol])
                    index_series = np.zeros(self.tot_patterns)
                    for j in sol:
                        index_series[j] = 1
                    self.data.df['decgraphW' + str(i)] = pd.Series(index_series,dtype='bool')
                    i += 1

  
