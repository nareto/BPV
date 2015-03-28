import numpy as np
import pdb
import pulp


class BPVsolver:
    def __init__(self,solver_type,tot_patterns,max_cardinality,max_rate,p):
        if len(p) != tot_patterns:
            err = "len(p) == tot_patterns must hold"
            raise RuntimeError(err)
        else:
            self.__is_solved__ = 0
            self.__all_solvers__ = {"exact": self.BPVexactsolver}#,"euristic","dynprog")
            self.__solver_type__ = None
            #print(self.__solver_type__)
            try:
                #pdb.set_trace()
                self.set_solver(solver_type)
            except:
                print("asdf")
                #pass
            self.tot_patterns = tot_patterns
            self.max_cardinality = max_cardinality
            self.max_rate = max_rate
            self.p = p


    def set_solver(self, solver_type):
        if solver_type not in self.__all_solvers__.keys():
            err = ("solver_type has to be one of"+" \"%s\""*len(solvers) % solvers.keys())
            raise RuntimeError(err)
        else:
            self.__solver_type__ = solver_type

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
                try:
                    self.__all_solvers__[self.solver()]()
                    self.__is_solved__ = 1
                except:
                    pass

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
    
    def BPVexactsolver(self):
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


