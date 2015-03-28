import numpy as np
import pdb

def solution_cardinality(solution):
    sum_xi = 0
    for i in solution:
        sum_xi += 1
    return sum_xi

def solution_entropy(p,solution):
    sum_entropy = 0
    for i in solution:
        sum_entropy += p[i]*np.log(1/p[i])
    return sum_entropy

def solution_rate(p,solution):
    sum_pi = 0
    for i in solution:
        sum_pi += p[i]
    return sum_pi

def check_solution(M,n,W,p,solution):
    cardinality = solution_cardinality(solution)
    rate = solution_rate(p,solution)
    #entropy = solution_entropy(p,solution)
    
    #if sum_entropy != M1.objective:
    #    print("ERROR, M1.objective = %f but variables add up to %f" %(M1.objective,sum_entropy))
    #    return(1)
    if cardinality > n:
        print("ERROR, %d variables were one (n = %d)" % (k, n))
        return(1)
    if rate > W:
        print("ERROR, p[i] sum up to %f > %f" % (rate, W))
        return(1)
    else:
        return(0)
