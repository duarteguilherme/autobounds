from .canonicalModel import canonicalModel
from .Q import Q, Query
from .Program import Program
from .DAG import DAG
from .Parser import Parser
from itertools import product
from collections import Counter
from numpy import array, asarray, ndarray, add, identity
from functools import reduce
from copy import deepcopy
import sympy
import numpy as np


def remove_common(a,b):
    """ Remove elements that repeat in two lists
    """
    a, b = a.copy(), b.copy()
    for i in a.copy():
        if i in b:
            a.remove(i)
            b.remove(i)
    return [a,b]


class Gset(ndarray):
    """ 
    This class will be implemented over numpy.vectors. 
    Each Sset is a vector over all strata of a c-component

    This will allow behavior such as inclusion and exclusion.
    In other words, it works like a multiset but with negative elements
    """
    def __new__(cls, input_array):
        return asarray(input_array).view(cls)
    def __eq__(self, other):
        return all(super().__eq__(other))    
    def __gt__(self, other):
        return all(super().__ge__(other))    and any(super().__gt__(other))
    def __lt__(self, other):
        return all(super().__le__(other)) and any(super().__lt__(other))    
    def __ge__(self, other):
        return all(super().__ge__(other))    
    def __le__(self, other):
        return all(super().__le__(other))    
    def __ne__(self, other):
        return any(super().__ne__(other))    
    def __truediv__(self, other):
        return any(super().__gt__(other)) and any(super().__lt__(other))
    def neq0(self):
        return super().__ne__(0)
    def is_empty(self):
        return not any(super().__ne__(0))
    def __and__(self, other):
        return any(ndarray.__and__(self.neq0(), other.neq0()) )
    def __add__(self, other):
        if len(other) == 0:
            if len(self) == 0:
                raise Exception('Cannot add two empty Gsets')
            return self
        if len(self) == 0:
            return other
        return Gset(super().__add__(other))
    def __sub__(self, other):
        if len(other) == 0:
            if len(self) == 0:
                raise Exception('Cannot add two empty Gsets')
            return self
        if len(self) == 0:
            return -1 * other
        return Gset(super().__sub__(other))



def sum_cover(cover: list[Gset]) -> Gset:
    if len(cover) == 0:
        return Gset([])
    return reduce(lambda a, b = None: a + b, cover)

def is_inconclusive(x1: Counter, x2: Counter) -> bool: # If there is no order between x1 and x2 (we are dealing with partial orders)
    return not (x1 >= x2 or x1 <= x2)


def compare_covers(x1: Counter, x2: Counter) -> int:
    """ Compare two covers of parameters 
    - return 1, if x1 is greater than x2
    - return 2, if x2 is  greater than x1
    - return 0, if that can be determined
    
    p1 and p2 are multisets, it will be represented as collections.Counter
    """
    t1 = greater_or_equal_to(x1, x2)
    t2 = less_or_equal_to(x1, x2)
    if t1 and not t2:
        return 1
    elif t2 and not t1:
        return 2
    else:
        return 0


def get_cover_addition(eset, subsets, ub = []):
    """ This function gets the cover of a Gset (multiset), 
    using additions. It adds all Gsets, until it
    find the minimum cover 
    
    Upper bound: O(n) = 2^n

    Input: 
        - eset -- Gset to be covered
        - subsets -- the Gset that will be considered as covers
        - ub -- best bounds so far -- empty if it is not
    Output:
        - a set of min-covers. As we are talking about partial order, 
        there might be more than one min-cover.
    """
    if eset.is_empty():
        raise Exception("Estimand to be covered cannot be empty")
    subsets = Gset(subsets)
    subset_index = array([j for j, k in enumerate(subsets) 
                          if (k & eset) ]) # Keep the relevants - no intersection must be removed
    covers = [ [  ] ] # covers will be id'ed by indexes. First index points to the empty element
    output = ub.copy()
    for i in subset_index:
        for j in range(len(covers)):
            newcover = covers[j] + [ i ] # Index for a new cover
            subsetcover = sum_cover(subsets[newcover])
            if subsetcover >= eset: # if eset is covered by the addition of s
                # Now we have to check all the already computed covers (output)
                # if the subsetcover (newcover) is strictly less than any cover, keep subsetcover
                # if the subsetcover is greater or than any cover, keep the old cover  
                # if all of them are inconclusive with respect to the new cover, just add the new cover at the end
                for k in range(len(output)):
                    if subsetcover >= ( kcover :=  sum_cover(subsets[output[k]])):
                        break
                    if subsetcover < kcover:
                        output[k] = newcover
                        break
                else: # If for does not break or len(output) is 0, 
                      # that means all are inconclusive, so we just add newcover to the end
                    output.append(newcover)
            else:
                covers.append(newcover)
    return output


def get_cover_subtraction(eset, subsets, ub):
    """ This function gets the cover of a Gset, 
    using subtractions. It removes Gsubsets, until it
    finds the minimum cover 

    Here the idea that only Gsubsets will be considered (only those will be relevant)
    And Gsubsets will be remove until the minimum cover
    
    Input: 
        - eset -- multiset to be covered
        - subsets -- the multisets that will be considered as covers
        - ub -- the upper bound multiset where the subtraction will start
    Output:
        - a set of min-covers. As we are talking about partial order, 
        there might be more than one min-cover.
    """
    if eset == Counter():
        raise Exception("Set to be covered cannot be empty")
    og_output = ub.copy() # og_output and output are index
    if len(og_output) == 0:
        return og_output
    subsets = Gset(subsets)
    output = [ ] # Pieces to be removed from each part of og_output
    subset_index = array([j for j, k in enumerate(subsets) 
                     if any([k <= sum_cover(subsets[i]) 
                             for i in og_output])  ]) # Keep the relevants - all those subsets that are included in the output
    for m in range(len(og_output)):
        output.append([])
        pieces = [ [  ] ] # pieces of subsets will be id'ed by indexes. pieces to be removed from the output
        for i in subset_index:
            # It only subtracts one, it must subtracts all of them
            for j in range(len(pieces)):
                newpiece = pieces[j]+ [ i ] # Index for a new piece to test
                p1 = sum_cover(subsets[og_output[m]]) 
                p2 = sum_cover(subsets[newpiece])
                subsetcover = p1 - p2
                if not ( p2 < p1 ):
                    break
                if subsetcover >= eset: # if eset is covered by the subtraction of new piece
                    # Now we have to check all the already computed covers (output)
                    # if the subsetcover is strictly less than any cover, keep subsetcover
                    # if the subsetcover is greater or equal than any cover, keep the old cover  
                    # if all of them are inconclusive with respect to the new cover, just add the new cover at the end
                    for k in range(len(output[m])):
                        if subsetcover >= ( kcover :=   sum_cover(subsets[og_output[m]]) - sum_cover(subsets[output[m][k]])):
                            break
                        if subsetcover < kcover:
                            output[m][k] = newpiece
                            break
                    else: # If for does not break or len(output) is 0, 
                        # that means all are inconclusive, so we just add newcover to the end
                        output[m].append(newpiece)
            else:
                pieces.append([i])
    if len(og_output) != len(output):
        raise Exception('Output and og_output must have the same length')
    # The final step is to check the og_output - output, and check minima and maxima
    # 1. Expanding output to match og_output
    final_output = [ ]
    for i in range(len(og_output)):
        if len(output[i]) > 0:
            for j in range(len(output[i])):
                final_output.append( [og_output[i], output[i][j]])
        else:
            final_output.append( [og_output[i], []])    
    if len(final_output) == 1:
        return(final_output)
    # 2. Now we check if there are minima and maxima in the list that can be excluded
    # For instance if the bounds in final_output[0] are less than the bounds in final_output[1], then we can exclude the bounds in final_output[1] 
    final_output_index = [ 0 ] # Can never be empty 
    for i in range(1, len(final_output)):      
        left1 = sum_cover(subsets[final_output[i][0]])
        right1 = sum_cover(subsets[k]) if len(k := final_output[i][1]) > 0 else None
        list1 = left1 if right1 is None else left1 - right1 
        for m in range(len(final_output_index)):
            j = final_output_index[m]
            left2 = sum_cover(subsets[final_output[j][0]])
            right2 = sum_cover(subsets[k]) if len(k := final_output[j][1]) > 0 else None
            list2 = left2 if right2 is None else left2 - right2 
            if list1 >= list2:
                break
            elif list1 < list2:
                final_output_index.remove(j)
                final_output_index.append(i)
                break
        else:
            final_output_index.append(i)
    final_output = [ final_output[i] for i in final_output_index ] # indexing
    return final_output


def get_subtraction_bounds(ub, lb, sub_ub, sub_lb, eset, subsets):
    print(ub)
    print(lb)
    print(sub_ub)
    print(sub_lb)
    input("")



class closedProblem:
    def __init__(self, dag, number_values = {}):
        """
        """
        self.canModel = canonicalModel()
        self.dag = dag
        self.canModel.from_dag(self.dag, number_values)
        self.Parser = Parser(dag, number_values)
        self.parameters = [ (1, x) for x in self.canModel.parameters ]
        self.estimand = [ ]
        self.data = { }
        self.c_comp = self.dag.find_c_components()
        self.top_order = self.dag.get_top_order()
        self.c_parameters = [ Counter(i)
                             for i in self.Parser.c_parameters ] 
        self.ub = [None for i in range(len(self.c_parameters))]
        self.lb = [None for i in range(len(self.c_parameters))]

    def query(self, expr, gset = True) -> tuple[Counter]:
        # Must have only one path. In other words, datasets such as P(X=1) will not work, as 
        # there will be more than one path
        """ 
        Important function:
        This function is exactly like parse in Parser class.
        However, here it returns a constraint structure.
        So one can do causalProgram.query('Y(X=1)=1') in order 
        to get P(Y(X=1)=1) constraint.
        """
        query = self.Parser.parse(expr, complete = True)
        if len(query) > 1:
            raise Exception('More than one path for each query is not allowed')
        return tuple( Counter(k)   for path in query for k in path )
    
    def find_solutions(self, read = True):
        """ Use solve_covering for each c_component 

        One implementation with multisets with negative elements will be implemented later
        Notice that this algorithm can get sharpness by inclusion-exclusion
        """
        pass 
        if len(self.estimand) == 0:
            raise Exception('Estimand is empty!')
        if len(self.data) == 0:
            raise Exception('Data is empty!')
        for index, part in enumerate(self.estimand): #  No need to duplicate -- solve_covering solves for estimand and its symmetric
            self.solve_covering(index)
        if read:
            self.read_solution()

    def solve_covering2(self, c_n: int, subtract: bool = True) -> str:
        """ Implements the complete covering algorithm

        """
        pass

    def solve_covering(self, c_n: int, subtract: bool = True) -> str:
        """ Implements the greedy covering algorithm

        Input: An index for a c_component (Integer)

        Arg: subtract indicates if the subtraction extension will be used

        Step 1: Eliminate all parts that sum to 1

        Step 2: If estimand is a sum, but only one c-component is not complete, consider that estimand as only one sum
        Otherwise, raise Exception


        Step 3: Find cover through sums, then through subtractions


        """
        # Step 1
        estimand = self.estimand[c_n].copy()
        sym_estimand = self.sym_estimand[c_n].copy()
        for k in self.c_parameters: # c_parameters include all the strata in a c-component
            if not bool(k - estimand): # if the part of the estimand is equal to the whole c-component
                self.ub[c_n] = 1
                self.lb[c_n] = 1 # Notice that it will work for sym_estimand as well
                # i.e. if the c-component part is 1 for the estimand, it will be 1 for sym_estimand as well
                return None
        # Step 2 -- prepare data
        data = { index: self.to_gset(value[c_n], self.c_parameters[c_n]) 
                                for index, value in self.data.items()  
                        }
        data_input = Gset(list(data.values()))
        # Transform estimand to gset too
        estimand = self.to_gset(estimand, self.c_parameters[c_n])
        sym_estimand = self.to_gset(sym_estimand, self.c_parameters[c_n])
        # Step 3
        # Two lists for control and control_keys rather than only one dictionary for loop goals
        # Step 4, use get_cover_addition and get_cover_subtraction
        # You have to transform them first to Gsets
        output_add = get_cover_addition(estimand, data_input)
        sym_output_add = get_cover_addition(sym_estimand, data_input)
        self.ub[c_n] = get_cover_subtraction(estimand, data_input, output_add) if subtract else output_add
        self.lb[c_n] = get_cover_subtraction(sym_estimand, data_input, sym_output_add) if subtract else sym_output_add

    def read_solution(self, sum_ub = '', sum_lb = '1 - '):
        data = list(self.data.keys())
        print(' \n Upper bounds: ')
        for sol in self.ub:
            if sol is None:
                raise Exception('Solution is empty. Run the algorithm first to get a solution')
            if sol == 1:
                continue
            else:
                for index, i in enumerate(sol):
                    print(str(index + 1), end = '. ' + sum_ub)
                    end_sub =  ' - ' if len(i[1]) > 0  else '' 
                    print(' + '.join([ data[j] for j in i[0] ]), end = end_sub)
                    print(' - '.join([ data[j] for j in i[1] ]), end = '\n')
        print(' \n Lower bounds: ')
        for sol in self.lb:
            if sol is None:
                raise Exception('Solution is empty. Run the algorithm first to get a solution')
            if sol == 1:
                continue
            else:
                for index, i in enumerate(sol):
                    print(str(index + 1), end = '. ' + sum_lb)
                    end_sub =  ' + ' if len(i[1]) > 0  else '' 
                    print(' - '.join([ data[j] for j in i[0] ]), end = end_sub)
                    print(' + '.join([ data[j] for j in i[1] ]), end = '\n')

    def read_solution_subtraction(self):
        pass
                    
    def find_bounds_subtraction(self, subestimand, solve_main = True):
        """ Find solution for a subtracted estimand 

        For cases like the ATE, where part of estimand is subtracted

        The solution will be found by using the same algorithm as before

        -- It only works for one c-component at a time 
        """
        subproblem = deepcopy(self)
        subproblem.set_estimand(subestimand)
        if solve_main:
            self.find_solutions(read = False)
        subproblem.find_solutions(read = False) 
        ub = self.ub
        lb = self.lb
        subub = subproblem.ub
        sublb = subproblem.lb
        # The upper bound will be the upper bound of the positive estimand minus the lower bound of the negative estimand
        # The lower bound will be the lower bound of the positive estimand minus the upper bound of the negative estimand
        target_c = [ i for i, l in enumerate(ub) if isinstance(l, list) ]
        if len(target_c) != 1:
            raise Exception('Current implementation only works for one c-component')
        target_c = target_c[0]
        data = { index: self.to_gset(value[target_c], self.c_parameters[target_c]) 
                                for index, value in self.data.items()  
                        }
        data_input = Gset(list(data.values()))
        res_ub, res_lb =  [ ], [ ]
        for i in ub[target_c]:
            for j in sublb[target_c]:
                p1 =  sum_cover(data_input[i[0]])
                p2 =  sum_cover(data_input[i[1]]) 
                p3 =  sum_cover(data_input[j[0]])
                p4 =  sum_cover(data_input[j[1]]) 
                p5 = Gset(np.ones(len(data_input[0])))
                prov = p1 - p2 - p5 + p3 - p4  # LOWER BOUNDS ARE p5 (1) - p3 + p4, so ub - lb , this turns to - p5 + p3 - p4
                for k in res_ub.copy():
                    if k[1] <= prov:
                        break
                    if k[1] > prov:
                        res_ub.remove(k)
                else:
                    res_ub.append([ [i[0] + j[0], i[1] + j[1] ],  prov] )
        for i in lb[target_c]:
            for j in subub[target_c]:
                p1 =  sum_cover(data_input[i[0]])
                p2 =  sum_cover(data_input[i[1]]) 
                p3 =  sum_cover(data_input[j[0]])
                p4 =  sum_cover(data_input[j[1]]) 
                # Notice that the lower bound starts with one -- that's the design here
                p5 = Gset(np.ones(len(data_input[0])))
                prov = p5 - p1 + p2 - p3 + p4
                for k in res_lb.copy():
                    if k[1] >= prov:
                        break
                    if k[1] < prov:
                        res_lb.remove(k)
                else:
                    res_lb.append([ [i[0] + j[0], i[1] + j[1] ],  prov] )
        self.ub[target_c] = [ remove_common(*i[0]) for i in res_ub ]
        self.lb[target_c] = [ remove_common(*i[0]) for i in res_lb ]
        self.read_solution(sum_ub = "- 1 + ", sum_lb = "1 - ")


    def to_gset(self, query: tuple[Counter], c_parameters: list[Counter]) -> Gset:
        """ Transform a query into a Gset
        """
        query = array([ query[i] for i in c_parameters ]  )
        return Gset(query)
    
    def get_symmetric(self, query:  tuple[Counter]):
        """ Get the symmetric of a query
        """
        return tuple( k - q if q != (k := self.c_parameters[index])  else k 
                     for index, q in enumerate(query) 
                       )
    
    def set_p_to_zero(self, query: tuple[Counter]):
        """ Set elements of a query to 0"""
        self.c_parameters = self.get_symmetric(query)

    def set_estimand(self, estimand):
        """ Load estimand,
        and also its symmetric quantity
        """
        self.estimand = estimand
        self.sym_estimand = self.get_symmetric(estimand)

    def add_sens_parameter(self, parameter_name, query):
        '''
        Add Sensitivity analysis parameters
        '''
        self.data = {**self.data, 
                    **{ parameter_name: query } }

    def load_data(self, data, do = None):
        # data comes in the format X,Y,Z

        # Notice that data has to be introduced separately for each c-component

        # This method assumes the user will introduce the data for each c-component separately
        # It will be fixed later
        data = data.split(',')
        data_input = [  '&'.join(
                                [ f'{k[0]}={k[1]}' for k in zip(data, i) ]
                            ) for i in product(*[[0,1]]*len(data))  ] 
        if do is not None:
            data_input = [ d.replace('=', f'({do})=') for d in data_input ]
        self.data = {**self.data, 
                    **{ i: self.query(i) for i in data_input } }
