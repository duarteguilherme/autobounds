from .canonicalModel import canonicalModel
from .Q import Query, Q, sub_list
from .Program import Program
from .DAG import DAG
from .Parser import Parser
import numpy as np
import pandas as pd
from copy import deepcopy
from itertools import product
from functools import reduce
from scipy.optimize import newton
import scipy
from numpy import log
import statsmodels.stats.proportion
import statsmodels.api as sm
from tqdm import tqdm

# Shared helper functions currently live in causalProblem.py
from .causalProblem import (
    generate_posterior_beta,
    generate_mn_sample,
    solve_gaussian,
    solve_kl_p,
    get_dirichlet_sample,
    simplify_first_nodes,
    replace_first_nodes,
    transform_constraint,
    get_constraint_from_row,
    get_query_data_do,
    get_summary_from_raw,
)

class Bounder:
    def __init__(self, dag, number_values = {}):
        """
        Causal problem has to have three elements: 
            a) canonicalModel: a canonical model;
            b) estimand to be optimized over;
            c.1) data;
            c.2) other constraints;
        sense is not mandatory anymore, but a function optimize that users can choose sense and optimizer.
        A parser needs to be included to translate expressions to canonicalModel language
        unconf_roots corresponds to roots for which we have data 
        """
        self.canModel = canonicalModel()
        self.dag = dag
        self.canModel.from_dag(self.dag, number_values)
        self.Parser = Parser(dag, number_values)
        self.number_values = { }
        for i in dag.V:
            self.number_values[i] = number_values[i] if i in number_values.keys() else 2
        self.parameters = [ (1, x) for x in self.canModel.parameters ]
        # self.parameters is exactly the same as self.canModel.parameters
        # the difference is self.parameters will keep track if parameters will not be used
        # this will be used to remove parameters that are not used in the final polynomial program
        self.estimand = None
        self.covariates = None
        self.constraints = [ ]
        self.unconf_first_nodes = [ ]
        self.samples = None
        self.safe_min = 0.0001 # This is a safe minimum to avoid division by zero
        self.maxtime, self.theta = None, 0.01  # There is no maximum time a program will be running
    def read_data(self, raw = None, covariates = None, inference = False, cond = [ ],
                  categorical = True, model = None, nsamples = 1000):
        """ This is the new method for loading data in place of 
        self.load_data, which will be outdated as a low version

        The idea is that load_data is not immediately executed, 
        but it is only evaluated at the time of writing program

        Notice that read_data only accepts raw data

        * cond must be a list of  variables that are used to condition the data
        For instance, if we have a dataset with X and Y, and we want to condition on X,
        we can introduce cond = ['X'] and the data will be conditioned on X.
        This options is useful when there is selection
        """
        self.categorical = categorical
        self.covariates = covariates
        self.inference = inference
        self.data_cond = cond
        if raw is not None:
            data = raw
            datam = deepcopy(data) if isinstance(data, pd.DataFrame) else pd.read_csv(data)
        else:
            raise Exception("Data was not introduced!")
        self.datam = datam
        if covariates is None:
            # If covariates do not exist, but there is no inference, just run the standard bounds and return
            self.covariates_data = pd.DataFrame({'X': [int(1)], 'prob_x': [1]})
            self.y_columns = [ i for i in self.datam.columns if i not in cond ]
            self.y = self.datam.drop_duplicates()[self.y_columns].astype(str).agg("_".join, axis=1)
            self.y, category_mapping = pd.factorize(self.y)
            self.category_decoder = dict(enumerate(category_mapping))
            if not inference: 
                self.load_data(raw = datam, cond = cond) 
                return None 
            else: # if covariates do not exist, but there is inference
                # If we want to do inference, then load_data is not executed immediately, but it will wait 
                # until it is evaluated by solve(). This of course is a workaround that will be removed in the future
                return None
        else: # If covariates exist, they become X
            if len(cond) > 0:
                raise Exception("Conditional data is not supported in read_data method if covariates are introduced. Please remove cond argument.")
            self.covariates_data = (get_summary_from_raw(self.datam[self.covariates])
                    .rename({'prob': 'prob_x'}, axis = 1))
            self.X = datam[covariates].to_numpy().reshape((-1, len(covariates)))
            self.X = sm.add_constant(self.X)
        self.covariates = covariates
        # load no-covariate data ( y )
        self.y_columns = [ k for k in datam.columns if k not in covariates ]
        self.y = datam.drop(columns = covariates).astype(str).agg("_".join, axis=1)
        self.y, category_mapping = pd.factorize(self.y)
        self.category_decoder = dict(enumerate(category_mapping))
        if not self.categorical: # If categorical is False, then we run a regression
            if model is None:
                model = sm.MNLogit(self.y, self.X) # Run multinomial logistic model -- in the future, this will allow for other models
                self.main_model = model.fit()
            else:
                self.main_model = model

    def calc_bounds_sample(self, prob, cond = [], verbose = False, limits   = [None, None]):
        """
        This method exists to solve the bounds problem
        for not hardcoded causalProblem

        This will require a copy of self
        """
        newproblem = deepcopy(self)
#        if self.covariates is not None:
#            datam = pd.DataFrame([ k.split('_') for k in newproblem.category_decoder.values() ], 
#                                            columns = newproblem.y_columns)
#        else:
#            datam = get_summary_from_raw(self.datam)
        datam = deepcopy(self.backbone_dataset)
        datam['prob'] = prob 
        if len(cond) > 0:
            datam = deepcopy(self.input_data)
            datam['prob'] = prob
        newproblem.load_data(datam, cond = cond)
        newprogram = newproblem.write_program()
        bounds = newprogram.run_scip(verbose = verbose, limits = limits, maxtime=self.maxtime, theta = self.theta)
        try:
            return (bounds[0]['dual'], bounds[1]['dual'])
        except:
            return (np.nan, np.nan)

    def generate_samples(self, n = 1000, randomize = True):
        """
        Generate samples from the posterior distribution of the coefficients
        of the main model.

        Parameters:
        - n: Number of samples to generate (default = 1000)
        - randomize: If True, randomizes the coefficients (default = True)
        """
        all_data = self.datam.value_counts().reset_index()
        all_data.rename(columns={all_data.columns[-1]: 'count'}, inplace=True)
        all_values = {col: np.arange(self.number_values[col]) 
                      for col in self.y_columns
                      if col not in self.data_cond
                      } # This is restricted for the case where only one conditional value exists
                       #  if there is more than one, each must be introduced separately 
        self.backbone_dataset = pd.DataFrame(list(product(*all_values.values())), columns=all_values.keys())
        self.samples = np.full((self.covariates_data.shape[0], n, self.backbone_dataset.shape[0]), np.nan)
        self.nsamples = n
        # Generate samples for each row in covariates_data
        # The dimensions of self.samples is 
        # (number of covariates, n, number of backbone dataset rows (prob))
        print("Generating samples:")
        for index, row in self.covariates_data.iterrows():
            if self.covariates_data.shape[0] > 1:
                print(f'\n{index + 1} of {self.covariates_data.shape[0]}')
            for j in range(n):                    
                self.samples[index, j, :] = (
                        get_dirichlet_sample(
                            self.backbone_dataset, all_data, row, self.covariates)
                )
            print('')
        
    def calculate_ci(self, nx = 1000, randomize = True, debug = False, 
                     verbose_optimizer = False, limits = [None, None]):
        """
        Calculate confidence intervals for the causal estimand.

        Parameters:
        - nx: Number of samples to generate for the X matrix (default = 1000)
        - categorical: If True, uses categorical data (default = False)
        """
        if self.samples is None:
            raise Exception("Samples have not been generated yet. Please call generate_samples() first.")
        nsamples = self.nsamples
        if self.categorical:
            self.lb_samples = np.full((self.covariates_data.shape[0], nsamples), np.nan)
            self.ub_samples = np.full((self.covariates_data.shape[0], nsamples), np.nan)
            print('Estimating CI: ')
            for index, row in self.covariates_data.iterrows():
#                print(index)
                for j in tqdm(range(nsamples)):
                    self.lb_samples[index, j], self.ub_samples[index, j] = self.calc_bounds_sample(
                            self.samples[index, j, :].reshape(-1), cond = self.data_cond, verbose = verbose_optimizer,
                            limits = limits
                        )
                    self.lb_samples[index, j] *= row['prob_x'] 
                    self.ub_samples[index, j] *= row['prob_x']
            return (self.lb_samples.sum(axis = 0), self.ub_samples.sum(axis = 0))
        else:
            if self.X.shape[0] > nx:
                newX =  self.X[
                    np.random.choice(self.X.shape[0], size = nx, replace = True), :]
            else:
                newX = self.X.copy()
            self.betas = np.array([ generate_posterior_beta(self.main_model, randomize) for i in range(nsamples) ])
            self.probs = np.array([ 
                [ generate_mn_sample(b, x)
                for b in self.betas ]
                for x in newX 
                ])
            self.lb_samples = np.full(self.probs.shape[0:2], np.nan)
            self.ub_samples = np.full(self.probs.shape[0:2], np.nan)
            for nx in range(self.probs.shape[0]):
                for nb in range(nsamples):
                    self.lb_samples[nx,nb], self.ub_samples[nx,nb] = (
                        (
                            self.calc_bounds_sample(self.probs[nx,nb],  verbose = verbose_optimizer,
                                                    limits = limits))
                    )
            return (self.lb_samples.mean(axis = 1), self.ub_samples.mean(axis = 1))

    def is_active(self, expr = '', ind = '', dep = ''):
        """ Call Parser.is_active()
        
        This is not just a wrapper -- it also returns a list of parameters into a query where each happens one time
        """
        params = [ Query(i) for i in self.Parser.is_active(expr, ind, dep) ]
        return reduce(lambda a,b : a + b, params)

    def solve(self, ci = False, nsamples = 10, maxtime = None, theta = 0.01, 
              verbose_optimizer = False, verbose_result = True, limits = [None, None]):
        """ Wrapper for causalProblem.write_program().solve()
        """
        print("Solving for point estimate bounds...")
        if maxtime is not None:
            self.maxtime = maxtime
        if self.estimand is None:
            raise ValueError("Estimand is not set. Please set an estimand using set_estimand() method.")
        if self.covariates is None:
            newproblem = deepcopy(self)
            try:
                self.input_data = self.datam if 'prob' in self.datam.columns else get_summary_from_raw(self.datam)
                newproblem.load_data(self.input_data, cond = self.data_cond )
            except:
                pass
            point_bounds = newproblem.write_program().run_scip(maxtime = maxtime, theta = theta, 
                                                               verbose = verbose_optimizer,
                                                                limits = limits)
            try:
                self.point_lb_dual = point_bounds[0]['dual']
                self.point_ub_dual = point_bounds[1]['dual']
                self.point_lb_primal = point_bounds[0]['primal']
                self.point_ub_primal = point_bounds[1]['primal']
            except:
                self.point_lb_dual, self.point_ub_dual = np.nan, np.nan
                self.point_lb_primal, self.point_ub_primal = np.nan, np.nan
        else:
            self.point_lb_dual = 0
            self.point_ub_dual = 0
            self.point_lb_primal = 0
            self.point_ub_primal = 0
            for index, row in self.covariates_data.iterrows():
                newproblem = deepcopy(self)
                newproblem.load_data(
                    # We load data from all the values where the covariate iteration
                    # is equal to the current row covariates
                    get_summary_from_raw(
                        self.datam.loc[
                                    self.datam[self.covariates]
                                    .eq(row[self.covariates].values).all(axis=1)
                                ].drop(self.covariates, axis = 1)
                                         ), 
                    cond = self.data_cond
                )
                point_bounds = newproblem.write_program().run_scip(maxtime = self.maxtime, theta = self.theta, 
                                                                   verbose = verbose_optimizer,
                                                                     limits = limits)
                try:
                    self.point_lb_dual += point_bounds[0]['dual'] * row['prob_x'] 
                    self.point_ub_dual += point_bounds[1]['dual'] * row['prob_x'] 
                    self.point_lb_primal += point_bounds[0]['primal'] * row['prob_x'] 
                    self.point_ub_primal += point_bounds[1]['primal'] * row['prob_x'] 
                except:
                    self.point_lb_dual, self.point_ub_dual = np.nan, np.nan
                    self.point_lb_primal, self.point_ub_primal = np.nan, np.nan
        if verbose_result:
            print(f"Point estimates\n")
            print(f"Dual: [{self.point_lb_dual}, {self.point_ub_dual}]")
            print(f"Primal: [{self.point_lb_primal}, {self.point_ub_primal}]")
        if not ci:
            return {
                "point lb dual": self.point_lb_dual,
                "point ub dual": self.point_ub_dual,
                "point lb primal": self.point_lb_primal,
                "point ub primal": self.point_ub_primal
            }
        if ci:
            if not self.inference:
                raise Exception("Confidence intervals can only be calculated if inference is True in read_data()")
            self.generate_samples(n = nsamples)
            self.ci_lb_bounds, self.ci_ub_bounds = self.calculate_ci(verbose_optimizer = verbose_optimizer, limits = limits)
            lb25 = np.quantile(self.ci_lb_bounds, 0.025)
            ub975 = np.quantile(self.ci_ub_bounds, 0.975)
            lb1 = np.quantile(self.ci_lb_bounds, 0.01)
            ub99 = np.quantile(self.ci_ub_bounds, 0.99)
            if verbose_result:
                print(f"95% Confidence intervals. Lower: {lb25},  Upper: {ub975}")
            return {
                "point lb dual": self.point_lb_dual,
                "point ub dual": self.point_ub_dual,
                "point lb primal": self.point_lb_primal,
                "point ub primal": self.point_ub_primal,
                "2.5% lb bounds": lb25,
                "97.5% ub bounds": ub975,
                "1% lb bounds": lb1,
                "99% ub bounds": ub99
            }

    def p(self, event, cond = None, sign = 1):
        """ 
        Wrapper for Parser.p
        """
        return self.Parser.p(event, cond, sign)

    def E(self, event, cond = None):
        """ Wrapper to calculate expected values 
        """
        event = event.strip()
        # Example: E(event = "Y(A=0)")
        main_var = event.split('(')[0]  # "Y" 
        second_part  = event.split(')')  # splits "Y(A=0)" -> "Y(A=0" and "",  "Y(A=0)=1" -> "Y(A=0" and "=1"
        if ',' in main_var:
            raise Exception('Issue: more than one variable introduced')
        if '-' in event:
            raise Exception('.E does not accept - terms. Construct expectations separately and then take the difference.')
        if len(second_part) > 1:
            if '=' in second_part[-1]:
                raise Exception('.E does not accept = terms. Did you mean to use .p?')
            elif len(second_part[-1]) > 0:
                raise Exception('Unexpected input in .E')
        for i in range(self.number_values[main_var]):
            if i == 0:
                continue
            try:
                res = res + Q(i) * self.p(event + '=' + str(i))
            except:
                #  If this is the first evaluation, uses cond
                # Then for the remaining evaluations (inside try)
                # cond will be multiplied automatically (cond is None, or 1)
                res = self.p(event + '=' + str(i), cond) #
        return res

    
    def set_ate(self, ind, dep, cond = None):
        """ Recipe for declaring ATEs"""
        query = self.p(f'{dep}({ind}=1)=1', cond = cond) - self.p(f'{dep}({ind}=0)=1', cond = cond) 
        self.set_estimand(query)
    
    def write_program(self):
        """ It returns an object Program
        """
        program = Program()
        self.check_constraints()
        program.parameters = [ x[1] 
                for x in self.parameters 
                if x[0] == 1 ] + [ 'objvar']
        zero_parameters = [ x[1] 
                for x in self.parameters 
                if x[0] == 0 ] 
        program.constraints = [
                transform_constraint(x, zero_parameters )
                for x in self.constraints
                ]
        program.optimize_remove_numeric_lines()
        return program
        
    def add_parameter(self, param_name):
        self.parameters += [(1, param_name)] 
    
    def check_constraints(self):
        """ 
        Check all constraints 
        and replace values for unconf_first_nodes
        """
        self.constraints = [ [ replace_first_nodes(self.unconf_first_nodes, y) 
            for y in x ]  
                for x in self.constraints ] 
    
    def add_prob_constraints(self):
        """
        This method is a default method to say that all the strata
        in one c-component has to sum to 1 (Kolmogorov)
        """
        # unconf_nodes is definitely important, because it 
        # handles the simplification everytime the first ancestrals
        # are not confounded (they are divided)
        unconf_nodes = [ x[0] for x in self.unconf_first_nodes ] 
        not_0_parameters = [ x[1] for x in self.parameters if x[0] != 0 ]
        for c in self.Parser.c_parameters:
            # Iterative over c_components
            prob_constraints = [ (1, [ x ]) 
                        for x in c
                if x in not_0_parameters 
                and x not in unconf_nodes ] 
            if len(prob_constraints) > 0:
                prob_constraints += [ (-1.0, ['1'])]
                self.add_constraint(Q(prob_constraints))
    
    def load_data_gaussian(self, data, N = 0, alpha = 0.05, cond = [ ], optimize = True, data_name = 'qp'):
        """ It accepts a file 
        """
        if N == 0:
            raise Exception("N cannot be 0!")
        datam = data if isinstance(data, pd.DataFrame) else pd.read_csv(data) 
        cond_data = datam[cond] if len(cond) > 0 else [ ]
        columns = [ x for x in datam.columns if x in list(self.dag.V) ]  + ['prob']
        datam = datam[columns]
        column_rest = [x for x in columns if x!= 'prob']
        grouped_data = datam.groupby(column_rest).sum()['prob'].reset_index()
        index, k, constraint = solve_gaussian(N, grouped_data['prob'], alpha, index = data_name)
        for i, row in grouped_data.iterrows():
            print(index + '_' + str(i))
            self.add_parameter(index + '_' + str(i))
            self.add_constraint(
                    get_constraint_from_row(row[column_rest], 
                                            index + '_' + str(i),
                                            self, 
                                            cond_data, 
                                            i))
        sum_qs = Query(-1)
        for i in range(k):
            sum_qs = sum_qs + Query(index + '_' + str(i)) 
        self.add_constraint(sum_qs)
        self.add_constraint(Query(constraint), "<=")
        if optimize:
            simplify_first_nodes(self, self.dag, datam, cond)
    
    def load_data_kl(self, data, N = 0, alpha = 0.05, cond = [ ], optimize = True):
        """ It accepts a file 
        """
        if N == 0:
            raise Exception("N cannot be 0!")
        datam = data if isinstance(data, pd.DataFrame) else pd.read_csv(data) 
        cond_data = datam[cond] if len(cond) > 0 else [ ]
        columns = [ x for x in datam.columns if x in list(self.dag.V) ]  + ['prob']
        datam = datam[columns]
        column_rest = [x for x in columns if x!= 'prob']
        grouped_data = datam.groupby(column_rest).sum()['prob'].reset_index()
        K = grouped_data.shape[0]
        for i, row in grouped_data.iterrows():
            min_max_kl = solve_kl_p(ns = N, alpha = alpha, K = K,
                    o = row['prob'] )
            self.add_constraint(
                    get_constraint_from_row(row[column_rest], 
                                            min_max_kl[0],
                                            self, 
                                            cond_data, 
                                            i), ">=")
            self.add_constraint(
                    get_constraint_from_row(row[column_rest], 
                                            min_max_kl[1],
                                            self, 
                                            cond_data, 
                                            i), "<=")
        if optimize:
            simplify_first_nodes(self, self.dag, datam, cond)
    
    def load_data_do(self, datam, do = [ ], optimize = True):
        for i in datam.groupby(do)['prob'].sum().tolist():
            if i != 1:
                raise Exception('Probabilities do not sum up to 1')
        for i in datam.columns:
            if i != 'prob':
                if i not in list(self.dag.V):
                    raise Exception('Included columns that do not exist in the causal model')
        cols = [ i for i in datam.columns if i != 'prob' and i not in do ]
        for i, row in datam.iterrows():
            self.add_constraint(get_query_data_do(row, cols, do, self) -
                                Query(float(row['prob']))
                                )
    
    def load_data(self, summary = None, raw = None, cond = [ ], do = [ ] ,optimize = True, covariates = None):
        """ It accepts a file 
        file must be csv. Columns will be added if they match parameters...
        Column prob must indicate probability.
        For example,
        >    X,Y,prob,
        >    1,0,0.25,
        >    0,1,0.25,
        >    1,1,0.25,
        >    0,0,0.25
        Conditioned columns must be added as a list , for instance, cond = ['M','C']
        -------------------------------------------------------------------
        Method: 
        1) For each row of data, data is parsed and added as a constraint to the problem.
        2) If conditioned data is present, arrangement for that are prepared
        Extra: 
        This method also implements one simplifier (first nodes simplifier).
        If data regarding first nodes is complete, then numeric values are added directly.
        """
        if summary is not None:
            data = summary
            datam = data if isinstance(data, pd.DataFrame) else pd.read_csv(data) 
        else:
            if raw is not None:
                data = raw
                datam = data if isinstance(data, pd.DataFrame) else pd.read_csv(data)
                datam = get_summary_from_raw(datam) 
            else:
                raise Exception("Data was not introduced!")
        if len(do) >= 1:
            if len(cond) >= 1:
                raise Exception('Data with cond and do at the same are not implemented yet')
            self.load_data_do(datam, do = do, optimize = True)
            return None
        cond_data = datam[cond] if len(cond) > 0 else [ ]
        columns = [ x for x in datam.columns if x in list(self.dag.V) ]  + ['prob']
        datam = datam[columns]
        column_rest = [x for x in columns if x!= 'prob']
        grouped_data = datam.groupby(column_rest).sum()['prob'].reset_index()
        if len(cond) > 0:
            for i, row in cond_data.drop_duplicates().iterrows():
                self.add_constraint(self.p('&'.join(
                                     [ f'{k}={int(row[k])}'
                                     for k in cond_data.columns ] )),
                                     '>=', 
                                     self.safe_min)
        for i, row in grouped_data.iterrows():
            #  ISSUE: need to add constraints for numeric tolerance
            # For instance if P(Y=1,X=0|M=1),
            # then P(M=1) >= 0.0001 for numeric stability
            self.add_constraint(
                    get_constraint_from_row(row[column_rest], 
                                            row['prob'], 
                                            self, 
                                            cond_data, 
                                            i))
        if optimize:
            simplify_first_nodes(self, self.dag, datam, cond)
    
    def set_p_to_zero(self, parameter_list):
        """
        For a particular list  of parameters
        ['X0111', 'Z0'], set them to 0 (This has to be improved)

        This method is pretty useful for efficient programs
        because it allows to remove not only parameters, but also constraints
        """
        if isinstance(parameter_list, Q):
            parameter_list = [ k[1][0] for k in parameter_list._event ]
            self.parameters = [ (x[0], x[1])
                    for x in self.parameters  
                    if x[1] not in parameter_list ] + [ (0, x) 
                            for x in parameter_list ]
        elif isinstance(parameter_list, list):
            self.parameters = [ (x[0], x[1])
                    for x in self.parameters  
                    if x[1] not in parameter_list ] + [ (0, x) 
                            for x in parameter_list ]
        else:
            raise Exception('Type error - cannot set it to 0')

    def add_assumption(self, constraint, symbol = "==", constraint2 = None):
        if constraint2 is not None:
            if not isinstance(constraint2, Q): # Do type checking
                constraint2 = Q(constraint2) 
        self.add_constraint(constraint, symbol, constraint2)
    
    def add_constraint(self, constraint, symbol = '==', constraint2 = None):
        """
        Input: a Q statement. For example Q([(-1, ['X1111', 'Z1']), (2, ['X1111'])])
    
        Symbol argument indicates if constraint will be an equality 
        or inequality. The default parameter will be an equality
        """
        if not isinstance(constraint, Q):
            raise TypeError('Constraint must be a Q object')
        if constraint2 is not None:
            constraint -= constraint2 
        # After right-hand side is 0, then the denominator can be ignored
        self.constraints.append(constraint._event + [ (1, [ symbol ] )])
        if constraint._cond is not None:
            self.constraints.append(sub_list(constraint._cond, 
                                             [(1 * self.safe_min, ['1'])] 
                                             )  + [ (1, [ '>=' ] )] )
        # An alternative is to check all the parameters for cond, and make them >= 0.001, when setting up the problem
        # Maybe the multiplicative constraint is already the best solution however
    
    def set_estimand(self,estimand, div = None):
        """
        Input: an expression similar to a constraint
        This algorithm there will 
        add estimand as a constraint with a new variable 
        objvar that will be added as a parameter.
        If the estimand is conditioned, then this condition 
        is multiplied by objvar, according to the algebraic formula.
        P(Y|X) = P(Y,X)/P(X) = objvar, then P(Y,X) - P(X) * objvar = 0
        """
        self.estimand =  estimand
        self.add_prob_constraints()
        if div is None:
            div = Query(1)
        else:
            self.add_constraint(div - self.safe_min, ">=")
        self.add_constraint(self.estimand -  (Query('objvar') * div ))
    
    def check_indep(self, c):
        """
        In a certain c-component 'c',
        check for possible independencies among 
        response variables
        ------
        Input: c-component
        Output: independent response variable tuples
        """
        c = list(c)
        if len(c) < 3:
            return []
        res = []
        for i in range(len(c)-1):
            for j in range(i+1, len(c)):
                if len(
                        self.dag.find_parents_u(c[i]).intersection(
                        self.dag.find_parents_u(c[j])
                        )) == 0:
                    res.append({c[i],c[j]})
        return res
    
    def add_indep(self, var):
        """ 
        Input: Var
        This method will be called by add_indeps in order 
        to simplify code. 
        Independences for particular values will be added as constraints
        """
        keys = list(var.keys())
        cons1 = []
        for i in [0,1]:
            cons1.append(quicksum([ self.parameters[k]
                    for k in 
                    self.get_response_expr({keys[i]: var[keys[i]]}) ] ))
        cons2 = quicksum([ self.parameters[k]
                for k in 
                self.get_response_expr(var) ])
        self.program.addCons(cons1[0]*cons1[1] - cons2 == 0)
    
    def add_rest_indep(self, indep):
        indep = list(indep)
        elem_1 = 2**(1+self.cn_index[indep[0]])
        elem_2 = 2**(1+self.cn_index[indep[1]])
        for i in range(elem_1):
            for j in range(elem_2):
                self.add_indep({indep[0]: i, indep[1]: j})
    
    def add_indep_constraints(self):
        """ For each components, check independencies 
        among variables and add them as constraints
        to the model. 
        """
        indeps = []
        for c in self.c_comp:
            indeps = indeps + self.check_indep(c)
        for i in indeps:
            self.add_rest_indep(i)

    

