from .canonicalModel import canonicalModel
from .Q import Query, Q
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
import inspect
import statsmodels.api as sm




def generate_posterior_beta(result, randomize = True):
    """
    Given a result from a regression, 
    generate a posterior beta
    """
    coef_mean = result.params
    coef_cov = result.cov_params()
    if randomize:
        coef_sampled = np.random.multivariate_normal(coef_mean.flatten(), coef_cov).reshape(coef_mean.shape)
    else:
        coef_sampled = coef_mean.reshape(coef_mean.shape)
    return coef_sampled

def generate_mn_sample(coef_sampled, X):
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        logits_sampled = X @ coef_sampled  # Compute logits
        probs_sampled = np.exp(logits_sampled)
        probs_sampled /= (1 + probs_sampled.sum(axis=1, keepdims=True))  # Normalize to probabilities
        # Add probability for reference category (first category as baseline)
        probs_sampled = np.column_stack([1 - probs_sampled.sum(axis=1), probs_sampled])
        return probs_sampled.reshape(-1)


class respect_to:
    """ Class to be used as a context manager 
    to respect to a causal Problem 
    """
    def __init__(self, problem):
        self.problem = problem
        self.globals = inspect.currentframe().f_back.f_globals

    def __enter__(self):
        self.globals['p'] = self.problem.p
        self.globals['E'] = self.problem.E
        self.globals['add_assumption'] = self.problem.add_assumption
        self.globals['set_estimand'] = self.problem.set_estimand
        self.globals['set_ate'] = self.problem.set_ate
        self.globals['solve'] = self.problem.solve
        self.globals['load_data'] = self.problem.load_data
        self.globals['read_data'] = self.problem.read_data
        self.globals['generate_samples'] = self.problem.generate_samples
        self.globals['calculate_ci'] = self.problem.calculate_ci


    def __exit__(self, exc_type, exc_value, traceback):
        del self.globals['p']
        del self.globals['E']
        del self.globals['add_assumption']
        del self.globals['set_estimand']
        del self.globals['set_ate']
        del self.globals['solve']
        del self.globals['load_data']
        del self.globals['read_data']
        del self.globals['generate_samples']
        del self.globals['calculate_ci']

def get_summary_from_raw(datam):
    """
    Gets a data set and returns a summary
    """
    nrow = datam.shape[0]
    cols = list(datam.columns)
    datam = deepcopy(datam)
    datam['prob'] = 1/nrow
    return (
        datam.groupby(cols)
        .sum()
        .reset_index()
    )

def multiply_matrix_gaussian(q, mu, sigma_inv):
    if len(q) != len(mu):
        " Q and mu have different sizes"
    len_proc = len(q)
    q_minus_mu = [ mu[i] - q[i]      for i in range(len_proc) ] 
    sum_result = Query(0)
    for i in range(len_proc):
        for j in range(len_proc):
            sum_result += q_minus_mu[i] * sigma_inv[i,j] * q_minus_mu[j]
    return sum_result

def solve_gaussian(nr, o, alpha, index = 'qp'):
    """ alpha is the level of confidence...
    nr is the number of rows
    p is the population distribution we are trying to find
    K is the number of pieces of data
    obs is the observed data 
    """
    if index == 'qp':
        print("Make sure that this dataset is the first to be introduced. For other datasets, remember to introduce the argument data_name")
    query_vectorize = np.vectorize(lambda a: Query(a))
    mu = np.array([o[:-1]])
    params = [ Query(index + '_' + str(i)) for i,f in enumerate(o[:-1]) ]
    mu_diag = np.diag(np.array(o[:-1]))
    sigma = (mu_diag - np.matmul(mu.transpose(),mu)) / nr
    sigma_inv_query = query_vectorize(np.linalg.pinv(sigma))
    mu_query = [ Query(i) for i in o[:-1] ]
    lh_side = multiply_matrix_gaussian(params, mu_query, sigma_inv_query)
    k = len(o)
    rh_side = Query(scipy.stats.chi2.ppf( 1- alpha, k - 1))
    res = lh_side - rh_side
    return (index,k, res)


def solve_kl_p(ns, K, o, alpha):
    """ alpha is the level of confidence...
    ns is the number of sample
    p is the population distribution we are trying to find
    K is the number of pieces of data
    o is the observed data 
    """
    KL = lambda p: o * log(o / p) + (1 - o) * log((1 - o) / (1 - p)) 
    thresh = log(2 * K / alpha) / ns
    optim_func = lambda p: KL(p) - thresh
    if o == 0:
        return np.array(
                statsmodels
                .stats
                .proportion.proportion_confint(0, ns, alpha = alpha/K, method = 'agresti_coull'))
    elif o == 1:
        return np.array(
                statsmodels
                .stats
                .proportion.proportion_confint(ns, ns, alpha = alpha/K, method = 'agresti_coull'))
    else:
        res = newton(optim_func, [o/2, (1+o)/2])
    if np.isnan(res[0]):
        res[0] = 0
    if np.isnan(res[1]):
        res[1] = 1
    return res

def get_dirichlet_sample(backbone, all_data, row, covariates):
    """
    Generate Dirichlet samples based on the provided data.

    Args:
    - backbone: DataFrame containing the backbone data
    - all_data: DataFrame containing all data
    - row: Current row data
    - covariates: List of covariates to match the data
    - n: Number of times to calculate the Dirichlet samples (default = 1000)

    Returns:
    - dirichlet_samples: Dirichlet samples generated from the matched data
    """
    if covariates is None:
        prov = backbone.merge(all_data).fillna(0)
    else:
        prov = backbone.merge(
            all_data[
            (all_data[covariates].values == row[covariates].values).all(axis=1)
            ], how = 'left').fillna(0)
    counts = prov['count'].values + 1
    dirichlet_samples = np.random.dirichlet(counts)
    return dirichlet_samples

# Simplifiers 
### 1) First nodes
def simplify_first_nodes(problem, dag, datam, cond): 
    """ 
    Firstly, all first nodes are collected from dag.
    Secondly, if data is complete for those nodes,
    they must be set to zero.
    """
    if len(cond) > 0: # Simplifier 1 cannot handle conditional data
        return None
    data_count = datam.drop('prob', axis = 1).nunique()
    complete_data = [ i  for i, j in dict(data_count).items() 
            if problem.number_values[i] == j ]
    if any([k for k in data_count == 1]): # Important, if data has selection, for instance all elements of X are 1, then return None
        return None
    first_nodes = [ k for k in dag.find_first_nodes() 
            if len(dag.find_u_linked(k)) == 1 and k in complete_data ]
    # Need to check if data is complete
    for k in first_nodes:
        problem.unconf_first_nodes += [ (k + str(i), 
            datam.groupby(k).sum().loc[i]['prob'] )
                for i in datam.groupby(k).sum().index ]
    problem.set_p_to_zero([ x[0] for x in problem.unconf_first_nodes ])


def replace_first_nodes(first_nodes, constraint):
    """ 
    Gets an expr inside a constraint, for instance,
    (1, ['X00.Y00', 'Z0']) and if Z0 is in first nodes, 
    it replace Z0 by 1 and it multiplies 1 by the prob
    """
    coef, var = constraint[0], constraint[1]
    for i,v in enumerate(var):
        for n in first_nodes:
            if v == n[0]:
                var[i] = '1'
                coef *= n[1]
                break
    return ( coef, var )
            


def transform_constraint(constraint, zero_parameters = []):
    """ 
    To be used inside write_program method
    This functions gets a constraint in 
    causalProblem format and translate it to 
    program format
    """
    res =  [ ['' if k[0] == 1 else str(k[0]) ] + 
            [ i for i in k[1] if i != '1' ] 
        for k in constraint ] 
    res = [ [ j for j in i if j != '' ] for i in res  ]
    res = [ [ j for j in i ] for i in res 
            if not any([ j in zero_parameters for j in i ]) ]  # Check if there are zero parameters
    return res

def get_constraint_from_row(row_data, row_prob, program, cond = [ ], n = 0):
    """ 
    Function to be employed in load_data method in causalProgram
    One introduces the row data , row prob , Parser
    and function returns constraints 
    """
    row_cond = cond.iloc[n] if len(cond) > 0  else []
    query = [ f'{row_data.index[j]}={int(row_data.iloc[j])}'
                    for j,k in enumerate(list(row_data)) ]
    if len(row_cond) > 0:
        query_cond = [ f'{row_cond.index[j]}={int(row_cond[j])}'
                    for j,k in enumerate(list(row_cond)) ]
        return program.p('&'.join(query)) - Query(row_prob) * program.p('&'.join(query_cond))
    return   program.p('&'.join(query)) - Query(row_prob)

def get_query_data_do(row, cols, do, self):
    do_str = ','.join([
        f'{i}={int(row[i])}' 
        for i in do ])
    str_query = '&'.join([
        f'{i}({do_str})={int(row[i])}' 
        for i in cols ])
    return self.p(str_query)


class causalProblem:
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
        self.estimand = [ ]
        self.covariates = None
        self.constraints = [ ]
        self.unconf_first_nodes = [ ]
        self.samples = None
        
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
        if raw is not None:
            data = raw
            datam = deepcopy(data) if isinstance(data, pd.DataFrame) else pd.read_csv(data)
        else:
            raise Exception("Data was not introduced!")
        self.datam = datam
        if covariates is None:
            # If covariates do not exist, but there is no inference, just run the standard bounds and return
            self.covariates_data = pd.DataFrame({'X': [int(1)], 'prob_x': [1]})
            self.y_columns = list(self.datam.columns)
            self.y = self.datam.astype(str).agg("_".join, axis=1)
            self.y, category_mapping = pd.factorize(self.y)
            self.category_decoder = dict(enumerate(category_mapping))
            if not inference: 
                self.load_data(raw = datam, cond = cond) 
                return None 
            else: # if covariates do not exist, but there is inference
#                print(f"Generating {nsamples} samples for inference...")
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

    def calc_bounds_sample(self, prob):
        """
        This method exists to solve the bounds problem
        for not hardcoded causalProblem

        This will require a copy of self
        """
        newproblem = deepcopy(self)
        if self.covariates is not None:
            datam = pd.DataFrame([ k.split('_') for k in newproblem.category_decoder.values() ], 
                                            columns = newproblem.y_columns)
        else:
            datam = get_summary_from_raw(self.datam)
        datam['prob'] = prob 
        newproblem.load_data(datam)
        newprogram = newproblem.write_program()
        bounds = newprogram.run_scip()
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
        all_values = {col: np.arange(self.number_values[col]) for col in self.y_columns}
        backbone_dataset = pd.DataFrame(list(product(*all_values.values())), columns=all_values.keys())
        self.samples = np.full((self.covariates_data.shape[0], n, backbone_dataset.shape[0]), np.nan)
        self.nsamples = n
        # Generate samples for each row in covariates_data
        # The dimensions of self.samples is 
        # (number of covariates, n, number of backbone dataset rows (prob))
        print("Generating samples:")
        for index, row in self.covariates_data.iterrows():
            if self.covariates_data.shape[0] > 1:
                print(f'\n{index + 1} of {self.covariates_data.shape[0]}')
            for j in range(n):                    
                print(f'{j}', end = ',')
                self.samples[index, j, :] = (
                        get_dirichlet_sample(
                            backbone_dataset, all_data, row, self.covariates)
                )
            print('')
        
    def calculate_ci(self, nx = 1000, randomize = True, debug = False):
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
            self.lower_samples = np.full((self.covariates_data.shape[0], nsamples), np.nan)
            self.upper_samples = np.full((self.covariates_data.shape[0], nsamples), np.nan)
            for index, row in self.covariates_data.iterrows():
                print(index)
                for j in range(nsamples):   
                    self.lower_samples[index, j], self.upper_samples[index, j] = self.calc_bounds_sample(
                            self.samples[index, j, :].reshape(-1)
                        )
                    self.lower_samples[index, j] *= row['prob_x'] 
                    self.upper_samples[index, j] *= row['prob_x']
            self.lower_samples = self.lower_samples.sum(axis = 0)
            self.upper_samples = self.upper_samples.sum(axis = 0)
            if debug: # Debug samples
                return (self.lower_samples. self.upper_samples)
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
            self.lower_samples = np.full(self.probs.shape[0:2], np.nan)
            self.upper_samples = np.full(self.probs.shape[0:2], np.nan)
            for nx in range(self.probs.shape[0]):
                for nb in range(nsamples):
                    self.lower_samples[nx,nb], self.upper_samples[nx,nb] = (
                        (
                            self.calc_bounds_sample(self.probs[nx,nb]))
                    )
            if debug: # Debug samples
                return (self.lower_samples. self.upper_samples)
            self.lower_samples = self.lower_samples.mean(axis = 1)
            self.upper_samples = self.upper_samples.mean(axis = 1)
        return (self.lower_samples, self.upper_samples)    
        # I have to simulate X and then calculate the probabilities

    def is_active(self, expr = '', ind = '', dep = ''):
        """ Call Parser.is_active()
        
        This is not just a wrapper -- it also returns a list of parameters into a query where each happens one time
        """
        params = [ Query(i) for i in self.Parser.is_active(expr, ind, dep) ]
        return reduce(lambda a,b : a + b, params)

    def solve(self, ci = False, nsamples = 10, maxtime = None, theta = 0.01):
        """ Wrapper for causalProblem.write_program().solve()
        """
        print("Solving for point estimate bounds...")
        if self.covariates is None:
            newproblem = deepcopy(self)
            try:
                input_data = self.datam if 'prob' in self.datam.columns else get_summary_from_raw(self.datam)
                newproblem.load_data(input_data)
            except:
                pass
            point_bounds = newproblem.write_program().run_scip(maxtime = maxtime, theta = theta)
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
                                         )
                )
                point_bounds = newproblem.write_program().run_scip(maxtime = maxtime, theta = theta)    
                try:
                    self.point_lb_dual += point_bounds[0]['dual'] * row['prob_x'] 
                    self.point_ub_dual += point_bounds[1]['dual'] * row['prob_x'] 
                    self.point_lb_primal += point_bounds[0]['primal'] * row['prob_x'] 
                    self.point_ub_primal += point_bounds[1]['primal'] * row['prob_x'] 
                except:
                    self.point_lb_dual, self.point_ub_dual = np.nan, np.nan
                    self.point_lb_primal, self.point_ub_primal = np.nan, np.nan
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
            self.ci_lb_bounds, self.ci_ub_bounds = self.calculate_ci()
            print(self.ci_lb_bounds)
            print(self.ci_ub_bounds)
            self.ci_lb_bounds = np.quantile(self.ci_lb_bounds, 0.025)
            self.ci_ub_bounds = np.quantile(self.ci_ub_bounds, 0.975)
            print(f"95% Confidence intervals. Lower: {self.ci_lb_bounds},  Upper: {self.ci_ub_bounds}")
            return {
                "point lb dual": self.point_lb_dual,
                "point ub dual": self.point_ub_dual,
                "point lb primal": self.point_lb_primal,
                "point ub primal": self.point_ub_primal,
                "2.75% lb bounds": self.ci_lb_bounds,
                "9.75% ub bounds": self.ci_ub_bounds,
                "1% lb bounds": np.quantile(self.ci_lb_bounds, 0.01),
                "99% ub bounds": np.quantile(self.ci_ub_bounds, 0.99)
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
    
    def add_constraint(self, constraint, symbol = '==', constraint2 = None, control = 0.0001):
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
                                             [(1 * control, ['1'])] 
                                             )  + [ (1, [ '>=' ] )] )
        # An alternative is to check all the parameters for cond, and make them >= 0.001, when setting up the problem
        # Maybe the multiplicative constraint is already the best solution however
    
    def set_estimand(self,estimand, div = None, control = 0.0001):
        """
        Input: an expression similar to a constraint
        This algorithm there will 
        add estimand as a constraint with a new variable 
        objvar that will be added as a parameter.
        If the estimand is conditioned, then this condition 
        is multiplied by objvar, according to the algebraic formula.
        P(Y|X) = P(Y,X)/P(X) = objvar, then P(Y,X) - P(X) * objvar = 0

        control is a numeric parameter to avoid numeric problem, such as division by 0
        """
        self.add_prob_constraints()
        if div is None:
            div = Query(1)
        else:
            self.add_constraint(div - Query(control), ">=")
        self.add_constraint(estimand -  (Query('objvar') * div ))
    
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
    





