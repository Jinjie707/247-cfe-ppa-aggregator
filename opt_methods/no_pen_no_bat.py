from pyomo.environ import *
import pyomo.opt as pyo
import numpy as np

from .optimization_case import OptimizationCase

################################
###########  Case 0  ###########
################################

class NoPenNoBat(OptimizationCase):
    """
    Optimization Case 0: Base case, no bat no pen

    Attributes:
        Attributed initialized from parent class.
    """
    def __init__(self, processed_inputs, ppa_params):
        super().__init__(processed_inputs, ppa_params)

    def construct(self):
        """
        Constructs the Pyomo model for the optimization problem.

        Defines:
        - Decision variables
        - Objective function
        - Constraints
        """
        [M, mu, lam, epsilon] = self.model_constants
        p_N = self.p_N
        d_N = self.d_N # Assume we can aggregate demands
  
        T = self.T
        lcoe = self.lcoe
        productions = self.productions
        demands =  self.demands
        
        perc = self.perc
   
        ## Previously used for daily/monthly matching, not used for now
        # range_dt = self.range_dt
        # range_t = self.range_t
    
        # Create ConcreteModel
        model = ConcreteModel(name="PPA Aggregator")
        
        ######### DECISION VARIABLES ###########
        model.beta = Var(p_N, T, within=NonNegativeReals, bounds=(0, 1), initialize=0) # 2d variable, ratio of production used in PPA provision / total production at t
        model.v_total = Var(T, within=NonNegativeReals, bounds = (0, M), initialize = 0) # total = RE production
  
        ######### OBJECTIVE FUNCTION ###########
        model.cost = Objective(
            expr = sum(sum(lcoe[v] * productions[v][t] * model.beta[v, t] for v in p_N) for t in T) # Total Cost of RE production used
            + sum(sum(model.beta[v, t] for v in p_N) for t in T) / 100, # Regularization term to minimize total number of RE assets
            sense = minimize, # Minimization Problem
        )
        
        ######### CONSTRAINTS ###########
        # Rule 1 is the governing constraint, total supply > total demands
        # v_total is the governing decision variable, suggesting total energy supplied from the PPA contract
        def rule_1(model):
            return (sum(model.v_total[t] for t in T) >= sum(demands[t] for t in T) * perc)
        model.cst1 = Constraint(rule=rule_1)

        # Rule 2 and Rule 3: upper bounds of total energy provision
        #   2. Bounded by total energy from RE
        #   3. Bounded by total Consumption
        def rule_2(model, t):
            return (model.v_total[t] <= demands[t])
        model.cst2 = Constraint(T, rule = rule_2)
        
        def rule_3(model, t):
            return (model.v_total[t] <= sum(model.beta[v, t] * productions[v][t] for v in p_N))
        model.cst3 = Constraint(T, rule=rule_3)
      
        self.model = model
    
    
    def solve(self, optimizer):
        """
        Solves the constructed Pyomo model using the specified solver.

        Args:
            optimizer (str): Solver name (e.g., "cbc", "glpk", "gurobi").

        Returns:
            dict: Processed result dictionary if solution is feasible.
        """
        self.construct()
        model = self.model

        ###### CALL TO SOLVER ######
        opt = pyo.SolverFactory(optimizer)
        #    opt.options['timelimit'] = 1000
        results = opt.solve(model, tee=True)  # solve and show the steps (recommended)
        model.solutions.store_to(results)  # load the solution into results object

        # Infeasible conditions                
        if results.solver.termination_condition in [
            TerminationCondition.infeasible, 
            TerminationCondition.unbounded,
            TerminationCondition.noSolution,
            TerminationCondition.other,
            TerminationCondition.error
            ]:
            self.opt_result = "Infeasible"
            return 0
        else:
            print(results.solver.status)
            T = self.T
            N = self.p_N
            
            # Access the result variables
            beta = [[value(model.beta[v, t]) for t in T] for v in N] 
            v_total = [value(model.v_total[t]) for t in T]
            
            output = [beta, v_total]
            
            self.opt_result = output

        return self.process_result()
    
    def process_result(self):
        """
        Processes raw optimization outputs into a dictionary.

        Returns:
            dict: Dictionary containing optimal price, beta, energy usage, and utilization metrics.
        """
        beta = self.opt_result[0]
        v_total = self.opt_result[1]
        lcoe = self.lcoe
        productions = self.productions
        demands = self.demands
        perc = self.perc

        ###########
        ## Costs
        ###########
        renewable_cost = np.tile([0.0], 8760)
        for i in self.p_N:
            renewable_cost += lcoe[i] * np.multiply(np.array(beta[i]), np.array(productions[i]))
        renewable_cost = np.sum(renewable_cost) 

        # Placeholders
        market_revenue = 0
        market_purchase_cost = 0
        market_sell = []
        
        optimal_price = (renewable_cost + market_purchase_cost - market_revenue) / np.sum(np.array(demands)*perc)
        
        ###########
        ## Others
        ###########
        # This re_util_ratio here is deprecated (not used), but kept to avoid too much change 
        v_renewable = np.array(v_total)
        renewable_utilization_ratio = [(v_renewable[i] / productions[i]) for i in range(len(productions))]

        output_dict = {
            'optimal price': optimal_price,
            'beta': beta,
            'v_total': v_total,
            'v_renewable': v_renewable,
            'v_buy':  None,
            'renewable utilization ratio': renewable_utilization_ratio,
            'market sell': (market_revenue, market_sell)
        }
        
        return output_dict