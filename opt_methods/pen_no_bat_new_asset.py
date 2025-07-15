from pyomo.environ import *
import pyomo.opt as pyo
import numpy as np

from .optimization_case import OptimizationCase

################################
###########  Case 7  ###########
################################

class PenNoBatNewAsset(OptimizationCase):
    """
    Optimization Case 7: OPtimizing for New Asset with Pen.

    Attributes:
        Attributed initialized from parent class.
        Penalty and New Gen attributes to be initialized seperately to remind this case.
    """
    def __init__(self, processed_inputs, ppa_params):
        super().__init__(processed_inputs, ppa_params)
        self.penalty_prices = self.market_prices.get_mkt_profile()
        self.penalty_factor = self.market_prices.get_mkt_factor()
        self.penalty_cap = self.market_prices.get_mkt_cap()
        
        self.new_gen = ppa_params.get_new_gen_params()
        self.new_gen_profile = self.new_gen.get_gen_profile()
        self.new_gen_lcoe = self.new_gen.get_gen_lcoe()
        self.new_gen_ppa_price = ppa_params.get_new_gen_ppa_price()
    
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
        
        new_prod = self.new_gen_profile
        new_lcoe = self.new_gen_lcoe
        
        perc = self.perc
        
        penalty_prices = self.penalty_prices
        penalty_factor = self.penalty_factor
        penalty_cap = self.penalty_cap
    
        ## Previously used for daily/monthly matching, not used for now
        # range_dt = self.range_dt
        # range_t = self.range_t
    
        # Create ConcreteModel
        model = ConcreteModel(name="PPA Aggregator")
        
        ######### DECISION VARIABLES ###########
        model.beta = Var(p_N, T, within=NonNegativeReals, bounds=(0, 1), initialize=0) # 2d variable, ratio of re production used in PPA provision / total production at t
        model.v_total = Var(T, within=NonNegativeReals, bounds = (0, M), initialize = 0) # total = buy + production
        model.new_gen_cap = Var(within=NonNegativeReals, bounds = (0, M), initialize = 0) # New gen cap
        model.v_buy = Var(T, within=NonNegativeReals, bounds=(0, M)) # Market purchase at penalty

        ######### OBJECTIVE FUNCTION ###########
        model.cost = Objective(
            expr = sum(sum(lcoe[v] * productions[v][t] * model.beta[v, t] for v in p_N) for t in T) # Total Cost of RE production used
            + sum(model.new_gen_cap * new_prod[t] * new_lcoe for t in T) # Cost of new gen. ASSUMPTION: fully used for the PPA
            + sum(model.v_buy[t] * np.clip(np.array(penalty_prices)*penalty_factor, 0, penalty_cap)[t] for t in T) # Total cost of market purchase at penalty
            + sum(sum(model.beta[v, t] for v in p_N) for t in T) / 100,  # Regularization term to minimize total number of RE assets
            sense = minimize, # Minimization Problem
        )
        
        ######### CONSTRAINTS ###########
        # Rule 1 is the governing constraint, total supply > total demands
        # v_total is the governing decision variable, suggesting total energy supplied from the PPA contract
        def rule_1(model):
            return (sum(model.v_total[t] for t in T) >= sum(demands[t] for t in T) * perc)
        model.cst1 = Constraint(rule=rule_1)
        
        # Rule 2 and Rule 3: upper bounds of total energy provision
        #   2. Bounded by total Consumption
        #   3. Bounded by total energy from RE + Market Purchase
        def rule_2(model, t):
            return (model.v_total[t] <= demands[t])
        model.cst2 = Constraint(T, rule = rule_2)
        
        def rule_3(model, t):
            return (model.v_total[t] <= sum(model.beta[v, t] * productions[v][t] for v in p_N) 
                    + model.new_gen_cap * new_prod[t]
                    + model.v_buy[t])
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
        productions = self.productions
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
            beta = [[value(model.beta[v, t]) for t in T] for v in N] # Result beta is a 2x8760 list
            v_total = [value(model.v_total[t]) for t in T]
            new_gen_cap = value(model.new_gen_cap)
            v_buy = [value(model.v_buy[t]) for t in T]
            
            # New gen processing
            new_beta = []
            new_prod = new_gen_cap * self.new_gen_profile
            
            # Compute total productions from existing assets
            temp_total = []
            for i in range(len(productions)):
                temp_total.append(beta[i] * productions[i])
            temp_total = np.sum(np.array(temp_total), axis = 0)
            
            # Deduce new gen production
            for i in range(8760):
                diff = v_total[i] - temp_total[i]
                if new_prod[i] == 0:
                    new_beta.append(0)
                else:
                    temp_new_beta = np.round((min(1, diff / new_prod[i])), 4)
                    new_beta.append(temp_new_beta)

            new_beta = np.nan_to_num(new_beta, nan=0)
            beta.append(new_beta)    
        
            output_dict = {'beta': beta,
                           'v_total': v_total,
                           'new_gen_cap': new_gen_cap,
                           'v_buy': v_buy
                            }

            self.opt_result = output_dict
            
            self.new_gen.set_new_gen_profile(new_gen_cap * self.new_gen_profile)

        return self.process_result()
    
    def process_result(self):
        """
        Processes raw optimization outputs into a dictionary.

        Returns:
            dict: Dictionary containing optimal price, beta, energy usage, and utilization metrics.
        """  
        beta = self.opt_result['beta']
        v_total = self.opt_result['v_total']
        new_gen_cap = self.opt_result['new_gen_cap']
        v_buy = self.opt_result['v_buy']
        
        new_gen_cap = self.opt_result['new_gen_cap']
        new_lcoe = self.new_gen_lcoe
        new_gen_ppa_price = self.new_gen_ppa_price
        new_gen_profile = self.new_gen.get_gen_profile()
        new_beta = beta[-1]
        
        self.generators.add_new_gen(self.new_gen)
        
        lcoe = self.lcoe + [new_lcoe]
        productions = self.productions 
        demands = self.demands
        perc = self.perc
        
        penalty_prices = self.penalty_prices
        penalty_factor = self.penalty_factor
        penalty_cap = self.penalty_cap
             
        ###########
        ## Costs
        ###########
        renewable_cost = np.tile([0.0], 8760)
        for i in (self.p_N):
            renewable_cost += lcoe[i] * np.multiply(np.array(beta[i]), np.array(productions[i]))
        renewable_cost = np.sum(renewable_cost) +  lcoe[-1] * np.sum(self.new_gen.get_gen_profile())
        
        scaled_penalty = penalty_prices * penalty_factor
        clipped_penalty = np.clip(scaled_penalty, 0, penalty_cap)
        market_purchase_cost = (np.array(clipped_penalty).flatten()) * (np.array(v_buy).flatten())
        market_purchase_cost = np.sum(market_purchase_cost)
        
        market_sell = np.multiply(np.array(1-new_beta), new_gen_profile)
        market_revenue = new_gen_ppa_price * np.sum(market_sell)
        
        optimal_price = (renewable_cost + market_purchase_cost - market_revenue) / np.sum(np.array(demands)*perc)

        ###########
        ## Others
        ###########
        # This re_util_ratio here is deprecated (not used), but kept to avoid too much change
        v_renewable = np.array(v_total) - np.array(v_buy) # Total renewable prodcution involved to meet the green demands
        renewable_utilization_ratio = [(v_renewable[i] / productions[i]) for i in range(len(productions))]

        output_dict = {
            'optimal price': optimal_price,
            'beta': beta,
            'v_total': v_total,
            'v_renewable': v_renewable,
            'v_buy':  v_buy,
            'renewable utilization ratio': renewable_utilization_ratio,
            'market sell': (market_revenue, market_sell),
            'optimal capacity': new_gen_cap
        }
        
        return output_dict