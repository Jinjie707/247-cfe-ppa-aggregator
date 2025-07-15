from pyomo.environ import *
import pyomo.opt as pyo
import numpy as np

from .optimization_case import OptimizationCase

################################
###########  Case 8  ###########
################################

class PenNewBat(OptimizationCase):
    """
    Optimization Case 8: OPtimizing for New Battery with Pen.

    Attributes:
        Attributed initialized from parent class.
        Penalty and New Bat attributes to be initialized seperately to remind this case.
    """
    def __init__(self, processed_inputs, ppa_params):
        super().__init__(processed_inputs, ppa_params)
        self.processed_inputs = processed_inputs
        self.existing_bat = ppa_params.get_new_bat_params()
        
        self.penalty_prices = self.market_prices.get_mkt_profile()
        self.penalty_factor = self.market_prices.get_mkt_factor()
        self.penalty_cap = self.market_prices.get_mkt_cap()
        
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
        T_minus_1 = self.T_minus_1
        lcoe = self.lcoe
        productions = self.productions
        demands =  self.demands
        perc = self.perc
        
        penalty_prices = self.penalty_prices
        penalty_factor = self.penalty_factor
        penalty_cap = self.penalty_cap
        
        ex_bat = self.existing_bat
        lcos = ex_bat.get_bat_lcos()
        bat_cap_limit = ex_bat.get_new_bat_max_cap()
        [bat_capacity, bat_rte, bat_charging_rate, bat_max_cycle, bat_n] = ex_bat.get_bat_params()
        
        ## Previously used for daily/monthly matching, not used for now
        # range_dt = self.range_dt
        # range_t = self.range_t
    
        # Create ConcreteModel
        model = ConcreteModel(name="PPA Aggregator")

        ######### DECISION VARIABLES ###########
        model.beta = Var(p_N, T, within=NonNegativeReals, bounds=(0, 1), initialize=0) # 2d variable, ratio of re production used in PPA provision / total production at t
        model.v_total = Var(T, within=NonNegativeReals, bounds = (0, M), initialize = 0) # total = buy + production + discharge
        model.v_used = Var(T, within=NonNegativeReals, bounds=(0, M), initialize={t: 0 for t in T}) # Renewable generation used in PPA provision 
        model.v_buy = Var(T, within=NonNegativeReals, bounds=(0, M)) # Market purchase at penalty
        
        model.u_c = Var(T, within=NonNegativeReals, bounds=(0, M), initialize=0) # Bat charge
        model.u_d = Var(T, within=NonNegativeReals, bounds=(0, M), initialize=0) # Bat discharge 
        model.soc = Var(T, within=NonNegativeReals, bounds = (0, M), initialize=0) # Bat SOC
        model.capacity = Var(within=NonNegativeReals, bounds = (0, M), initialize=0) # New Bat capacity

        ######### OBJECTIVE FUNCTION ###########
        model.cost = Objective(
            expr = sum(sum(lcoe[v] * productions[v][t] * model.beta[v, t] for v in p_N) for t in T) # Total Cost of RE production used
            + sum(sum(model.beta[v, t] for v in p_N) for t in T) / 100 # Regularization term to minimize total number of RE assets
            + sum(model.v_buy[t] * np.clip(np.array(penalty_prices)*penalty_factor, 0, penalty_cap)[t] for t in T) # Total cost of market purchase at penalty
            + model.capacity * lcos * bat_max_cycle *365, # Bat cost. ASSUMPTION: Bat cost is consider as fully charged every day
            sense = minimize, # Minimization Problem
        )
        
        ######### CONSTRAINTS ###########
        # Rule 1 is the governing constraint, total supply > total demands
        # v_total is the governing decision variable, suggesting total energy supplied from the PPA contract
        def rule_1(model):
            return (sum(model.v_total[t] for t in T) >= sum(demands[t] for t in T) * perc)
        model.cst1 = Constraint(rule=rule_1)

        # Rule 2 and Rule 3: upper bounds of total energy provision
        #   2. Bounded by total energy from RE + Bat discharge + Market Purchase
        #   3. Bounded by total Consumption
        def rule_2(model, t):
            return (model.v_total[t] <= model.v_used[t] + model.u_d[t]*bat_rte + model.v_buy[t])
        model.cst2 = Constraint(T, rule=rule_2)

        # Hourly provision, no carry-over
        def rule_3(model, t):
            return (model.v_total[t] <= demands[t])
        model.cst3 = Constraint(T, rule = rule_3)

        # Rule 4, 5, 6: Battery charging boundaries
        #   4. Charging quantity is bounded by (total RE production - RE prod used for PPA Provision)
        #       - Assumes no charging with market purchase
        #   5: Charging quantity is bounded by avaliable battery capacity
        #   6: Charging quantity is bounded by charging rate
        def rule_4(model, t):
            return (model.u_c[t] <= sum(model.beta[v,t]*productions[v][t] for v in p_N) - model.v_used[t])
        model.cst4 = Constraint(T, rule = rule_4)
        
        def rule_5(model, t):
            return (model.u_c[t] <= model.capacity - model.soc[t])
        model.cst5 = Constraint(T, rule = rule_5)
        
        def rule_6(model, t):
            return (model.u_c[t] <= bat_charging_rate)
        model.cst6 = Constraint(T, rule = rule_6)
        
        # Rule 7, 8, 9: Battery charging boundaries
        #   7. Discharging quantity is bounded by (demand - RE provision - Market Purchase)
        #   8: Discharging quantity is bounded by avaliable battery energy
        #   9: Discharging quantity is bounded by discharging rate
        def rule_7(model, t):
            return (model.u_d[t]*bat_rte <= demands[t] - model.v_used[t] - model.v_buy[t])
        model.cst7 = Constraint(T, rule = rule_7)
        
        def rule_8(model, t): 
            return (model.u_d[t] <= model.soc[t])
        model.cst8 = Constraint(T, rule = rule_8)
        
        def rule_9(model, t):
            return (model.u_d[t] <= bat_charging_rate)
        model.cst9 = Constraint(T, rule = rule_9)
        
        # Rule 10, 12, 13: SOC constraints (I have no idea where is rule 11 :D)
        
        # Rule 10: SOC constraint, update battery status at each time step
        def rule_10(model, t):
            return (model.soc[t+1] == model.soc[t] + model.u_c[t] - model.u_d[t])
        model.cst10 = Constraint(T_minus_1, rule = rule_10)
        
        def rule_12(model, t):
            return (model.soc[t] <= model.capacity)
        model.cst12 = Constraint(T, rule = rule_12)
        
        def rule_13(model):
            return(model.soc[0] == 0)

        # Rule 15: New Bat capacity is bounded by input limit
        def rule_15(model):
            return (model.capacity <= bat_cap_limit)
        model.cst15 = Constraint(rule = rule_15)

        # Rule 17: constraint to enforce battery charging cycle alignment; 
        # A relaxed constraint as it's enforced at yearly level rather than daily
        # TODO: It's there a computationally effective way to improve this?
        def rule_17(model):
            return (sum(model.u_c[t] for t in T) <= 365*model.capacity*bat_max_cycle)
        model.cst17 = Constraint(rule = rule_17)

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
            p_N = self.p_N
            ex_bat = self.existing_bat
            [bat_capacity, bat_rte, bat_charging_rate, bat_max_cycle, bat_n] = ex_bat.get_bat_params()
            
            # Access the result variables
            beta = [[value(model.beta[v, t]) for t in T] for v in p_N]
            v_total = [value(model.v_total[t]) for t in T]
            v_buy = [value(model.v_buy[t]) for t in T]
            
            u_c = [value(model.u_c[t]) for t in T]
            u_d = [value(model.u_d[t]) for t in T]
            bat_soc = [value(model.soc[t]) for t in T]
            bat_capacity = value(model.capacity)
            
            # New Bat processing
            bat_charging_profile = np.array(u_c) - np.array(u_d)
            
            bat_results = {'Capacity': bat_capacity,
                           'Charging profile': bat_charging_profile,
                           'SOC': bat_soc,
                           'u_c': u_c,
                           'u_d': u_d,
                           'n_hour': np.round(bat_capacity / bat_charging_rate, 2)
                           }
            
            self.existing_bat.set_opt_result(bat_results)
          
            # Stupidly setting back to the list
            self.processed_inputs['Existing Battery'] = (True, self.existing_bat)
    
            output_dict = {'beta': beta,
                            'v_total': v_total,
                            'v_buy': v_buy,
                            'bat_results': bat_results,
                            'capacity': bat_capacity}
            
            self.opt_result = output_dict

        return self.process_result()
    
    def process_result(self):
        """
        Processes raw optimization outputs into a dictionary.

        Returns:
            dict: Dictionary containing optimal price, beta, energy usage, and utilization metrics.
        """  
        beta = self.opt_result['beta']
        v_total = self.opt_result['v_total']
        v_buy = self.opt_result['v_buy']
        
        bat_results = self.opt_result['bat_results']
        bat_u_c = bat_results['u_c']
        
        lcoe = self.lcoe
        productions = self.productions
        demands = self.demands
        perc = self.perc
        
        penalty_prices = self.penalty_prices
        penalty_factor = self.penalty_factor
        penalty_cap = self.penalty_cap
        
        ex_bat = self.existing_bat
        lcos = ex_bat.get_bat_lcos()
        [bat_capacity, bat_rte, bat_charging_rate, bat_max_cycle, bat_n] = ex_bat.get_bat_params()
    
        ###########
        ## Costs
        ###########
        renewable_cost = np.tile([0.0], 8760)
        for i in self.p_N:
            renewable_cost += lcoe[i] * np.multiply(np.array(beta[i]), np.array(productions[i]))
        renewable_cost = np.sum(renewable_cost)
        
        bat_cost = bat_capacity * lcos * bat_max_cycle * 365
        
        scaled_penalty = penalty_prices * penalty_factor
        clipped_penalty = np.clip(scaled_penalty, 0, penalty_cap)
        market_purchase_cost = (np.array(clipped_penalty).flatten()) * (np.array(v_buy).flatten())
        market_purchase_cost = np.sum(market_purchase_cost)
        
        market_revenue = 0
        market_sell = []
        
        optimal_price = (renewable_cost + bat_cost + market_purchase_cost - market_revenue) / np.sum(np.array(demands)*perc)

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
            'optimal bat cap': bat_capacity
        }
        
        return output_dict