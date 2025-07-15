from abc import ABC, abstractmethod

class OptimizationCase(ABC):
    """
    Abstract base class for defining different optimization problem cases.
    
    This class sets up the common input structure and data used across all optimization cases,
    such as generators, customers, market prices, battery data, and matching terms.
    
    Concrete child classes should implement the `construct()` and `solve()` methods to define
    and solve specific optimization scenarios.
    
    Attributes:
        model_constants (list): A list of constant parameters [M, mu, lambda, epsilon] used in optimization.
        perc (float): The target 24/7 CFE percentage (e.g. 1.0 for 100%).
        time_step (int): Matching resolution (e.g. 1 for hourly, 24 for daily).
        maturity (int): PPA contract duration in years.
        generators (Generators): Generators object from input.
        customers (Customers): Customers object from input.
        market_prices (MarketPrice): MarketPrice data object, if applicable
        existing_bat (Storage): Existing Storage object, if applicable.
        lcoe (list): List of levelized cost of energy for each generator.
        productions (list of np.ndarray): List of generator generation profiles.
        demands (np.ndarray): Aggregate consumption profile from all customers.
        d_N (range): Index range for customers.
        p_N (range): Index range for generators.
        T (range): Time steps of the optimization horizon.
        T_minus_1 (range): Time steps minus one (used in battery constraints).
        range_dt (range): Time step range within matching granularity (e.g. 0–23 for daily if hourly steps).
        range_t (range): Time step range aligned to matching resolution.
        model (ConcreteModel): Pyomo model object to be defined in child classes. Initialized to None.
        opt_result (dict): Dictionary to store the optimization results. Initialized to None.
    """
    def __init__(self, processed_inputs, ppa_params):
        self.model_constants = [10e6, 10000, 10, 10e-3] # [M, mu, lam, epsilon]
        
        self.perc, self.time_step, self.maturity = ppa_params.get_ppa_terms()
        
        self.generators = processed_inputs['Generators']
        self.customers = processed_inputs['Customers']
        self.market_prices = processed_inputs['Market Prices'][1] # Empty ([]) if not the case
        self.existing_bat = processed_inputs['Existing Battery'][1] # Empty ([]) if not the case
        
        self.lcoe = self.generators.get_all_gen_lcoes()
        self.productions = self.generators.get_all_gen_profiles()
        self.demands = self.customers.get_agg_cus_profile()
        
        self.d_N = range(self.customers.get_cus_numbers())
        self.p_N = range(self.generators.get_gen_numbers())
        self.T = range(self.customers.get_cus_profile_len())
        self.T_minus_1 = range(self.customers.get_cus_profile_len() - 1)
        self.range_dt = range(0, self.time_step, 1)
        self.range_t = range(0, self.customers.get_cus_profile_len(), self.time_step)
        
        self.model = None
        self.opt_result = None
        
    @abstractmethod
    def construct(self):
        pass
    
    @abstractmethod
    def solve(self):
        pass
