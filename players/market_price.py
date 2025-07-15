import numpy as np

class MarketPrice:
    """
    A class to represent market price used for penalty term.
    Including name, market price time series, penalty cap and penalty factor

    Attributes:
        name (str): The name of the market price.
        profile (np.ndarray): Market price time series.
        cap (int): The upper limit of penalty price.
        factor (int): The multiplication factor of penalty price.
    """
    def __init__(self, name, profile, cap, factor):
        self.name = name
        self.profile = profile
        self.cap = cap
        self.factor = factor
        
    ### Some getter functions to access attributes ###
    
    def get_mkt_name(self):
        return self.name

    def get_mkt_profile(self):
        return np.array(self.profile)
    
    def get_mkt_cap(self):
        return self.cap
    
    def get_mkt_factor(self):
        return self.factor
    
    # Setter to set cap and factor
    def set_factor_cap(self, factor, cap):
        self.cap = cap
        self.factor = factor
        
    