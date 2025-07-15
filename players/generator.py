import numpy as np

class Generator:
    """
    A class to represent a generator, including the name, type, 
    LCOE and production profile.

    Attributes:
        name (str): The name of the generator.
        type (str): The type/category of the customer (e.g., Solar, Wind).
        profile (np.ndarray): An array representing the generator's hourly energy generation profile in MWh.
    """
    def __init__(self, name, gen_type, profile, lcoe):
        self.name = name
        self.type = gen_type
        self.profile = np.array(profile, dtype = 'float') # Type cast to 1d nparray
        self.lcoe = lcoe

    ### Some getter functions to access attributes ###
    
    def get_gen_name(self):
        return self.name
    
    def get_gen_type(self):
        return self.type
        
    def get_gen_profile(self):
        return self.profile
    
    def get_gen_lcoe(self):
        return self.lcoe
    
    ### Setter function to set new gen profile with typecast
    def set_new_gen_profile(self, profile):
        self.profile = np.array(profile)
