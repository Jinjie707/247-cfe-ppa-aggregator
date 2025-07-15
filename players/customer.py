import numpy as np

class Customer:
    """
    A class to represent a customer, including the name, type and consumption profile.

    Attributes:
        name (str): The name of the customer.
        type (str): The type/category of the customer (e.g., commercial, industrial).
        profile (np.ndarray): An array representing the customer's hourly energy demand profile in MWh.
    """
    def __init__(self, name, cus_type, profile):
        self.name = name
        self.type = cus_type
        self.profile = np.array(profile, dtype = 'float') # Type cast to 1d nparray
        
    ### Some getter functions to access attributes ###
    
    def get_cus_name(self):
        return self.name
    
    def get_cus_type(self):
        return self.type
        
    def get_cus_profile(self):
        return self.profile
   