import numpy as np

class Storage:
    """
    A class to represent a battery storage.

    Attributes:
        name (str): Name of the battery.
        type (str): Type or label of the battery.
        bat_params (list): List of battery parameters in the format 
            [capacity (MWh), round-trip efficiency, charging rate, max cycles, duration (hours)].
        lcos (float): Levelized Cost of Storage in $/MWh.
        opt_result (dict): Dictionary to store the results of optimization.
        u_c (np.ndarray): Charging profile.
        u_d (np.ndarray): Discharging profile.
        soc (np.ndarray): State of charge over time.
        charging_profile (np.ndarray): Net charging profile (charge - discharge) over time.
        max_new_cap (float): Maximum capacity allowed for new battery.
    """
    def __init__(self, name, bat_type, bat_params, lcos):
        self.name = name
        self.type = bat_type
        self.bat_params = bat_params
        
        self.cap = bat_params[0]
        self.rte = bat_params[1]
        self.charging_rate = bat_params[2]
        self.max_cycle = bat_params[3]
        self.n_hour = bat_params[4]
        
        self.lcos = lcos
        self.opt_result = None
        self.u_c = None
        self.u_d = None
        self.soc = None
        self.charging_profile = None
        
        self.max_new_cap = None
        
    ### Some getter functions to access attributes ###
    
    def get_bat_lcos(self):
        return self.lcos

    def get_bat_cap(self):
        return self.cap
    
    def get_max_cycle(self):
        return self.max_cycle
    
    def get_bat_params(self):
        self.bat_params = [self.cap, self.rte, self.charging_rate, self.max_cycle, self.n_hour]
        return self.bat_params
    
    def get_bat_charging_profile(self):
        if self.charging_profile is None:
            print("NO, no info on battery charging profile. Need to run optimization first")
        else: 
            return self.charging_profile
        
    def get_bat_soc(self):
        if self.soc is None:
            print("NO, no info on battery SOC. Need to run optimization first")
        else: 
            return self.soc
        
    def get_u_d(self):
        if self.u_d is None:
            print("You are not supposed to do this")
        else:
            return self.u_d
            
    def get_u_c(self):
        if self.u_c is None:
            print("You are not supposed to do this")
        else:
            return self.u_c
        
    def get_new_bat_max_cap(self):
        if self.max_new_cap is None:
            print("You are not supposed to do this")
        else: 
            return self.max_new_cap
    
    # Setter to update optimization results
    def set_opt_result(self, result_dic):
        self.opt_result = result_dic
        self.cap = result_dic['Capacity']
        self.n_hour = result_dic['n_hour']
        self.u_c = np.array(result_dic['u_c']).flatten() # Idk if need flatten, just in case
        self.u_d = np.array(result_dic['u_d']).flatten() # Idk if need flatten, just in case
        self.soc = np.array(result_dic['SOC']).flatten() # Idk if need flatten, just in case
        self.charging_profile = np.array(result_dic['Charging profile']).flatten()
    
    # Setter to update new bat cap  
    def set_new_bat_max_cap(self, max_cap):
        self.max_new_cap = max_cap