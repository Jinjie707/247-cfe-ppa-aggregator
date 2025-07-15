import numpy as np
import pandas as pd

class DataProcessor:
    """
    A class to process generation and consumption data, 
    compute CFE matching (with battery)

    Attributes:
        prods (np.ndarray): 2d nparray (8763, n) of generator data.
        demands (np.ndarray): 2d nparray (8762, m) of customer data.
        storage (Storage, optional): A storage object. Defaults to None.
    """

    def __init__(self, input_prod_array, input_demand_array, storage=None):
        self.prods = input_prod_array
        self.demands = input_demand_array
        self.storage = storage

    def get_prod_view_df(self):
        """
        Prepares a DataFrame for viewing and selecting generators in Streamlit UI.

        Returns:
            pd.DataFrame: DataFrame with generator metadata and time series data.
        """
        prod_view_info = []
        no_prods = self.prods.shape[1]
        reshaped_prods = self.prods[3:].T

        for i in range(no_prods):
            temp_name = self.prods[0][i]
            temp_type = self.prods[1][i]
            temp_lcoe = self.prods[2][i]
            temp_prods = reshaped_prods[i]

            temp_row = {'Name': temp_name,
                        'Type': temp_type,
                        'LCOE': temp_lcoe,
                        'View': False,
                        'Value': temp_prods
                        }
            prod_view_info.append(temp_row)
            
        prod_view_df = pd.DataFrame(prod_view_info)
        
        return prod_view_df
    
    def get_demand_view_df(self):
        """
        Prepares a DataFrame for viewing and selecting customers in Streamlit UI.

        Returns:
            pd.DataFrame: DataFrame with customer metadata and time series data.
        """
        demand_view_info = []
        no_demands = self.demands.shape[1]
        reshaped_demands = self.demands[2:].T

        for i in range(no_demands):
            temp_name = self.demands[0][i]
            temp_type = self.demands[1][i]
            temp_demands = reshaped_demands[i]
            
            temp_row = {'Name': temp_name,
                        'Type': temp_type,
                        'View': False,
                        'Value': temp_demands
                        }
            demand_view_info.append(temp_row)
            
        demand_view_df = pd.DataFrame(demand_view_info)
        
        return demand_view_df

    def compute_agg_matching(self, matching_method, selected_prods, selected_demands):
        """
        Computes the CFE matching score and related metrics (with battery).

        Args:
            matching_method (int): Aggregation level (e.g., 1 for hourly, 24 for daily).
            selected_prods (np.ndarray): 2d nparray (8760, n), time series of generation data`.
            selected_demands (np.ndarray): 2d nparray (8760, m), time series of consumption data.

        Returns:
            dict: Matching results and metrics.
        """
        total_prods = np.sum(selected_prods, axis = 1)
        total_demands = np.sum(selected_demands, axis = 1)
                
        agg_prods = total_prods.reshape(-1, matching_method).sum(axis=1)
        agg_demands = total_demands.reshape(-1, matching_method).sum(axis=1)
        
        # No battery case
        if self.storage is None:
            total_excess_prods = 0
            prov = 0
            for i in range(len(agg_prods)):
                cur_prov = min(agg_prods[i], agg_demands[i])
                prov += cur_prov
                total_excess_prods += max(agg_prods[i] - cur_prov, 0)

            if np.sum(agg_demands) == 0:
                return 100, np.sum(agg_prods)

            agg_matching = np.round(prov / np.sum(agg_demands)*100, 2)
            
            res_total_demands = np.round(np.sum(agg_demands)/1000, 2)
            res_matched_demands = np.round(prov/1000, 2)
            res_unmatched_demands = np.round((res_total_demands - res_matched_demands), 2)
            unmatched_ratio = np.round(res_unmatched_demands / res_total_demands*100, 2)
            
            res_total_gen = np.round(np.sum(agg_prods)/1000, 2)
            res_allocated_gen = res_matched_demands
            res_excess_gen = np.round((res_total_gen - res_allocated_gen), 2)
            waste_ratio = np.round(total_excess_prods / np.sum(agg_prods) * 100, 2)
            
            res_dic =  {
                'agg_matching': agg_matching,
                'total_demands': res_total_demands,
                'matched_demands': res_matched_demands,
                'unmatched_demands': res_unmatched_demands,
                'unmatched_ratio': unmatched_ratio,
                'total_gen': res_total_gen,
                'allocated_gen': res_allocated_gen,
                'excess_gen': res_excess_gen,
                'waste_ratio': waste_ratio
            }
            
        # With battery case
        else:
            total_prods = np.sum(selected_prods, axis = 1)
            total_demands = np.sum(selected_demands, axis = 1)
            
            agg_prods = total_prods.reshape(-1, matching_method).sum(axis=1) # Aggregated / reshaped
            agg_demands = total_demands.reshape(-1, matching_method).sum(axis=1) # Aggregated / reshaped
            
            [bat_cap, bat_rte, bat_charging_rate, bat_max_cycle, bat_n_hour] = self.storage.get_bat_params()
            total_excess_prods = 0
            re_bat_prov = 0
            # Initialize battery max daily charging cycle limit
            bat_max_daily_charge = bat_cap * bat_max_cycle
            bat_total_char = 0
            bat_soc = 0
            bat_profile = []
            bat_total = [0]
            
            if matching_method == 1: # Hourly scenario
                bat_cur_daily_charge = 0
                for i in range(len(agg_prods)):
                    
                    if i%24 == 0: # reset daily charging limit
                        bat_cur_daily_charge = 0 
                    if agg_prods[i] >= agg_demands[i]: # Charing scenario
                        temp_char_capacity = bat_cap - bat_soc
                    
                        temp_daily_limit = max(bat_max_daily_charge - bat_cur_daily_charge, 0)
                        
                        temp_excess_prod = agg_prods[i] - agg_demands[i]
                        temp_char_capacity = min(temp_daily_limit, temp_char_capacity)
                        temp_char_capacity = min(temp_char_capacity, bat_charging_rate)
                        temp_charge = min(temp_char_capacity, temp_excess_prod)
                        
                        bat_soc += temp_charge
                     
                        bat_cur_daily_charge += temp_charge
                        total_excess_prods += (temp_excess_prod - temp_charge)
                        re_bat_prov += agg_demands[i]
                        bat_total_char += temp_charge
                        bat_profile.append(temp_charge)
                        bat_total.append(bat_total[-1] + temp_charge)

                    else: # Discharging scenario
                        temp_insuf_prod = agg_demands[i] - agg_prods[i]
                        
                        temp_dischar_capacity = min(bat_soc, bat_charging_rate) # discharging is limited by total energy available and discharging rate
                        temp_dischar = min(temp_dischar_capacity, temp_insuf_prod)
                        
                        bat_soc -= temp_dischar
                        re_bat_prov += agg_prods[i] + temp_dischar*bat_rte
                        bat_profile.append(-temp_dischar)
                        bat_total.append(bat_total[-1] - temp_dischar)
                        
                
            elif matching_method == 24: # Daily matching scenario, matching method = 24
                for i in range(len(agg_prods)):
                    if agg_prods[i] >= agg_demands[i]: # charging scenario - If the total production is more the total consumption in a day
                        bat_cur_daily_charge = 0
                        max_daily_charge = agg_prods[i] - agg_demands[i]
                        max_daily_charge = min(bat_max_daily_charge, max_daily_charge)

                        for j in range(24): 
                            temp_char_capacity = bat_cap - bat_soc # Avaliable SOC at the moment
  
                            temp_daily_limit = max(max_daily_charge - bat_cur_daily_charge, 0)
                            temp_excess_prod = max(total_prods[24*i + j] - total_demands[24*i + j], 0)
                            
                            temp_char_capacity = min(temp_daily_limit, temp_char_capacity)
                
                            temp_char_capacity = min(temp_char_capacity, bat_charging_rate)
                            temp_charge = min(temp_char_capacity, temp_excess_prod)
                                
                            bat_soc += temp_charge
                            bat_cur_daily_charge += temp_charge
                            total_excess_prods += (temp_excess_prod - temp_charge)
                            re_bat_prov += total_demands[24*i + j]
                    
                            bat_profile.append(temp_charge)
                       
                            bat_total.append(bat_total[-1] + temp_charge)
                            
                        bat_total_char += bat_cur_daily_charge
                    else: # Discharging scenario
                        bat_cur_daily_discharge = 0
                        max_daily_discharge = agg_demands[i] - agg_prods[i]
                        
                        for j in range(24):
                            temp_insuf_prod = max(total_demands[24*i+j] - total_prods[24*i+j], 0)
                            
                        
                            temp_dischar = min(bat_soc, bat_charging_rate)
                            temp_dischar = min(temp_dischar, max_daily_discharge)
                            temp_dischar = min(temp_dischar, temp_insuf_prod)
                            
                            bat_soc -= temp_dischar
                            re_bat_prov += total_prods[24*i+j] + temp_dischar * bat_rte
                            max_daily_discharge -= temp_dischar
                            bat_cur_daily_discharge += temp_dischar
                            bat_profile.append(-temp_dischar)
                            bat_total.append(bat_total[-1] - temp_dischar)
                    
            agg_matching = np.round(re_bat_prov / np.sum(agg_demands)*100, 2)
            
            res_total_demands = np.round(np.sum(agg_demands)/1000, 2)
            res_matched_demands = np.round(re_bat_prov/1000, 2)
            res_unmatched_demands = np.round((res_total_demands - res_matched_demands), 2)
            unmatched_ratio = np.round(res_unmatched_demands / res_total_demands*100, 2)
            
            # Total renewable generation
            res_total_gen = np.round(np.sum(agg_prods)/1000, 2)
            res_allocated_gen = np.round((np.sum(agg_prods) - total_excess_prods)/1000, 2)
            bat_energy_loss = np.round(res_allocated_gen - res_matched_demands, 2)
            
            res_excess_gen = np.round((res_total_gen - res_allocated_gen), 2)
            waste_ratio = np.round(total_excess_prods / np.sum(agg_prods) * 100, 2)
            
            bat_utilization_ratio = np.round((bat_total_char / (365*bat_cap * bat_max_cycle))*100, 2)
            res_dic =  {
                'agg_matching': agg_matching,
                'total_demands': res_total_demands,
                'matched_demands': res_matched_demands,
                'unmatched_demands': res_unmatched_demands,
                'unmatched_ratio': unmatched_ratio,
                'total_gen': res_total_gen,
                'allocated_gen': res_allocated_gen,
                'excess_gen': res_excess_gen,
                'waste_ratio': waste_ratio,
                'bat_uti_ratio': bat_utilization_ratio,
                'bat_energy_loss': bat_energy_loss
            }
                 
        return res_dic

    def get_bat_modified_prod(self, matching_method, total_prods, total_demands):
        """
        Returns battery-modified production profile after accounting for charging/discharging.

        Args:
            matching_method (int): 1 = hourly, 24 = daily
            total_prods (np.ndarray): 1d nparray (8760,), time series of aggregated generation
            total_demands (np.ndarray): 1d nparray (8760,), time series of aggregated consumption

        Returns:
            np.ndarray: 1d nparray (8760, ), adjusted aggregated production after battery smoothing, used for plotting
        """
        agg_prods = total_prods.reshape(-1, matching_method).sum(axis=1)
        agg_demands = total_demands.reshape(-1, matching_method).sum(axis=1)
        
        [bat_cap, bat_rte, bat_charging_rate, bat_max_cycle, bat_n_hour] = self.storage.get_bat_params()
        total_excess_prods = 0
        re_bat_prov = 0
        # Initialize battery max daily charging cycle limit
        bat_max_daily_charge = bat_cap * bat_max_cycle
        bat_total_char = 0
        bat_soc = 0
        bat_profile = []
        bat_total = [0]
        
        # Compute battery charge & discharge wrt the battery constraints
        if matching_method == 1: # Hourly scenario
            for i in range(len(agg_prods)):
                bat_cur_daily_charge = 0

                if agg_prods[i] >= agg_demands[i]: # Charing scenario
                    temp_char_capacity = bat_cap - bat_soc
                    temp_daily_limit = max(bat_max_daily_charge - bat_cur_daily_charge, 0)
                    temp_excess_prod = agg_prods[i] - agg_demands[i]

                    temp_char_capacity = min(temp_daily_limit, temp_char_capacity)
                    temp_char_capacity = min(temp_char_capacity, bat_charging_rate)
                    temp_charge = min(temp_char_capacity, temp_excess_prod)
                    
                    bat_soc += temp_charge
                    bat_cur_daily_charge += temp_charge
                    total_excess_prods += (agg_prods[i] - temp_charge)
                    re_bat_prov += agg_demands[i]
                    bat_total_char += temp_charge
                    bat_profile.append(temp_charge)
                    bat_total.append(bat_total[-1] + temp_charge)
        
                    
                    if i%24 == 0: # reset daily charging limit
                        bat_cur_daily_charge = 0 
                
                else: # Discharging scenario
                    temp_insuf_prod = agg_demands[i] - agg_prods[i]
                    
                    temp_dischar_capacity = min(bat_soc, bat_charging_rate) # discharging is limited by total energy available and discharging rate
                    temp_dischar = min(temp_dischar_capacity, temp_insuf_prod)
                    
                    bat_soc -= temp_dischar
                    re_bat_prov += agg_prods[i] + temp_dischar*bat_rte
                    bat_profile.append(-temp_dischar)
                    bat_total.append(bat_total[-1] - temp_dischar)
        
        elif matching_method == 24: # Daily matching scenario, matching method = 24
            for i in range(len(agg_prods)):
                if agg_prods[i] >= agg_demands[i]: # charging scenario - If the total production is more the total consumption in a day
                    bat_cur_daily_charge = 0
                    max_daily_charge = agg_prods[i] - agg_demands[i]
                    max_daily_charge = min(bat_max_daily_charge, max_daily_charge)

                    for j in range(24): 
                        temp_char_capacity = bat_cap - bat_soc # Avaliable SOC at the moment

                        temp_daily_limit = max(max_daily_charge - bat_cur_daily_charge, 0)
                        temp_excess_prod = max(total_prods[24*i + j] - total_demands[24*i + j], 0)
                        
                        temp_char_capacity = min(temp_daily_limit, temp_char_capacity)
            
                        temp_char_capacity = min(temp_char_capacity, bat_charging_rate)
                        temp_charge = min(temp_char_capacity, temp_excess_prod)

                            
                        bat_soc += temp_charge
                        bat_cur_daily_charge += temp_charge
                        total_excess_prods += (total_prods[24*i + j] - temp_charge)
                        re_bat_prov += total_demands[24*i + j]
                
                        bat_profile.append(temp_charge)
                    
                        bat_total.append(bat_total[-1] + temp_charge)
                        
                    bat_total_char += bat_cur_daily_charge
                else: # Discharging scenario
                    bat_cur_daily_discharge = 0
                    max_daily_discharge = agg_demands[i] - agg_prods[i]
                    
                    for j in range(24):
                        temp_insuf_prod = max(total_demands[24*i+j] - total_prods[24*i+j], 0)
                        
                    
                        temp_dischar = min(bat_soc, bat_charging_rate)
                        temp_dischar = min(temp_dischar, max_daily_discharge)
                        temp_dischar = min(temp_dischar, temp_insuf_prod)
                        
                        bat_soc -= temp_dischar
                        re_bat_prov += total_prods[24*i+j] + temp_dischar * bat_rte
                        max_daily_discharge -= temp_dischar
                        bat_cur_daily_discharge += temp_dischar
                        bat_profile.append(-temp_dischar)
                        bat_total.append(bat_total[-1] - temp_dischar)
                    
        modified_prod = total_prods - bat_profile
            
        return modified_prod
