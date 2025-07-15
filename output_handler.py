import streamlit as st
import pandas as pd
import numpy as np

class OutputHandler:
    """
    A class to process and enable download of the optimization results

    Attributes:
        case (int): Case index (0 to 11)
    """
    def __init__(self, case_type):
        self.case = case_type

    def save_results(self, processed_inputs, opt_result_dict):
        """
        Main public method to trigger saving logic based on the case index.
        Calls the appropriate save_results_case_X method.

        Args:
            processed_inputs (dict): Inputs used for optimization.
            output (dict): Output results from optimization.
        """
        if self.case == 0:
            self.save_results_case_0(processed_inputs, opt_result_dict)
        elif self.case == 1:
            self.save_results_case_1(processed_inputs, opt_result_dict)
        elif self.case == 2:
            self.save_results_case_2(processed_inputs, opt_result_dict)
        elif self.case == 3:
            self.save_results_case_3(processed_inputs, opt_result_dict)
        elif self.case == 4:
            self.save_results_case_4(processed_inputs, opt_result_dict)
        elif self.case == 5:
            self.save_results_case_5(processed_inputs, opt_result_dict)
        elif self.case == 6:
            self.save_results_case_6(processed_inputs, opt_result_dict)
        elif self.case == 7:
            self.save_results_case_7(processed_inputs, opt_result_dict)
        elif self.case == 8:
            self.save_results_case_8(processed_inputs, opt_result_dict)
        elif self.case == 9:
            self.save_results_case_9(processed_inputs, opt_result_dict)
        elif self.case == 10:
            self.save_results_case_10(processed_inputs, opt_result_dict)
        elif self.case == 11:
            self.save_results_case_11(processed_inputs, opt_result_dict)

    def process_result(self, processed_inputs, opt_result_dict):
        """
        Generate a DataFrame containing generation, consumption and usage data.
        Remark that battery properties not written here.
        
        Args:
            processed_inputs (dict): Dictionary containing preprocessed data from InputHandler.
            opt_result_dict (dict): Dictionary containing optimization result from OptimizationCase.solve().

        Returns:
            pd.DataFrame: A DataFrame contains results relevant to generations and consumptions.
        """
        generator_list = processed_inputs['Generators'].get_generator_list()
        customer_list = processed_inputs['Customers'].get_customer_list()
        
        beta = opt_result_dict['beta']
        
        v_total = opt_result_dict['v_total']
        v_renewable = opt_result_dict['v_renewable']
        v_buy = opt_result_dict['v_buy']

        output_df = pd.DataFrame()

        # Add generators data to output_df with formatting
        for i in range(len(generator_list)):
            gen_temp = generator_list[i]
            prod_item = gen_temp.get_gen_profile()
            beta_item = beta[i]
            used_prod_item = np.multiply(np.array(prod_item), np.array(beta_item))
            
            temp_name = gen_temp.get_gen_name()
            prod_name = temp_name + ' Hourly Production (MWh)'
            used_prod_name = temp_name + ' PPA Contribution (MWh)'
            beta_name = temp_name + ' beta'
            
            output_df[prod_name] = prod_item
            output_df[used_prod_name] = used_prod_item
            output_df[beta_name] = beta_item
        
        output_df['Agg Hourly Renewable Provision (MWh)'] = v_renewable
        output_df['Hourly Purchase from Mkt (MWh)'] = v_buy
        output_df['Hourly Total Energy Provision (MWh)'] = v_total

        # Add customers data to output_df with formatting
        for i in range(len(customer_list)):
            cus_temp = customer_list[i]
            item = cus_temp.get_cus_profile()
            temp_name = cus_temp.get_cus_name()
            name = temp_name + ' Hourly Demand (MWh)'
            output_df[name] = item
        output_df['Hourly Total Consumption (MWh)'] = processed_inputs['Customers'].get_agg_cus_profile()
        
        return output_df
        
    ##########################################################
    # A bunch of helper functions that will be called multiple times for diff cases
    
    def _helper_save_results(self, processed_inputs, opt_result_dict):
        """
        Display a button to download generations and consumptions output as a CSV.
        This has to be the first to call as it has the header.
        """
        output_df = self.process_result(processed_inputs, opt_result_dict)
        
        st.subheader('Download Optimization Results', divider = 'blue') 
        @st.fragment()
        def download():
            st.download_button(
                label = 'Download optimization result to CSV',
                data =  output_df.to_csv(index = False),
                file_name = 'optimization_output.csv',
                mime = 'text/csv',
                key = 'save results'
            )
        download()
        
    def _helper_save_ppa_price(self, opt_result_dict):
        """
        Display a button to download optimized PPA price as a txt file.
        """
        ppa_price = np.round(opt_result_dict['optimal price'], 4)
        
        @st.fragment()
        def download():
            st.download_button(
                label = 'Save Optimal PPA price to text',
                data = 'Optimal PPA price is ' + str(ppa_price) + '$/MWh',
                file_name = 'ppa_price.txt',
                mime = 'text/plain',
                key = 'save ppa price'
            )
        download()

    def _helper_save_ppa_price_and_bat_cap(self, opt_result_dict, opt_bat_cap):
        """
        Display a button to download optimized PPA price and battery capacity as a txt file.
        """
        ppa_price = np.round(opt_result_dict['optimal price'], 4)
        data = ('Optimal PPA price is ' + str(ppa_price) + '$/MWh' + ' \n' 
            + 'Optimal battery capacity is ' + str(np.round(opt_bat_cap, 4)) + 'MWh')
        
        @st.fragment()
        def download():
            st.download_button(
                label = 'Save optimal PPA price and battery capacity to text',
                data = data,
                file_name = 'ppa_price_and_bat_capacity.txt',
                mime = 'text/plain',
                key = 'save both ppa price and battery capacity optimized'
            )
        download()
        
    def _helper_save_ppa_price_and_gen_cap(self, opt_result_dict):
        """
        Display a button to download optimized PPA price and new generator capacity as a txt file.
        """
        ppa_price = np.round(opt_result_dict['optimal price'], 4)
        new_gen_cap = np.round(opt_result_dict['optimal capacity'], 4)
        data = ('Optimal PPA price is ' + str(ppa_price) + '$/MWh' + ' \n' 
            + 'Optimal generator capacity is ' + str(new_gen_cap) + 'MWh')
        
        @st.fragment()
        def download():
            st.download_button(
                label = 'Save optimal PPA price and generator capacity to text',
                data = data,
                file_name = 'ppa_price_and_gen_capacity.txt',
                mime = 'text/plain',
                key = 'save both ppa price and generator capacity optimized'
            )
        download()
        
    def _helper_save_ppa_price_and_bat_cap_and_gen_cap(self, opt_result_dict, opt_bat_cap):
        """
        Display a button to download optimized PPA price, new battery capacity 
        and new generator capacity as a txt file.
        """
        ppa_price = np.round(opt_result_dict['optimal price'], 4)
        new_gen_cap = np.round(opt_result_dict['optimal capacity'], 4)
        data = ('Optimal PPA price is ' + str(ppa_price) + '$/MWh' + ' \n' 
            + 'Optimal generator capacity is ' + str(new_gen_cap) + 'MWh' + ' \n' 
            + 'Optimal battery capacity is ' + str(np.round(opt_bat_cap, 4)) + 'MWh')
        
        @st.fragment()
        def download():
            st.download_button(
                label = 'Save optimal PPA price, generator capacity & battery capacity to text',
                data = data,
                file_name = 'ppa_price_and_bat_capacity.txt',
                mime = 'text/plain',
                key = 'save ppa price, generator capacity and battery capacity optimized'
            )
        download()
        
    def _helper_save_bat_profile_and_soc(self, bat_profile, bat_soc):
        """
        Display a button to download optimized battery data as a CSV.
        """
        bat_df = pd.DataFrame()
        bat_df['Change in Energy (MWh)'] = np.array(bat_profile)
        bat_df['Level of Energy (LOE) (MWh)'] = np.array(bat_soc)
        @st.fragment()
        def download():
            st.download_button(
                label = 'Download battery profile as CSV',
                # data = csv
                data =  bat_df.to_csv(index = False),
                file_name = 'bat_opt_output.csv',
                mime = 'text/csv',
                key = 'save results with battery'
            )
        download()
    
    def _helper_get_bat_results(self, processed_inputs):
        """
        Helper function to get battery parameters.
        """
        ex_bat = processed_inputs['Existing Battery'][1]
        bat_profile = ex_bat.get_bat_charging_profile()
        bat_soc = ex_bat.get_bat_soc()
        bat_cap = ex_bat.get_bat_cap()
        
        return bat_cap, bat_soc, bat_profile
        
    ##########################################################
    # A bunch (12 to be precise) of functions that calls result output action for each case
    # Each calls some helper functions
            
    def save_results_case_0(self, processed_inputs, opt_result_dict): 
        self._helper_save_results(processed_inputs, opt_result_dict)
        self._helper_save_ppa_price(opt_result_dict)
        
    def save_results_case_1(self, processed_inputs, opt_result_dict): 
        self._helper_save_results(processed_inputs, opt_result_dict)
        self._helper_save_ppa_price_and_gen_cap(opt_result_dict)
        
    def save_results_case_2(self, processed_inputs, opt_result_dict): 
        bat_cap, bat_soc, bat_profile = self._helper_get_bat_results(processed_inputs)

        self._helper_save_results(processed_inputs, opt_result_dict)
        self._helper_save_ppa_price_and_bat_cap(opt_result_dict, bat_cap)
        self._helper_save_bat_profile_and_soc(bat_profile, bat_soc)
        
    # New battery New Asset
    def save_results_case_3(self, processed_inputs, opt_result_dict): 
        bat_cap, bat_soc, bat_profile = self._helper_get_bat_results(processed_inputs)

        self._helper_save_results(processed_inputs, opt_result_dict)
        self._helper_save_ppa_price_and_bat_cap_and_gen_cap(opt_result_dict, bat_cap)
        self._helper_save_bat_profile_and_soc(bat_profile, bat_soc)
        
    def save_results_case_4(self, processed_inputs, opt_result_dict): 
        bat_cap, bat_soc, bat_profile = self._helper_get_bat_results(processed_inputs)
        
        self._helper_save_results(processed_inputs, opt_result_dict)
        self._helper_save_ppa_price(opt_result_dict)
        self._helper_save_bat_profile_and_soc(bat_profile,bat_soc)
        
    def save_results_case_5(self, processed_inputs, opt_result_dict):         
        bat_cap, bat_soc, bat_profile = self._helper_get_bat_results(processed_inputs)
        
        self._helper_save_results(processed_inputs, opt_result_dict)
        self._helper_save_ppa_price_and_gen_cap(opt_result_dict)
        self._helper_save_bat_profile_and_soc(bat_profile,bat_soc)
    
    def save_results_case_6(self, processed_inputs, opt_result_dict): 
        self._helper_save_results(processed_inputs, opt_result_dict)
        self._helper_save_ppa_price(opt_result_dict)
        
    def save_results_case_7(self, processed_inputs, opt_result_dict): 
        self._helper_save_results(processed_inputs, opt_result_dict)
        self._helper_save_ppa_price_and_gen_cap(opt_result_dict)
        
    def save_results_case_8(self, processed_inputs, opt_result_dict): 
        bat_cap, bat_soc, bat_profile = self._helper_get_bat_results(processed_inputs)

        self._helper_save_results(processed_inputs, opt_result_dict)
        self._helper_save_ppa_price_and_bat_cap(opt_result_dict, bat_cap)
        self._helper_save_bat_profile_and_soc(bat_profile, bat_soc)
        
    def save_results_case_9(self, processed_inputs, opt_result_dict): 
        bat_cap, bat_soc, bat_profile = self._helper_get_bat_results(processed_inputs)
        
        self._helper_save_results(processed_inputs, opt_result_dict)
        self._helper_save_ppa_price_and_bat_cap_and_gen_cap(opt_result_dict, bat_cap)
        self._helper_save_bat_profile_and_soc(bat_profile, bat_soc)
        
    def save_results_case_10(self, processed_inputs, opt_result_dict): 
        bat_cap, bat_soc, bat_profile = self._helper_get_bat_results(processed_inputs)
        
        self._helper_save_results(processed_inputs, opt_result_dict)
        self._helper_save_ppa_price(opt_result_dict)
        self._helper_save_bat_profile_and_soc(bat_profile,bat_soc)
        
    def save_results_case_11(self, processed_inputs, opt_result_dict): 
        bat_cap, bat_soc, bat_profile = self._helper_get_bat_results(processed_inputs)
        
        self._helper_save_results(processed_inputs, opt_result_dict)
        self._helper_save_ppa_price_and_gen_cap(opt_result_dict)
        self._helper_save_bat_profile_and_soc(bat_profile,bat_soc)
        
