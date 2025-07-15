import streamlit as st
import pandas as pd
import numpy as np
from players import Generator
from players import Generators
from players import Customer
from players import Customers
from players import MarketPrice
from players import Storage

from Input_viewer import DataProcessor
from Input_viewer import InputPlotter

class InputHandler:
    """
    Handles all user input interactions in the Streamlit interface for 
    the 24/7 CFE PPA Aggregator. This includes file uploads, parameter 
    collection, data validation, and preprocessing.
    """
    def __init__(self):
        self.all_files_uploaded = False
        
        self.uploaded_prod_array = None
        self.uploaded_demand_array = None
        
        self.existing_bat = False
        self.consider_penalty = False
        
        self.generators = []
        self.customers = []
        self.storages = None
        self.market_price = []
    
    def get_inputs(self):
        """
        Main entry point to handle user inputs via the Streamlit UI.
        Triggers upload, validation, processing, and structured packaging
        of input data.

        Returns:
            dict: A dictionary of processed inputs:
                {
                    "Generators": Generators,
                    "Customers": Customers,
                    "Market Prices": (bool, MarketPrice or None),
                    "Existing Battery": (bool, Storage or None)
                }
        """
        self.upload_data()
        self.generate_reminder()
        if self.all_files_uploaded:
            self.process_and_plot_data()
            processed_inputs = self.load_data()
            return processed_inputs
    
    
    def load_data(self):
        """
        Packages processed class attributes into a structured dictionary.

        Returns:
            dict: Dictionary of user input objects.
        """
        processed_input_dict = {
            'Generators': Generators(self.generators),
            'Customers': Customers(self.customers),
            'Market Prices': (self.consider_penalty, self.market_price), # A tuple, (Bool, MarketPirce() OR [])
            'Existing Battery': (self.existing_bat, self.storages) # A Tuple, (Bool, Storage OR None)
        }
        
        return processed_input_dict

    
    def upload_data(self):
        """
        Handles all user-facing file uploads and battery/penalty parameter
        collection. Includes validation for shape/length of data arrays.
        """
        st.title("24/7 CFE PPA Aggregator")
        
        # Display update notes
        with st.columns([1, 2])[0]:
            with st.expander("Update Notes - 20th May", icon= ':material/priority_high:'):
                st.write('''
                    Version: v0.2  |  Update Date: 2025-05-20
                    
                    - New Features: Implemented battery capacity optimization cases.
                        - Select Mode of Optimization -> Optimize PPA price and new gen/bat capacity ->
                        Select Types of new assets -> New battery
                        
                    - Bug Fix. 
                ''')
            
        st.header('Data Upload', divider = 'blue')
        
        gen_col, cus_col = st.columns([1.5,2])
        bat_col, mkt_col = st.columns([1.5,2])
        
        # Upload generator profile
        with gen_col:
            st.subheader('Upload Generator Data', divider = 'grey')
            gen_inputs = st.file_uploader("Upload generation data", type="xlsx")
        if gen_inputs != None:
            prods = np.array(pd.read_excel(gen_inputs))
            if len(prods) != (8760+3):
                st.write(len(prods))
                st.write('PRODS: Input not matching 8760 steps')
            else: 
                self.uploaded_prod_array = prods
        
        # Upload cusomer consumption profile    
        with cus_col:
            st.subheader('Upload Customer Data', divider = 'grey')
            cus_inputs = st.file_uploader("Upload consumption data", type="xlsx")
        if cus_inputs != None:
            demands = np.array(pd.read_excel(cus_inputs))
            if len(demands) != (8760+2):
                st.write(len(demands))
                st.write('DEMAND: Input not matching 8760*2 steps')
            else:
                self.uploaded_demand_array = demands
        
        # Upload battery parameters: LCOS, bat_capacity, bat_rte, bat_charging_rate, bat_max_cycle
        # Creates a Storage object with the uploaded params
        with bat_col:
            st.subheader('Enter Battery Parameters', divider = 'grey')
            
            existing_bat = st.radio(
                "Do you want to consider an existing battery?",
                ["No","Yes"], index = 0)
            
            if existing_bat == 'Yes':
                self.existing_bat = True
                            
                st.write("Enter the battery parameters:")
                lcos = st.number_input('Insert LCOS ($/MWh)', min_value = 0.0)
                
                col1, col2 = st.columns(2)
                with col1:
                    bat_capacity = st.number_input('Insert battery capacity (MWh)', min_value = 0.0, value = 100.0)
                    if bat_capacity == 0:
                        st.write("Existing battery should NOT having capacity of 0 MWh!")
                with col2:
                    bat_rte = st.number_input('Insert battery round trip efficiency', min_value = 0.0, max_value = 1.0, value = 0.9)

                col3, col4 = st.columns(2)
                with col3:
                    bat_charging_rate = st.number_input('Insert battery max charging rate (MW)', min_value = 0.001, value = 25.0)
                with col4:
                    bat_max_cycle = st.number_input('Insert battery max charging cycles', min_value = 0.1, value = 1.0)
                
                bat_n_hour = bat_capacity / bat_charging_rate
                
                bat_profile =  [bat_capacity, bat_rte, bat_charging_rate, bat_max_cycle, bat_n_hour]
                temp_sto = Storage("BAT NAME", 'BAT TYPE', bat_profile, lcos)
                self.storages = temp_sto
                
            elif existing_bat == 'No':
                pass
        
        # Upload market prices: market_rices, penalty_factor, penalty_cap
        # Creates an MarketPrice object with the uploaded params
        # Allows option of using RECs price or Spot price time series
        #   - But essentially there is no difference, it's just a time series of numbers
        #   - Currently disabled RECs price
        #   - TODO: Improve the penalty price representation
        with mkt_col:
            st.subheader('Upload Penalty Price Data', divider = 'grey')
            consider_penalty = st.radio(
                "Do you want to consider a penalty linked to market prices?",
                ["No (May lead to infeasible solution in achieving CFE % target)", 
                #  'Yes, penalty wtih RECs price', 
                 'Yes, penalty with spot price'], index = 0)
            
            # This branch with RECs price is disabled from UI (see 3 lines above)
            if consider_penalty ==  'Yes, penalty wtih RECs price':
                with st.container(border = False):
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.write('Please upload RECs price')
                        self.consider_penalty = True
                        
                        self.input_spotdata = st.file_uploader("Upload RECs price", type="xlsx", label_visibility = 'collapsed')
                        if self.input_spotdata:
                            spot = np.array(pd.read_excel(self.input_spotdata))                            
                            with col2:
                                st.write('Penalty Price Visualization')
                                df_penalty_view = pd.DataFrame(spot)
                                st.line_chart(df_penalty_view, width = 150, height = 200)
                if self.input_spotdata:
                    with st.container(border = False):
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            penalty_factor = st.number_input('Insert the level of penalty cost', value = 3, min_value = 0)
                            st.write('The current penalty cost is ', penalty_factor, ' times of the spot price')
                        with col2:
                            penalty_cap = st.number_input('Insert the cap of penalty cost', value = 500, min_value = 0)
                            st.write('The current cap is ', penalty_cap, ' $/MWh')
                        self.market_price = MarketPrice('PENALTY', spot, penalty_cap, penalty_factor)

            # Current option
            elif consider_penalty ==  'Yes, penalty with spot price':
                
                with st.container(border = False):
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.write('Please upload spot price')
                        self.consider_penalty = True
                        self.input_spotdata = st.file_uploader("Upload spot price", type="xlsx", label_visibility = 'collapsed')
                        if self.input_spotdata:
                            spot = np.array(pd.read_excel(self.input_spotdata))                            
                            with col2:
                                st.write('Penalty Price Visualization')
                                df_penalty_view = pd.DataFrame(spot)
                                st.line_chart(df_penalty_view, width = 150, height = 200)
                
                if self.input_spotdata:
                    st.subheader('Enter Penalty Parameters')
                    with st.container(border = False):
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            penalty_factor = st.number_input('Insert the level of penalty cost', value = 3, min_value = 0)
                            st.write('The current penalty cost is ', penalty_factor, ' times of the spot price')
                        with col2:
                            penalty_cap = st.number_input('Insert the cap of penalty cost', value = 500, min_value = 0)
                            st.write('The current cap is ', penalty_cap, ' $/MWh')
                        self.market_price = MarketPrice('PENALTY', spot, penalty_cap, penalty_factor)
                    

    def generate_reminder(self):
        """
        Validates completeness of user input and displays reminders for missing components.
        """
        is_gen_uploaded = self.uploaded_prod_array is not None
        if not is_gen_uploaded:
            st.write('Please upload generation data.')

        is_cus_uploaded = self.uploaded_demand_array is not None
        if not is_cus_uploaded:
            st.write('Please upload customer data.')

        is_sto_uploaded = not self.existing_bat or self.storages is not None
        if self.existing_bat and not is_sto_uploaded:
            st.write('Please add BESS Information.')

        is_penalty_uploaded = not self.consider_penalty or self.market_price != []
        if self.consider_penalty and not is_penalty_uploaded:
            st.write('Please add market price information.')
            
        self.all_files_uploaded = is_gen_uploaded and is_cus_uploaded and is_sto_uploaded and is_penalty_uploaded


    def process_and_plot_data(self):
        """
        Processes uploaded generation and consumption arrays into Generator and Customer
        class objects based on user selection. Also visualizes and calculates matching metrics.
        """
        st.divider()
            
        dp = DataProcessor(self.uploaded_prod_array, self.uploaded_demand_array, self.storages)
        prod_view_df = dp.get_prod_view_df()
        demand_view_df = dp.get_demand_view_df()
        st.header('Select Generators & Customers',divider = 'blue')
        # st.subheader('View and Select Generators & Customers')
        col1, col2 = st.columns([1,1])
        
        with col1: 
            st.write('Generator List')
            prod_edit_df = st.data_editor(prod_view_df, hide_index = True, column_config = {'Value':None})
        with col2:
            st.write('Customer List')
            demand_edit_df = st.data_editor(demand_view_df, hide_index = True, column_config = {'Value':None})

        st.write('')
        st.write('')
        st.write('')
        
        # Extract selected data dataframe
        selected_prods_df = prod_edit_df[prod_edit_df['View'] == True]
        selected_demands_df = demand_edit_df[demand_edit_df['View'] == True]
        
        if selected_prods_df.empty:
            st.write('Please select at least one generation asset.')
            self.all_files_uploaded = False
        elif selected_demands_df.empty:
            st.write('Please select at least one customer.')
            self.all_files_uploaded = False
        else:
            self.all_files_uploaded = True
            st.header('Portfolio 24/7 CFE Analysis', divider = 'blue')
            
            # Compute matching score
            # matching_col_1 to display for matching timestep selection
            # matching_col_2 to display metrics
            matching_col_1, matching_col_2 = st.columns([1, 4])
            with matching_col_1:
                st.subheader('Matching Option')
                time_step_dict = {'Hourly': 1, 'Daily': 24} #, 'Weekly': 48*7}
                matching_input = st.selectbox('Select matching timestep',
                                        ( 'Hourly', 'Daily'))#, 'Weekly'))
                matching_method = time_step_dict[matching_input]

            with matching_col_2:
                selected_prod_array = np.column_stack(selected_prods_df['Value']) # 2d nparray with shape (8760, n)
                selected_demand_array = np.column_stack(selected_demands_df['Value']) # 2d nparray with shape (8760, m)

                # Result metric df is computed with DataProcessor
                result_df = dp.compute_agg_matching(matching_method, selected_prod_array, selected_demand_array) # Returns result dictionary, refer to DataProcessor class

                # UI Displays of the Metrics
                st.subheader('Matching Metrics')
                st.metric(label = 'Max possible annual CFE % at the currenct matching timestep is', value = f"{result_df['agg_matching']}%", delta = None)

                col11, col12, col13, col14 = st.columns([1,1,1,1])
                with col11: 
                    st.metric(label = 'Total Consumption (GWh)', value = f"{result_df['total_demands']}", delta = None)
                with col12:
                    st.metric(label = 'Matched Consumption (GWh)', value = f"{result_df['matched_demands']}", delta = None)
                with col13:
                    st.metric(label = 'Unmatched Consumption (GWh)', value = f"{result_df['unmatched_demands']}", delta = None)
                with col14:
                    st.metric(label = 'Unmatched Con / Total Con', value = f"{result_df['unmatched_ratio']}%", delta = None)

                col21, col22, col23, col24 = st.columns(4)
                with col21:
                    st.metric(label = 'Total Generation (GWh)', value = f"{result_df['total_gen']}", delta = None)
                with col22:
                    st.metric(label = 'Allocated Generation (GWh)', value = f"{result_df['allocated_gen']}", delta = None)
                with col23:
                    st.metric(label = 'Excess Generation (GWh)', value = f"{result_df['excess_gen']}", delta = None)
                with col24:
                    st.metric(label = 'Excess Gen / Total Gen', value = f"{result_df['waste_ratio']}%", delta = None)
                    
                if self.existing_bat:
                    col31, col32, col33 = st.columns([1,1,2])
                    with col31:
                        st.metric(label = 'Bat Util Ratio', value = f"{result_df['bat_uti_ratio']}%", delta = None)
                    with col32:
                        st.metric(label = 'Bat Energy Loss (GWh)', value = f"{result_df['bat_energy_loss']}", delta = None)
        
            st.divider()
            
            # UI Displays of Matching Plot
            st.subheader('Matching Plot')
            
            input_plotter = InputPlotter(self.storages) # TODO: Better storage case passing around classes
            input_plotter.matching_plot(matching_method, selected_prods_df, selected_demands_df)
            
            #####################################################
            # Instantiate Generator and Customer objects according to selected generators and customers data
            # Append to the attributed list, will be later converted to Generators / Customers object
            for i in range(len(selected_prods_df)):
                temp_name = selected_prods_df.iloc[i]['Name']
                temp_type = selected_prods_df.iloc[i]['Type']
                temp_prods = selected_prods_df.iloc[i]['Value']
                temp_lcoe = selected_prods_df.iloc[i]['LCOE']
                
                temp_generator = Generator(temp_name, temp_type, temp_prods, temp_lcoe)
                self.generators.append(temp_generator)
            

            for i in range(len(selected_demands_df)):
                temp_name = selected_demands_df.iloc[i]['Name']
                temp_type = selected_demands_df.iloc[i]['Type']
                temp_demands = selected_demands_df.iloc[i]['Value']
                
                temp_customer = Customer(temp_name, temp_type, temp_demands)
                self.customers.append(temp_customer)

