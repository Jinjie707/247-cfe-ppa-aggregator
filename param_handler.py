import streamlit as st
import pandas as pd
import numpy as np
from players import Generator
from players import Storage
from ppa_params import PPAParams

class ParamHandler:
    """
    A class for collecting and processing user-defined optimization mode and PPA parameters.
    
    Attributes:
        mode_dict (dict): Stores selected optimization options and configuration values.
        has_existing_bat (bool): Whether an existing battery is present in the scenario.
        consider_penalty (bool): Whether penalty price is considered.
    """
    def __init__(self):
        self.mode_dict = {}
        self.has_existing_bat = None
        self.consider_penalty = None
    
    # Get parameters on the selected mode of optimization
    def get_opt_mode_and_ppa_params(self, processed_inputs):
        """
        Main function to be called to render Streamlit UI for selecting optimization mode and PPA settings.
        Also builds and returns a PPAParams object based on user inputs.

        Args:
            processed_inputs (dict): Preprocessed data inputs from InputHandler, including battery info and market prices.

        Returns:
            PPAParams: An instance of the PPAParams class containing all configuration and user inputs.
        """
        self.has_existing_bat =  processed_inputs['Existing Battery'][0]
        self.consider_penalty = processed_inputs['Market Prices'][0]

        st.header('24/7 CFE PPA Optimization', divider = 'blue')
        main_col1, main_col2 = st.columns([1.5,2])
        
        # Left column for Optimization mode parameters
        with main_col1:
            mode_dict = {
                'Existing Battery': self.has_existing_bat,
                'Consider Penalty': self.consider_penalty,
                
                # Simulation is not used for now
                'Simulation': False,
                'Number of Simulations': 0,

                'New Asset': False,
                'New Asset Params': [],
                'New Asset PPA Price': None,

                'New Battery': False,
                'New Battery Params': []
                }

            # Displays section header
            st.subheader('Select Mode of Optimization', divider = 'gray')
            
            # Display selection radio button for optimization mode:
            #   1. Optimize only PPA price
            #   2. Optimize both PPA price and new gen/bat cap
            mode = st.radio(
                "Please select the optimization mode",
                ["Optimize PPA price", "Optimize PPA price and new gen/bat capacity"])#, "Run simulations"])
            
            # Display selection radio button for optimization mode: optimize with new gen/bat
            if mode == "Optimize PPA price and new gen/bat capacity":
                ## For battery, only one bat allowed. No new bat if there's existing bat
                if self.has_existing_bat:
                    st.write('Only 1 bat allowed, since you have an existing battery, NO NEW Battery allowed.')
                    new_asset_options  = ["New generation assets"]
                else:
                    new_asset_options = st.multiselect(
                        "Select Types of new assets (multiple new generations, max 1 bat)",
                        ["New generation assets", "New battery"],
                        ["New generation assets"],
                    )
                # For Optimizing only new gen
                if new_asset_options == ['New generation assets']:
                    mode_dict['New Asset'] = True
                
                    # Takes in input: ref_prod_profile, ref_prod_cap, lcoe, pap_ppa_price (for selling excess prod)
                    self.all_files_uploaded = False
                    col1, col2 = st.columns([1.5, 1])
                    with col1:
                        new_asset_prod_file = st.file_uploader("Upload reference production data of the new asset ", type="xlsx")
                    with col2:
                        ref_capacity = st.number_input('Insert capacity of the reference data (MW)', min_value = 0)
                        new_lcoe = st.number_input('Insert LCOE ($/MWh)', min_value = 0)
                        pap_ppa_price = st.number_input('Insert PAP PPA price ($/MWh)', min_value = 0)
                    if new_asset_prod_file != None:
                        self.all_files_uploaded = True
                        new_asset_prod = np.array(pd.read_excel(new_asset_prod_file)['0'])
                        new_asset_prod =  new_asset_prod / ref_capacity
                        new_gen = Generator('New Generator', 'Unknown', new_asset_prod, new_lcoe)
                        mode_dict["New Asset Params"] = new_gen
                        mode_dict['New Asset PPA Price'] = pap_ppa_price
                
                # For optimizing only new bat
                elif new_asset_options == ['New battery']:
                    mode_dict['New Battery'] = True
                    st.write("Enter the battery parameters:")
                    
                    # Takes in input: lcos, bat_rte, bat_charging_rate, bat_max_cycle, bat_max_cap
                    col1, col2 = st.columns(2)
                    with col1:
                        lcos = st.number_input('Insert LCOS ($/MWh)', value = 120.0, min_value = 0.0)
                    with col2:
                        bat_rte = st.number_input('Insert battery round trip efficiency', value = 0.9, min_value = 0.0)

                    col3, col4 = st.columns(2)
                    with col3:
                        bat_charging_rate = st.number_input('Insert max charging rate', min_value = 1.0, step = 0.5, value = 25.0)
                    with col4:
                        bat_max_cycle = st.number_input('Insert battery max charging cycles', value = 1.0, min_value = 0.0, step = 0.5)
                        
                    col5, col6 = st.columns(2)
                    with col5:
                        bat_max_cap = st.number_input('Insert the maximum capacity for new battery (MWh)', min_value = 0.0, step = 0.5, value = 1000.0)
                    
                    # Set bat_cap to 0, this is optimization result to be updated
                    # bat_n_hour is redundant, but keep it just in case.
                    bat_capacity = 0
                    bat_n_hour = 0
                    
                    bat_params = [bat_capacity, bat_rte, bat_charging_rate, bat_max_cycle, bat_n_hour]
                    new_bat = Storage('New Bat', 'Unknown', bat_params, lcos)
                    new_bat.set_new_bat_max_cap(bat_max_cap)
                    mode_dict['New Battery Params'] = new_bat
                
                # For optimizing both new gen and new bat
                elif new_asset_options == ['New generation assets', 'New battery'] or new_asset_options == ['New battery','New generation assets']:
                    # Part 1: new asset
                    mode_dict['New Asset'] = True
                    self.all_files_uploaded = False
                    col1, col2 = st.columns([1.5, 1])
                    with col1:
                        new_asset_prod_file = st.file_uploader("Upload reference production data of the new asset ", type="xlsx")
                    with col2:
                        ref_capacity = st.number_input('Insert capacity of the reference data (MW)', min_value = 0)
                        new_lcoe = st.number_input('Insert LCOE ($/MWh)', min_value = 0)
                        pap_ppa_price = st.number_input('Insert PAP PPA price ($/MWh)', min_value = 0)
                    if new_asset_prod_file != None:
                        self.all_files_uploaded = True
                        new_asset_prod = np.array(pd.read_excel(new_asset_prod_file)['0'])
                        new_asset_prod =  new_asset_prod / ref_capacity
                        new_gen = Generator('New Generator', 'Unknown', new_asset_prod, new_lcoe)
                        mode_dict["New Asset Params"] = new_gen
                        mode_dict['New Asset PPA Price'] = pap_ppa_price
                        
                    # Part 2: new bat
                    mode_dict['New Battery'] = True
                    st.write("Enter the battery parameters:")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        lcos = st.number_input('Insert LCOS ($/MWh)', value = 120.0, min_value = 0.0)
                    with col2:
                        bat_rte = st.number_input('Insert battery round trip efficiency', value = 0.9, min_value = 0.0)

                    col3, col4 = st.columns(2)
                    with col3:
                        bat_charging_rate = st.number_input('Insert max charging rate', min_value = 1.0, step = 0.5, value = 25.0)
                    with col4:
                        bat_max_cycle = st.number_input('Insert battery max charging cycles', value = 1.0, min_value = 0.0, step = 0.5)
                        
                    col5, col6 = st.columns(2)
                    with col5:
                        bat_max_cap = st.number_input('Insert the maximum capacity for new battery (MWh)', min_value = 0.0, step = 0.5, value = 1000.0)
                    
                    bat_capacity = 0
                    bat_n_hour = 0
                    
                    bat_params = [bat_capacity, bat_rte, bat_charging_rate, bat_max_cycle, bat_n_hour]
                    new_bat = Storage('New Bat', 'Unknown', bat_params, lcos)
                    new_bat.set_new_bat_max_cap(bat_max_cap)
                    mode_dict['New Battery Params'] = new_bat
                    
                else:
                    print("Something wrong in input handler / new asset options: ", new_asset_options)
        
            st.markdown(
                """<style>
            div[class*="stRadio"] > label > div[data-testid="stMarkdownContainer"] > p {
                font-size: 20px;
            }
                </style>
                """, unsafe_allow_html=True)

        # Right column for PPA Parameters
        with main_col2:
            st.subheader('Select PPA Parameters', divider = 'gray')
            # CFE Target
            with st.container():
                st.write("Select target CFE %")
                coveragepct = st.select_slider(
                    "a hidden label", label_visibility = 'collapsed',
                    options=[f"{i}%" for i in np.arange(0, 101, 1)],
                    value="100%",
                )
                coveragepct = int(coveragepct[:-1])/100
            
            # Matching time step (Now only hourly matching)
            st.write("Select matching timestep (NOTE: Only Hourly matching for current version)")
            option = st.selectbox(
                'Select time step',
                ('Hourly'))#, 'Daily', 'Weekly', 'Monthly'))

            time_step = 1
            
            if option == "Hourly":
                time_step = 1
            elif option == "Daily":
                time_step = 24
            elif option == "Weekly":
                time_step = 24*7
            elif option == "Monthly":
                time_step = 7*30
            else:
                print("Hi Something is wrong")
            
            # Length of contract (Now only 1 year)
            st.write("Enter the length of PPA contract (NOTE: Only 1-year analysis avaliable for current version)")
            maturity = st.number_input(
                'Insert the length of contract (Year)', value = 1, step = 1, min_value = 1, max_value = 1)

            # Redundant code about PPA Penalty term
            # NOTE Not used now. Penalty is now considered in the InputHandler.
            # Keep here to avoid too much changes to the code structure.
            penalty_factor = -1 
            penalty_cap = -1

        st.divider()
        
        ppa_params = PPAParams(mode_dict, coveragepct, time_step, maturity, penalty_factor, penalty_cap)
        return ppa_params
