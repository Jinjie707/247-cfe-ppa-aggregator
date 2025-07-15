import streamlit as st

from optimizer import Optimizer
from input_handler import InputHandler
from param_handler import ParamHandler
from output_handler import OutputHandler
from plotter import Plotter

# Set Streamlit page layout to wide
st.set_page_config(layout="wide")

# Instantiate input and parameter handler
ih = InputHandler()
ph = ParamHandler()

# InputHandler to load and preprocess user inputs
# Returns a dictionary processed_inputs
# {
#   "Generators": Generators object,
#   "Customers": Customers object,
#   "Market Prices": (bool consider_penalty, MarketPrice object or None),
#   "Existing Battery": (bool existing_bat, Storage object or None)
# }
processed_inputs = ih.get_inputs()

# Proceed only if all required input files have been uploaded
if ih.all_files_uploaded:
    
    # ParameterHandler to parse processed inputs into optimization mode and ppa parameters
    # Returns a PPAParams object
    ppa_params = ph.get_opt_mode_and_ppa_params(processed_inputs) 

    # Create two columns, col1 for run buttom and col2 is empty spaceholder
    col1, col2 = st.columns([1, 4])
    with col1:
        runbtndiv = st.empty()
    
    # Display Run Model button
    runbtn = runbtndiv.button("Run Model", type="primary")
    
    # When run button is clicked, hide button, show spinner, and run the model
    if runbtn:
        with st.spinner("Running model..."):
            runbtndiv.empty()  # Remove button after clicking
            
            # Get case number from PPAParams, returns an int
            case_type = ppa_params.get_case()
            
            # Instantiate Optimizer with all inputs
            ppa_optimizer = Optimizer(processed_inputs, ppa_params, case_type)
            
            #############################################
            #### Switch to print debugging statement ####
            
            ppa_optimizer.print_case()
            
            #############################################
            
            # Solve the optimization problem, returns a dictionary
            result_dict = ppa_optimizer.solve('glpk')
            
            # Infeasible case: Returns a statement
            if result_dict == 0:
                st.subheader('Problem is infeasible, please consider to add penalty terms')
                st.write('Refer to the Portfolio 24/7 CFE Analysis for reachable CFE % Target')
                st.divider()
                
            # Feasible cases: Proceeds to output and plotting
            else:
                oh = OutputHandler(case_type)
                oh.save_results(processed_inputs, result_dict)
                
                plt = Plotter(case_type, processed_inputs, ppa_params, result_dict, 24*30)
                plt.plot()
