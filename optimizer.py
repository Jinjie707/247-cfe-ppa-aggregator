from pyomo.environ import *
import streamlit as st
from opt_methods import *

class Optimizer:
    """
    A class to handle optimization problem setup and solving.

    Attributes:
        processed_inputs (dict): Dictionary containing preprocessed data from InputHandler.
        ppa_params (PPAParams): PPAParams object from ParamHandler. 
        opt_case (int): Integer representing the case type (from 0 to 11).
        model (OptimizationCase): A OptimizationCase instance.
    """
    def __init__(self, processed_inputs, ppa_params, case_type):
        self.processed_inputs = processed_inputs
        self.ppa_params = ppa_params
        self.opt_case = case_type
        self.model = None

    def solve(self, optimizer='glpk'):
        """
        First construct an OptimizationCase instance accroding to the case type,
        then run the selected optimization case using a specified solver.

        Args:
            optimizer (str): Name of the solver to use (default: 'glpk').

        Returns:
            dict: A dictionary of solution results from the model's `solve()` method.
        """
        model = self.identify_case()
        
        return model.solve(optimizer)
       
       
    def identify_case(self):
        """
        Construct the appropriate optimization problem based on the specified case type.

        Returns:
            OptimizationCase: A instance representing the optimization probelm of the corresponding case. 
        """
        if self.opt_case == 0:
            model = NoPenNoBat(self.processed_inputs, self.ppa_params)
    
        elif self.opt_case == 1:
            model = NoPenNoBatNewAsset(self.processed_inputs, self.ppa_params)
            
        elif self.opt_case == 2:
            model = NoPenNewBat(self.processed_inputs, self.ppa_params)
            
        elif self.opt_case == 3:
            model = NoPenNewBatNewAsset(self.processed_inputs, self.ppa_params)
            
        elif self.opt_case == 4:
            model = NoPenExBat(self.processed_inputs, self.ppa_params)
            
        elif self.opt_case == 5:
            model = NoPenExBatNewAsset(self.processed_inputs, self.ppa_params)
            
        elif self.opt_case == 6:
            model = PenNoBat(self.processed_inputs, self.ppa_params)
            
        elif self.opt_case == 7:
            model = PenNoBatNewAsset(self.processed_inputs, self.ppa_params)
            
        elif self.opt_case == 8:
            model = PenNewBat(self.processed_inputs, self.ppa_params)
            
        elif self.opt_case == 9:
            model = PenNewBatNewAsset(self.processed_inputs, self.ppa_params)
            
        elif self.opt_case == 10:
            model = PenExBat(self.processed_inputs, self.ppa_params)
            
        elif self.opt_case == 11:
            model = PenExBatNewAsset(self.processed_inputs, self.ppa_params)
            
        self.model = model
        
        return model
        
    def print_case(self):
        """
        Debugging print, display the selected case after clicking run button.
        """
        
        if self.opt_case == 0:
            st.header('Case 0: No Pen No Bat')

        elif self.opt_case == 1:
            st.header('Case 1: No Pen No Bat New Asset')
            
        elif self.opt_case == 2:
            st.header('Case 2: No Pen New Bat')
            
        elif self.opt_case == 3:
            st.header('Case 3: No Pen New Bat New Asset')
            
        elif self.opt_case == 4:
            st.header('Case 4: No Pen Ex Bat')
            
        elif self.opt_case == 5:
            st.header('Case 5: No Pen Ex Bat New Asset')
            
        elif self.opt_case == 6:
            st.header('Case 6: Pen No Bat')
            
        elif self.opt_case == 7:
            st.header('Case 7: Pen No Bat New Asset')
            
        elif self.opt_case == 8:
            st.header('Case 8: Pen New Bat')
            
        elif self.opt_case == 9:
            st.header('Case 9: Pen New Bat New Asset')
            
        elif self.opt_case == 10:
            st.header('Case 10: Pen Ex Bat')
            
        elif self.opt_case == 11:
            st.header('Case 11: Pen Ex Bat New Asset')
            
