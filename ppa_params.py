class PPAParams:
    """
    A class to store and manage PPA parameters
    
    This includes:
    - Optimization mode flags (e.g. whether a new battery or new generation is considered)
    - PPA contract params (target coverage %, matching timestep, maturity)
    - Penalty pricing terms
    - Case ID for optimizer selection (used to distinguish 12 cases)
    """
    
    def __init__(self, mode_dict, coveragepct, time_step, maturity, penalty_factor, penalty_cap):
        
        self.mode_dict = mode_dict
        
        self.has_new_bat = mode_dict['New Battery']
        self.new_bat_params = mode_dict['New Battery Params'] # A Storage object
        
        self.has_new_asset = mode_dict['New Asset']
        self.new_asset_params = mode_dict['New Asset Params'] # A generator object
        self.new_asset_ppa_price = mode_dict['New Asset PPA Price']
        
        self.coveragepct = coveragepct
        self.time_step = time_step
        self.maturity = maturity
        
        self.penalty_factor = penalty_factor
        self.penalty_cap = penalty_cap

        self.case = None
        
    def get_case(self):
        """
        Identify the case number (0 to 11) based on optimization configuration:
        - Presence of penalty pricing
        - Presence of existing or new battery
        - Whether new generator is optimized
        
        Returns:
            int: A case index between 0 and 11.
        """
        consider_penalty = self.mode_dict['Consider Penalty']
        has_existing_battery = self.mode_dict['Existing Battery']
        new_battery = self.mode_dict['New Battery']
        new_asset = self.mode_dict['New Asset']
        
        if has_existing_battery and new_battery:
            raise ValueError("Invalid case: has_existing_battery and new_battery cannot both be True.")

        if consider_penalty:
            if has_existing_battery:
                if new_asset:
                    self.case = 11
                else:
                    self.case = 10
            elif new_battery:
                if new_asset:
                    self.case = 9
                else:
                    self.case = 8
            else:  # Neither existing nor new battery
                if new_asset:
                    self.case = 7
                else:
                    self.case = 6
        else:  # No penalty
            if has_existing_battery:
                if new_asset:
                    self.case = 5
                else:
                    self.case = 4
            elif new_battery:
                if new_asset:
                    self.case = 3
                else:
                    self.case = 2
            else:  # Neither existing nor new battery
                if new_asset:
                    self.case = 1
                else:
                    self.case = 0
                    
        return self.case

    ##--- A bunch of getter functon to access each parameterss ---##
    def get_ppa_terms(self):
        """
        Return the core PPA terms: coverage %, timestep, and contract length.

        Returns:
            tuple: (coveragepct, time_step, maturity)
        """
        return self.coveragepct, self.time_step, self.maturity
    
    def get_penalty_params(self):
        """
        Return the penalty factor and cap.

        Returns:
            tuple: (penalty_factor, penalty_cap)
        """
        return self.penalty_factor, self.penalty_cap
    
    def get_new_gen_params(self):
        """
        Get parameters of the new generation asset (if applicable).

        Returns:
            Generator object or None
        """
        if self.has_new_asset:
            return self.new_asset_params
        else:
            print('No new assets')
            
    def get_new_gen_ppa_price(self):
        """
        Get the PPA price offered for the new generation asset (if applicable).

        Returns:
            float or None
        """
        if self.has_new_asset:
            return self.new_asset_ppa_price
        else:
            print('No new assets')
    
    def get_new_bat_params(self):
        """
        Get parameters of the new battery asset (if applicable).

        Returns:
            Storage object or False
        """
        if self.has_new_bat:
            return self.new_bat_params
        else:
            return False