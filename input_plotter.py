import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from .data_processor import DataProcessor


class InputPlotter:
    """
    A class to visualize generation and consumption matching.

    Attributes:
        storage (bool): Indicates whether battery storage is enabled.
    """
    def __init__(self, bool_storage):
        self.storage = bool_storage
        
    def matching_plot(self, matching_method, selected_prods_df, selected_demands_df):
        """
        Main function to plot a stacked bar graph for generations and line plots for consumption, 
        realised matching and bat_adjusted realised matching (if battery).
        TODO: Quite messy and chunky, could consider to split into helper functions to assist further developments

        Args:
            matching_method (int): Matching method granularity (e.g., 1 for hourly, 24 for daily).
            selected_prods_df (pd.DataFrame): DataFrame containing generator names and generation profile
            selected_demands_df (pd.DataFrame): DataFrame containing customer names and consumption profile
        """
        col1, col2 = st.columns([1, 4])
        with col1:
            if matching_method == 1:
                view_time_select = st.selectbox(
                    'Select view timestep',
                    ( 'Hourly', 'Daily', 'Weekly', 'Monthly'), key = 'asdada')
            else: 
                view_time_select = st.selectbox(
                    'Select view timestep',
                    ('Daily', 'Weekly', 'Monthly'), key = 'asdada')
                
            
        num_dict = {'Hourly': 1, 'Daily': 24, 'Weekly': 24 * 7, 'Monthly': 24*30}
        freq_dict = {'Hourly': 'h', 'Daily': 'D', 'Weekly': 'W', 'Monthly': 'ME'}

        view_time_step = num_dict[view_time_select]
        time_unit = None
        plot_title = None

        if view_time_step == 1:
            no_periods = 24*7
            time_unit = 3
            plot_title ='Hourly view, showing first 7 days'
        elif view_time_step == 24:
            no_periods = 30
            time_unit = 15
            plot_title ='Daily view, showing first 30 days'
        elif view_time_step == 24*7:
            no_periods = 52
            time_unit = 10
            plot_title ='Weekly view, showing 52 weeks'
        elif view_time_step == 24*30:
            no_periods = 12
            time_unit = 30
            plot_title ='Monthly view, showing 12 months'
        
        end = int(no_periods * view_time_step)
        
        # Hard code here to enforce all 17520 periods in a year
        if view_time_step == 24*30:
            end = 8760
            
        # Get consumtpion and productions in 2d nparray (n, 8760)
        selected_prod_array = np.column_stack(selected_prods_df['Value']).T 
        selected_demand_array = np.column_stack(selected_demands_df['Value']).T

        # Battery case, get battery adjusted aggregated generation, 1d nparray with shape (8760,)
        if self.storage:
            temp_agg_prods =  selected_prod_array.sum(axis = 0)
            agg_demands = selected_demand_array.sum(axis = 0)
            
            temp_dp = DataProcessor(temp_agg_prods, agg_demands, self.storage)
            modified_agg_prods = temp_dp.get_bat_modified_prod(matching_method, temp_agg_prods, agg_demands)
            bat_agg_prods = modified_agg_prods.reshape(-1, matching_method).sum(axis = 1)
        
        # Aggregate generations and demands, 1d nparray with shape (8760,)
        agg_prods = selected_prod_array.sum(axis = 0).reshape(-1, matching_method).sum(axis = 1)
        agg_demands = selected_demand_array.sum(axis = 0).reshape(-1, matching_method).sum(axis = 1)

        ###################### Aggregation of Viewing Prods and Demands ######################
        # Formatting the time-aggregated generations: hourly, daily, weekly, monthly
        # reshaped_prods, reshaped_demands: 1d array (x,), x depends on matching time step and viewing timestep
        
        total_prods_list = []
        view_dict = {}
        days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31] # TODO: Some hardcoding, not considering leap year with 29-day Feb
        acc_days_monthly = [31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]
        
        ##### Reshaping of Prods #####
        for i in range(selected_prod_array.shape[0]): # For each asset
            # Special case for monthly view
            if view_time_step == 24*30:
                reshaped_prods = []
                for j in range(12):
                    if j == 0:
                        start_temp = 0
                        end_temp = 31*24
                    else:
                        start_temp = 24 * acc_days_monthly[j-1]
                        end_temp = 24 * acc_days_monthly[j]
                    
                    monthly_prod = np.sum(selected_prod_array[i][start_temp:end_temp])
                    reshaped_prods.append(monthly_prod)
                reshaped_prods = np.array(reshaped_prods)
    
            # Other cases: half-hourly/hourly/daily/weekly
            else:
                reshaped_prods = selected_prod_array[i][:end].reshape(-1, view_time_step).sum(axis = 1)
      
            total_prods_list.append(reshaped_prods)
            temp_name = selected_prods_df['Name'].iloc[i]
            view_dict[temp_name] = reshaped_prods

        prod_view_df = pd.DataFrame(view_dict)

        ##### Reshaping Demands #####
        
        if view_time_step == 24*30:
            temp_sum_demands = selected_demand_array.sum(axis = 0)
            reshaped_demands = []
            for j in range(12):
                if j == 0:
                    start_temp = 0
                    end_temp = 31*24
                else:
                    start_temp = 24 * acc_days_monthly[j-1]
                    end_temp = 24 * acc_days_monthly[j]
                
                monthly_demand = np.sum(temp_sum_demands[start_temp:end_temp])
                reshaped_demands.append(monthly_demand)
            reshaped_demands = np.array(reshaped_demands)
        else:
            reshaped_demands = selected_demand_array.sum(axis = 0)[:end].reshape(-1, view_time_step).sum(axis = 1)[:no_periods]  
        
        # TODO: Here I hardcoded the x axis of viewing plot to start at 2025-01-01, should improve
        dates = pd.date_range(start="2025-01-01", periods=no_periods, freq=freq_dict[view_time_select])

        # Initialize list to contain realized matching
        agg_realized_prov = []
        
        # First to discuss the case with Battery Smoothing
        if self.storage:
            bat_agg_realized_prov = [] 

            ###################### Hourly matching ######################
            if matching_method == 1: 
                if view_time_step == 1:
                    for i in range(no_periods):
                        temp_prov = min(agg_prods[i], agg_demands[i])
                        agg_realized_prov.append(float(temp_prov))
                        
                        temp_bat_prov = min(bat_agg_prods[i], agg_demands[i])
                        bat_agg_realized_prov.append(float(temp_bat_prov))
        
                        
                elif view_time_step == 24:
                    # no_periods = 30, daily view, total 30 days
                    for i in range(no_periods):
                        temp_prov = 0
                        temp_demand = 0
                        temp_bat_prov = 0
                        for j in range(24):
                            temp_prov += min(agg_prods[i*24 + j], agg_demands[i*24 + j])
                            temp_demand += agg_demands[i*24 + j]
                            
                            temp_bat_prov += min(bat_agg_prods[i*24 + j], agg_demands[i*24 + j])
                        
                        bat_agg_realized_prov.append(float(temp_bat_prov))
                        agg_realized_prov.append(float(temp_prov))
                        
                elif view_time_step == 24*7:
                    for i in range(no_periods):
                        temp_prov = 0
                        temp_bat_prov = 0
                        temp_demand = 0
                        for j in range(24*7):
                            temp_prov += min(agg_prods[i*(24*7) + j], agg_demands[i*(24*7) + j])
                            temp_bat_prov += min(bat_agg_prods[i*(24*7) + j], agg_demands[i*(24*7) + j])
                            temp_demand += agg_demands[i*(24*7)+ j]
                
                        agg_realized_prov.append(float(temp_prov))
                        bat_agg_realized_prov.append(float(temp_bat_prov))
                        
                elif view_time_step == 24*30:
                    for i in range(no_periods):
                        temp_prov = 0
                        temp_bat_prov = 0
                        if i == 0:
                            prev_days = 0
                        else:
                            prev_days =  acc_days_monthly[i-1]
                            
                        for j in range(days_per_month[i]* 24):
                            actual_prov = min(agg_prods[prev_days*24 + j], agg_demands[prev_days*24 + j])
                            actual_bat_prov = min(bat_agg_prods[prev_days*24 + j], agg_demands[prev_days*24 + j])
                            temp_prov += actual_prov
                            temp_bat_prov += actual_bat_prov
                        agg_realized_prov.append(float(temp_prov))
                        bat_agg_realized_prov.append(float(temp_bat_prov))
            
            # # Daily matching case
            # matching method = 48, viewing step =  48
            ###################### Daily matching ######################
            else: 
                if view_time_step == 24:
                    for i in range(no_periods):
                        temp_prov = min(agg_prods[i], agg_demands[i])
                        agg_realized_prov.append(float(temp_prov))
                        
                        temp_bat_prov = min(bat_agg_prods[i], agg_demands[i])
                        bat_agg_realized_prov.append(float(temp_bat_prov))
                        
                elif view_time_step == 24*7:
                    for i in range(no_periods):
                        temp_prov = 0
                        temp_bat_prov = 0
                        temp_demand = 0
                        for j in range(7):
                            temp_prov += min(agg_prods[i*7 + j], agg_demands[i*7 + j])
                            temp_bat_prov += min(bat_agg_prods[i*7 + j], agg_demands[i*7 + j])
                            temp_demand += agg_demands[i*7+ j]
                
                        agg_realized_prov.append(float(temp_prov))
                        bat_agg_realized_prov.append(float(temp_bat_prov))
                        
                elif view_time_step == 24*30:
                    for i in range(no_periods):
                        temp_prov = 0
                        temp_bat_prov = 0
                        if i == 0:
                            prev_days = 0
                        else:
                            prev_days =  acc_days_monthly[i-1]
                            
                        for j in range(days_per_month[i]):
                            actual_prov = min(agg_prods[prev_days + j], agg_demands[prev_days + j])
                            actual_bat_prov = min(bat_agg_prods[prev_days + j], agg_demands[prev_days + j])
                            temp_prov += actual_prov
                            temp_bat_prov += actual_bat_prov
                        agg_realized_prov.append(float(temp_prov))
                        bat_agg_realized_prov.append(float(temp_bat_prov))   

            # Prepare df for plotting
            df1 = pd.DataFrame({
                "date": dates,
                "Consumption": reshaped_demands[:int(no_periods)],
                "Matched Consumption":agg_realized_prov,
                "Bat-adjusted Matched Consumption": bat_agg_realized_prov,
                "Label1": 'Consumption',
                'Label2': 'Matched Consumption',
                'Label3': 'Bat-adjusted Matched Consumption'
            })
            
            df1_melted = df1.melt(id_vars="date", 
                      value_vars=["Consumption", "Matched Consumption", "Bat-adjusted Matched Consumption"],
                      var_name="Label", value_name="value")
            df_consumption = df1_melted[df1_melted["Label"] == "Consumption"]
            df_others = df1_melted[df1_melted["Label"] != "Consumption"]
                        
            df2 = prod_view_df.copy()
            df2['date'] = dates
            df2_melted = df2.melt(id_vars=["date"], var_name="Asset", value_name="value")
            
            # Plot df2 as a stacked area chart of generations
            stacked_area = alt.Chart(df2_melted).mark_bar(opacity=1, size = time_unit).encode(
                x=alt.X("date:T" , axis=alt.Axis(title="Date"),scale=alt.Scale(nice=True)),
                y=alt.Y("value:Q", stack="zero", axis=alt.Axis(title="Stacked Generation/Demand (MWh)")),  # Stacks the two columns
                color=alt.Color("Asset:N", scale=alt.Scale(scheme="greens")),  # Different colors for stacked values
                tooltip=["date:T", "Asset:N", "value:Q"]
            )

            # Plot: Fixed red line for consumption
            consumption_line = alt.Chart(df_consumption).mark_line().encode(
                x="date:T",
                y="value:Q",
                color=alt.Color("Label:N", scale=alt.Scale(domain=["Consumption"], range=["red"])),
                tooltip=["date:T", "Label:N", "value:Q"]
            )
            # Plot: Scheme-colored lines for matched consumption and bat-adjusted matched consumption
            label_order = ["Matched Consumption", "Bat-adjusted Matched Consumption"]
            others_line = alt.Chart(df_others).mark_line().encode(
                x="date:T",
                y="value:Q",
                color=alt.Color("Label:N", scale=alt.Scale(scheme="blues", domain=label_order)),
                tooltip=["date:T", "Label:N", "value:Q"]
            )

            # Combined chart
            combined_chart = (stacked_area + consumption_line + others_line).properties(
                title=alt.TitleParams(
                    text=plot_title,
                    anchor="middle",
                    orient="top",
                    fontSize=16,
                    offset=10
                )
            ).interactive().resolve_scale(color='independent')
            
            with col2:
                st.altair_chart(combined_chart, use_container_width=True)

        # No battery matching plot
        else:
            ###################### Hourly matching ######################
            if matching_method == 1: 
                if view_time_step == 1:
                    for i in range(no_periods):
                        temp_prov = min(agg_prods[i], agg_demands[i])
                        agg_realized_prov.append(float(temp_prov))
                        
                elif view_time_step == 24:
                    # no_periods = 30, daily view, total 30 days
                    for i in range(no_periods):
                        temp_prov = 0
                        temp_demand = 0
                        for j in range(24):
                            temp_prov += min(agg_prods[i*24 + j], agg_demands[i*24 + j])
                            temp_demand += agg_demands[i*24 + j]
                
                        agg_realized_prov.append(float(temp_prov))
                        
                elif view_time_step == 24*7:
                    for i in range(no_periods):
                        temp_prov = 0
                        temp_demand = 0
                        for j in range(24*7):
                            temp_prov += min(agg_prods[i*(24*7) + j], agg_demands[i*(24*7) + j])
                            temp_demand += agg_demands[i*(24*7)+ j]
                
                        agg_realized_prov.append(float(temp_prov))
                        
                elif view_time_step == 24*30:
                    for i in range(no_periods):
                        temp_prov = 0
                        if i == 0:
                            prev_days = 0
                        else:
                            prev_days =  acc_days_monthly[i-1]
                            
                        for j in range(days_per_month[i]* 24):
                            actual_prov = min(agg_prods[prev_days*24 + j], agg_demands[prev_days*24 + j])
                            temp_prov += actual_prov
                        agg_realized_prov.append(float(temp_prov))
            
            # # Daily matching case
            # matching method = 48, viewing step =  48
            ###################### Daily matching ######################
            else: 
                if view_time_step == 24:
                    for i in range(no_periods):
                        temp_prov = min(agg_prods[i], agg_demands[i])
                        agg_realized_prov.append(float(temp_prov))
                        
                elif view_time_step == 24*7:
                    for i in range(no_periods):
                        temp_prov = 0
                        temp_demand = 0
                        for j in range(7):
                            temp_prov += min(agg_prods[i*7 + j], agg_demands[i*7 + j])
                            temp_demand += agg_demands[i*7+ j]
                
                        agg_realized_prov.append(float(temp_prov))
                        
                elif view_time_step == 24*30:
                    for i in range(no_periods):
                        temp_prov = 0
                        if i == 0:
                            prev_days = 0
                        else:
                            prev_days =  acc_days_monthly[i-1]
                            
                        for j in range(days_per_month[i]):
                            actual_prov = min(agg_prods[prev_days + j], agg_demands[prev_days + j])
                            temp_prov += actual_prov
                        agg_realized_prov.append(float(temp_prov))
            
            # Prepare df for plotting 
            # TODO: Should have helper function for concise plotting algo
            df1 = pd.DataFrame({
                "date": dates,
                "consumption": reshaped_demands[:int(no_periods)],
                "real_matching":agg_realized_prov,
                "Label1": 'Consumption',
                'Label2': 'Matched Consumption'
            })
            
            df2 = prod_view_df.copy()
            df2['date'] = dates
            df2_melted = df2.melt(id_vars=["date"], var_name="Asset", value_name="value")
                        
            # Plot df2 as a stacked area chart of generations
            stacked_area = alt.Chart(df2_melted).mark_bar(opacity=1, size = time_unit).encode(
                x=alt.X("date:T" , axis=alt.Axis(title="Date"),scale=alt.Scale(nice=True)),
                y=alt.Y("value:Q", stack="zero", axis=alt.Axis(title="Stacked Generation/Demand (MWh)")),  # Stacks the two columns
                color=alt.Color("Asset:N", scale=alt.Scale(scheme="greens")),  # Different colors for stacked values
                tooltip=["date:T", "Asset:N", "value:Q"]
            )
            
            line_chart = alt.Chart(df1).mark_line(color="red").encode(
                x=alt.X("date:T",  axis=alt.Axis(title="Date"), scale=alt.Scale(nice=True)),
                y=alt.Y("consumption:Q"),
                color=alt.Color("Label1:N", scale=alt.Scale(scheme="set1")),
                tooltip=["date:T", "consumption:Q", 'Label1:N']
            )
            
            line_chart_2 = alt.Chart(df1).mark_line(color="blue").encode(
                x=alt.X("date:T",  axis=alt.Axis(title="Date"), scale=alt.Scale(nice=True)),
                y=alt.Y("real_matching:Q"),
                # color = 'Label2:N',
                color=alt.Color("Label2:N", scale=alt.Scale(scheme="category10")),
                tooltip=["date:T", "real_matching:Q", 'Label2:N']
            )

            # Combine both charts
            combined_chart = (stacked_area + line_chart + line_chart_2).properties(
                title=alt.TitleParams(
                    text = plot_title, 
                    anchor="middle", 
                    orient="top",
                    fontSize=16,
                    offset = 10)).interactive().resolve_scale(color='independent')

            # Display in Streamlit
            with col2:
                st.altair_chart(combined_chart, use_container_width=True)

