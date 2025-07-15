import streamlit as st
import pandas as pd
import numpy as np
import datetime

from data_processor import DataProcessor
from input_plotter import InputPlotter


if (gen_inputs and cus_inputs) != None:
    data_processor = DataProcessor(prods, demands)
    st.subheader('View and Select Generators & Customers')
    col1, col2 = st.columns([1,1])

    prod_view_df = data_processor.make_view_df(prods)
    demand_view_df = data_processor.make_view_df(demands)
    
    with col1: 
        st.write('Generator List')
        prod_edit_df = st.data_editor(prod_view_df, hide_index = True, column_config = {'Value':None})
    with col2:
        st.write('Customer List')
        demand_edit_df = st.data_editor(demand_view_df, hide_index = True, column_config = {'Value':None})
        
    # st.divider()
    
    st.write('')
    st.write('')
    st.write('')
    
    # Extract selected data dataframe
    selected_prods_df = prod_edit_df[prod_edit_df['View'] == True]
    selected_demands_df = demand_edit_df[demand_edit_df['View'] == True]

    if not selected_demands_df.empty and not selected_prods_df.empty:
        
        selected_prod_array = np.column_stack(selected_prods_df['Value']) # shape (17520, n_prods)
        selected_demand_array = np.column_stack(selected_demands_df['Value'])

        # Select matching method
        st.header('PPA Analysis', divider = 'blue')

        
        
        # Compute matching score

        matching_col_1, matching_col_2 = st.columns([1, 4])
        with matching_col_1:
            st.subheader('Matching Option')
            time_step_dict = {'Half-hourly': 1, 'Hourly': 2, 'Daily': 48} #, 'Weekly': 48*7}
            matching_input = st.selectbox('Select matching timestep',
                                    ('Half-hourly', 'Hourly', 'Daily'))#, 'Weekly'))
            matching_method = time_step_dict[matching_input]

    

        with matching_col_2:
            result_df = data_processor.compute_agg_matching(matching_method, selected_prod_array, selected_demand_array)
            matching_list = data_processor.compute_matching_list(matching_method, selected_prod_array, selected_demand_array)

          
            st.subheader('Result Analysis')
            st.metric(label = 'Max possible annual CFE % at the currenct matching timestep is', value = f"{result_df['agg_matching']}%", delta = None)
            
            # df_cons_res = pd.DataFrame([{'Total Consumption (GWh)': result_df['total_demands'],
            #                             'Matched Consumption (GWh)': result_df['matched_demands'],
            #                             'Unmatched Consumption (GWh)': result_df['unmatched_demands']}])
            # st.dataframe(df_cons_res)
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
    
        st.divider()
        st.subheader('Matching Plot')
        plotter = Plotter()
        
        # Input format: dataframe
        plotter.matching_plot(matching_method, selected_prods_df, selected_demands_df, matching_list)


    else:
        st.subheader('Please select at least ONE generation asset/customer to view matching analysis.')
    # ---------------- View gen / demand data -----------------------
    
    # selected_prods_df = pd.DataFrame(selected_prod_array, columns = selected_prods_df['Asset']).reset_index(drop = True).astype('float64')
    # selected_demands_df = pd.DataFrame(selected_demand_array, columns = selected_demands_df['Asset']).reset_index(drop = True).astype('float64')
    # print('')
    # print('')
    # print('')
            
    # print(selected_prods_df.columns)
    
    # st.subheader('View Generations / Consumption Data')
    # prod_view_start_col, prod_view_end_col, demand_view_start_col, demand_view_end_col = st.columns([1,1,1,1])
    # with prod_view_start_col:
    #     prod_view_start = st.date_input('Select gendata starting date', value = datetime.date(2025,1,1), min_value = datetime.date(2025, 1, 1),
    #                                     max_value = datetime.date(2025, 12, 31), key = 1)
    # with prod_view_end_col:
    #     prod_view_end = st.date_input('Select gendata ending date', value = datetime.date(2025,1,31), min_value = prod_view_start,
    #                                     max_value = datetime.date(2025, 12, 31), key = 2)
    # with demand_view_start_col:
    #     demand_view_start = st.date_input('Select cusdate starting date', value = datetime.date(2025,1,1), min_value = datetime.date(2025, 1, 1),
    #                                     max_value = datetime.date(2025, 12, 31), key = 3)
    # with demand_view_end_col:
    #     demand_view_end = st.date_input('Select cusdate ending date', value = datetime.date(2025,1,31), min_value = demand_view_start,
    #                                     max_value = datetime.date(2025, 12, 31), key =4)
    #     # st.line_chart(selected_demands_df.iloc[:100])
    # prod_view_col, demand_view_col = st.columns([1,1])
    
    # with prod_view_col:
    #     start_temp_1 = prod_view_start.timetuple().tm_yday-1
    #     end_temp_1 = prod_view_end.timetuple().tm_yday
        
    #     st.line_chart(selected_prods_df.iloc[start_temp_1* 48 : end_temp_1 * 48])
        
    # with demand_view_col:
    #     start_temp_2 = demand_view_start.timetuple().tm_yday-1
    #     end_temp_2 = demand_view_end.timetuple().tm_yday
        
    #     st.line_chart(selected_demands_df.iloc[start_temp_2 * 48 : end_temp_2 * 48])
        

    
    # reshaped_demands = selected_demand_array[:end].reshape(-1, view_time_step).sum(axis = 1)
    # reshaped_matching = self.realized_matching[:end].reshape(-1, view_time_step).sum(axis = 1)
    
    # total_prods_list = []
    # view_dict = {}
    # for i in self.generators.get_generator_list():
        
    #     temp_prod = i.get_gen_profile()
    #     reshaped_prod = temp_prod[:end].reshape(-1, view_time_step).sum(axis = 1)
    #     total_prods_list.append(reshaped_prod)
    #     temp_name = i.get_gen_name()
    #     view_dict[temp_name] = reshaped_prod

    # view_df = pd.DataFrame(view_dict)
    
    # # Bar Charts
    # # dd = prod_df.groupby(prod_df.index // 24).sum()

        
    # dates = pd.date_range(start="2025-01-01", periods=no_periods, freq=freq_dict[view_time_select])
    

    # df1 = pd.DataFrame({
    #     "date": dates,
    #     "value1": reshaped_demands[:int(no_periods)],
    # })
    
    
    # df3 = pd.DataFrame({
    #     "date": dates,
    #     "value1": reshaped_matching[:int(no_periods)],
    # })
    
    
    # df2 = view_df.copy()
    # df2['date'] = dates
    # df2_melted = df2.melt(id_vars=["date"], var_name="Asset", value_name="value")
    

    
    #         # Plot df1 as a line chart
    # line_chart = alt.Chart(df1).mark_line(color="red").encode(
    #     x=alt.X("date:T",  axis=alt.Axis(title="Date"), scale=alt.Scale(nice=True)),
    #     y=alt.Y("value1:Q"),
    #     tooltip=["date:T", "value1:Q"]
    # )
    
    
    # line_chart_2 = alt.Chart(df3).mark_line(color="blue").encode(
    #     x=alt.X("date:T",  axis=alt.Axis(title="Date"), scale=alt.Scale(nice=True)),
    #     y=alt.Y("value1:Q"),
    #     tooltip=["date:T", "value1:Q"]
    # )
    
    # # Plot df2 as a stacked area chart
    # stacked_area = alt.Chart(df2_melted).mark_bar(opacity=1, size = time_unit).encode(
    #     x=alt.X("date:T" , axis=alt.Axis(title="Date"),scale=alt.Scale(nice=True)),
    #     y=alt.Y("value:Q", stack="zero", axis=alt.Axis(title="Stacked Generation/Demand (MWh)")),  # Stacks the two columns
    #     color=alt.Color("Asset:N", scale=alt.Scale(scheme="greens")),  # Different colors for stacked values
    #     tooltip=["date:T", "Asset:N", "value:Q"]
    # )
    
    
    # # # Combine both charts
    # # combined_chart = (stacked_area + line_chart + line_chart_2).properties(
    # #     title=plot_title).interactive()
    
    
    # # Combine both charts
    # combined_chart = (stacked_area + line_chart + line_chart_2).properties(
    #     title=alt.TitleParams(
    #         text = plot_title, 
    #         anchor="middle", 
    #         orient="top",
    #         fontSize=16,
    #         offset = 10)).interactive()

    # # Display in Streamlit
    # st.altair_chart(combined_chart, use_container_width=True)

