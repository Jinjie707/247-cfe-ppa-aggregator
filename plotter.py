import streamlit as st
import altair as alt
import pandas as pd
import numpy as np

from datetime import datetime
from datetime import timedelta

class Plotter:
    def __init__(self, case_type, processed_inputs, ppa_params, output, T):
        self.case = case_type
        
        self.ex_bat = processed_inputs['Existing Battery'][1] 
        if ppa_params.get_new_bat_params(): # Returns None if no new bat
            self.ex_bat = ppa_params.get_new_bat_params()

        self.input_proddatas =  processed_inputs['Generators'].get_all_gen_profiles()
        self.input_demanddatas = processed_inputs['Customers'].get_all_cus_profiles()
        
        self.perc, self.time_step, self.maturity = ppa_params.get_ppa_terms()
        
        self.gen_names = processed_inputs['Generators'].get_all_gen_names()
        
        self.agg_prod = processed_inputs['Generators'].get_agg_gen_profile()
        self.agg_demand = processed_inputs['Customers'].get_agg_cus_profile()
             
        self.beta = output['beta']
        
        self.v_total = output['v_total']
        self.v_renewable = output['v_renewable']
        self.v_buy = output['v_buy']
        
        self.rn_ratio = output['renewable utilization ratio']
        self.opt_price = output['optimal price']
        self.mkt_rev = output['market sell']
    
        self.output = output
        
        self.T = T
        self.start_date = datetime(2025, 1, 1, 00, 00)
        self.end_date =  datetime(2025, 12, 31, 23, 00)
        self.X = [self.start_date + i* timedelta(hours=1) for i in range(int((self.end_date - self.start_date)/timedelta(hours=1)))][:self.T]

    ##########################################################
    # A bunch (12 to be precise) of functions that calls result plot action for each case
    # Each calls some helper functions
    def plot(self):
        self._helper_start_plotting()
        
        if self.case == 0:
            self.plot_0()
        elif self.case == 1:
            self.plot_1()
        elif self.case == 2:
            self.plot_2()
        elif self.case == 3:
            self.plot_3()
        elif self.case == 4:
            self.plot_4()
        elif self.case == 5:
            self.plot_5()
        elif self.case == 6:
            self.plot_6()
        elif self.case == 7:
            self.plot_7()
        elif self.case == 8:
            self.plot_8()
        elif self.case == 9:
            self.plot_9()
        elif self.case == 10:
            self.plot_10()
        elif self.case == 11:
            self.plot_11()
        st.divider()

    ##########################################################
    # A bunch of helper functions that will be called multiple times for diff cases
    # Each displays a metric, a table or a plot
    
    # Displays section header
    def _helper_start_plotting(self):
        st.subheader('Results and Plots', divider = 'blue')
    
    # Displays optimal PPA price metric
    def _helper_opt_price_metric(self):
        st.metric(label = 'Optimal PPA price is', value = f"{np.round(self.opt_price, 2)} $/MWh", delta = None)

    # Displays optimal new gen cap metric
    def _helper_opt_gen_cap_metric(self):
        new_gen_cap = self.output['optimal capacity']  
        st.metric(label = 'Optimal generator capacity is', value = f"{np.round(new_gen_cap, 2)} MW", delta = None)
    
    # Displays optimal new bat cap metric
    def _helper_opt_bat_cap_metric(self):
        new_bat_cap = self.ex_bat.get_bat_cap()
        st.metric(label = 'Optimal battery capacity is', value = f"{np.round(new_bat_cap, 2)} MWh", delta = None)
    
    # Displays usage vs. prod line plot for each gen asset
    def _helper_usage_vs_prod_plot(self):
        st.write('PPA Renewable Provision vs. Total Renewable Production')
        col_a, col_b = st.columns(2)
        
        T = self.T
        usage_vs_production_df = []
        for i in range(len(self.input_proddatas)):
            temp = pd.DataFrame(
                {'Timestep':self.X,
                'Production': self.input_proddatas[i][:T],
                'Usage': np.multiply(np.array(self.beta[i][:T]), np.array(self.input_proddatas[i][:T])),
                }
            )
            usage_vs_production_df.append(temp)
        
        for i in range(len(self.input_proddatas)):
            if i %2 == 0:
                with col_a:
                    st.caption('Renewable asset: ' + self.gen_names[i])
                    st.line_chart(
                        usage_vs_production_df[i],
                        x="Timestep",
                        y=['Production', 'Usage'],
                        x_label = 'Time',
                        y_label = 'Energy (MWh)')
            else:
                with col_b:
                    st.caption('Renewable asset: ' + self.gen_names[i])
                    st.line_chart(
                        usage_vs_production_df[i],
                        x="Timestep",
                        y=['Production', 'Usage'],
                        x_label = 'Time',
                        y_label = 'Energy (MWh)')
    
    # Displays renewable util table
    def _helper_renewable_util_ratio_df(self):  
        st.divider()
        st.write('Generator Utilization Ratio')
        temp_res = {}
        for i in range(len(self.input_proddatas)):
            temp = np.sum(np.multiply(np.array(self.beta[i]), np.array(self.input_proddatas[i]))) / np.sum(self.input_proddatas[i])
            temp_res[self.gen_names[i]] = np.round(temp, 3)
        st.dataframe(pd.DataFrame.from_dict(temp_res, orient="index", columns=["util_ratio"]))
    
    # Displays renewable and bat util table
    def _helper_renewable_bat_util_ratio_df(self):  
        st.divider()
        st.write('Asset Utilization Ratio')
        temp_res = {}
        for i in range(len(self.input_proddatas)):
            temp = np.sum(np.multiply(np.array(self.beta[i]), np.array(self.input_proddatas[i]))) / np.sum(self.input_proddatas[i])
            temp_res[self.gen_names[i]] = np.round(temp, 3)
            
        bat_u_c = self.ex_bat.get_u_c()
        bat_u_d = self.ex_bat.get_u_d()
        bat_cap = self.ex_bat.get_bat_cap()
        bat_max_cycle = self.ex_bat.get_max_cycle()
        temp_res['Battery'] = np.round(np.sum(bat_u_c) / (365*bat_cap*bat_max_cycle), 3)
        
        st.dataframe(pd.DataFrame.from_dict(temp_res, orient="index", columns=["util_ratio"]))

    # Displays PPA energy source bat plot
    def _helper_ppa_component_plot(self):
        demands = np.sum(self.agg_demand) * self.perc
        
        asset_names = self.gen_names
        asset_prods = self.input_proddatas
        asset_betas = self.beta

        res_dict = {}
        for i in range(len(asset_names)):
            res_dict[asset_names[i]] = np.round(np.sum(asset_prods[i] * asset_betas[i]) / demands, 3)

        perc_asset_df = pd.DataFrame(list(res_dict.items()), columns=["Asset", "Percentage"])
        
        st.divider()
        st.write('Decompositon of Total PPA Energy Source')
        st.bar_chart(perc_asset_df, x="Asset", y="Percentage")
    
    # Displays PPA energy source table
    def _helper_ppa_component_df(self):
        st.divider()
        st.write('PPA Energy Source Component')
        demands = np.sum(self.agg_demand) * self.perc 
        asset_names = self.gen_names
        asset_prods = self.input_proddatas
        asset_betas = self.beta
        
        res_dict = {}
        for i in range(len(asset_names)):
            res_dict[asset_names[i]] = np.round(np.sum(asset_prods[i] * asset_betas[i]) / demands, 3)
        
        st.dataframe(pd.DataFrame.from_dict(res_dict, orient="index", columns=["weight"]))
    
    # Displays PPA energy source with penalty bar plot
    def _helper_ppa_component_with_pen_plot(self):
        demands = np.sum(self.agg_demand) * self.perc
        
        asset_names = self.gen_names
        asset_prods = self.input_proddatas
        asset_betas = self.beta

        mkt_purchase = self.v_buy
        
        res_dict = {}
        for i in range(len(asset_names)):
            res_dict[asset_names[i]] = np.round(np.sum(asset_prods[i] * asset_betas[i]) / demands, 3)
        
        res_dict['Market Purchase'] = np.round(np.sum(mkt_purchase) / demands, 3)
        perc_asset_df = pd.DataFrame(list(res_dict.items()), columns=["Asset", "Percentage"])
        
        st.divider()
        st.write('Decompositon of Total PPA Energy Source')
        st.bar_chart(perc_asset_df, x="Asset", y="Percentage")
    
    # Displays PPA energy source with Penalty table
    def _helper_ppa_component_with_pen_df(self):
        st.divider()
        st.write('PPA Energy Source Component')
        demands = np.sum(self.agg_demand) * self.perc
        asset_names = self.gen_names
        asset_prods = self.input_proddatas
        asset_betas = self.beta

        mkt_purchase = self.v_buy
        
        res_dict = {}
        for i in range(len(asset_names)):
            res_dict[asset_names[i]] = np.round(np.sum(asset_prods[i] * asset_betas[i]) / demands, 3)
        
        res_dict['Market Purchase'] = np.round(np.sum(mkt_purchase) / demands, 3)
    
        st.dataframe(pd.DataFrame.from_dict(res_dict, orient="index", columns=["weight"]))

    # Displays total consumption vs. total production line plot
    def _helper_demand_prod_plot(self):
        T = self.T
        plot_df = pd.DataFrame(
            {
                "Timestep": self.X,
                "Demand": self.agg_demand[:T],
                "Production": self.agg_prod[:T],
            }
        )
        st.divider()
        st.write('Demand vs. PPA Renewable Provision')
        st.line_chart(plot_df,
            x="Timestep",
            y=["Demand", "Production"], 
            x_label = 'Time',
            y_label = 'Energy (MWh)')
        
    # Displays total consumption vs. total production vs. penalty purchase line plot
    def _helper_demand_prod_pen_plot(self):
        T = self.T
        plot_df = pd.DataFrame(
            {
                "Timestep": self.X,
                "Demand": self.agg_demand[:T],
                "Penalty Coverage": self.v_buy[:T],
                "Production": self.agg_prod[:T],
            }
        )
        
        st.divider()
        st.write('Demand vs. PPA Renewable Provision vs. Penalty Coverage')
        st.line_chart(plot_df,
            x="Timestep",
            y=["Demand", "Penalty Coverage", "Production"], 
            x_label = 'Time',
            y_label = 'Energy (MWh)')
    
    # Displays total consumption vs. total production vs. Bat LOE line plot
    def _helper_demand_prod_loe_plot(self):
        T = self.T
        bat_soc = self.ex_bat.get_bat_soc()
        
        st.divider()
                    
        bat_soc_df = pd.DataFrame({'Timestep': self.X,
                'BESS LOE': np.array(bat_soc)[:T],
                })
        st.write('BESS Level of Energy (LOE)')
        st.line_chart(bat_soc_df,
            x="Timestep",
            y=["BESS LOE"], 
            x_label = 'Time',
            y_label = 'Energy (MWh)')
        
        st.divider()
        st.write('Demand vs. PPA Renewable Provision vs. BESS LOE')
        
        demand_vs_production_vs_soc_df = pd.DataFrame(
                {'Timestep': self.X,
                'Production': self.agg_prod[:T],
                'Demand': self.agg_demand[:T],
                'BESS LOE': np.array(bat_soc)[:T],
                }
            )
            
        temp_df = demand_vs_production_vs_soc_df.melt(id_vars=["Timestep"], 
                                                value_vars=["Demand", "Production", "BESS LOE"], 
                                                var_name="Category", 
                                                value_name="Energy")

        chart = alt.Chart(temp_df).mark_line().encode(
            x=alt.X("Timestep", title="Time"),
            y=alt.Y("Energy", title="Energy (MWh)"),
            color=alt.Order("Category", sort=["Demand", "Production", "BESS LOE"])  # preserves your order
        ).properties(
            width=700,
            height=400
        )

        st.altair_chart(chart, use_container_width=True)    
        
    # Displays total consumption vs. total production vs. LOE vs. penalty purchase line plot  
    def _helper_demand_prod_loe_pen_plot(self):
        T = self.T
        bat_soc = self.ex_bat.get_bat_soc()
        
        st.divider()
                    
        bat_soc_df = pd.DataFrame({'Timestep': self.X,
                'BESS LOE': np.array(bat_soc)[:T],
                })
        st.write('BESS Level of Energy (LOE)')
        st.line_chart(bat_soc_df,
            x="Timestep",
            y=["BESS LOE"], 
            x_label = 'Time',
            y_label = 'Energy (MWh)')
        
        st.divider()
        st.write('Demand vs. PPA Renewable Provision vs. BESS LOE vs. Penalty Coverage')
        
        demand_vs_production_vs_soc_df = pd.DataFrame(
                {'Timestep': self.X,
                'Production': self.agg_prod[:T],
                "Penalty Coverage": self.v_buy[:T],
                'Demand': self.agg_demand[:T],
                'BESS LOE': np.array(bat_soc)[:T],
                }
            )
            
        temp_df = demand_vs_production_vs_soc_df.melt(id_vars=["Timestep"], 
                                                value_vars=["Demand", "Production", 'Penalty Coverage', "BESS LOE"], 
                                                var_name="Category", 
                                                value_name="Energy")

        chart = alt.Chart(temp_df).mark_line().encode(
            x=alt.X("Timestep", title="Time"),
            y=alt.Y("Energy", title="Energy (MWh)"),
            color=alt.Order("Category", sort=["Demand", "Production", "Penalty Coverage", "BESS LOE"])  # preserves your order
        ).properties(
            width=700,
            height=400
        )

        st.altair_chart(chart, use_container_width=True)    


    ##########################################################
    # A bunch (12 to be precise) of functions that calls result plot action for each case
    # Each calls some helper functions
    #   Left column: metrics and tables
    #       Metrics: optimal PPA price, optimal new gen cap and optimal new bat cap
    #       Tables: gen and bat util ratio and market purchase ratio
    #   Right column: plots
    #       1. gen usage vs. prod plot
    #       2. total consumption vs. total production (vs. bat LOE) plot
    #       3. PPA component (gen, bat & market purchase) plot
    
    def plot_0(self):
        col1, col2 = st.columns([1, 3])
        with col1:
            self._helper_opt_price_metric()
            self._helper_renewable_util_ratio_df()
            self._helper_ppa_component_df()
        with col2:
            self._helper_usage_vs_prod_plot()
            self._helper_demand_prod_plot()
            self._helper_ppa_component_plot()
    
    def plot_1(self):
        col1, col2 = st.columns([1, 3])
        with col1:
            self._helper_opt_price_metric()
            self._helper_opt_gen_cap_metric()
            self._helper_renewable_util_ratio_df()
            self._helper_ppa_component_df()
        with col2:
            self._helper_usage_vs_prod_plot()
            self._helper_demand_prod_plot()
            self._helper_ppa_component_plot()
    
    def plot_2(self):
        col1, col2 = st.columns([1, 3])
        with col1:
            self._helper_opt_price_metric()
            self._helper_opt_bat_cap_metric()
            self._helper_renewable_bat_util_ratio_df()
            self._helper_ppa_component_df()
        with col2:
            self._helper_usage_vs_prod_plot()
            self._helper_demand_prod_loe_plot() 
            self._helper_ppa_component_plot()

            
    def plot_3(self):
        col1, col2 = st.columns([1, 3])
        with col1:
            self._helper_opt_price_metric()
            self._helper_opt_gen_cap_metric()
            self._helper_opt_bat_cap_metric()   
            self._helper_renewable_bat_util_ratio_df()
            self._helper_ppa_component_df()
        with col2:
            self._helper_usage_vs_prod_plot()
            self._helper_demand_prod_loe_plot()
            self._helper_ppa_component_plot()
            
    def plot_4(self):
        col1, col2 = st.columns([1, 3])
        with col1:
            self._helper_opt_price_metric()
            self._helper_renewable_bat_util_ratio_df()
            self._helper_ppa_component_df()
        with col2:
            self._helper_usage_vs_prod_plot()
            self._helper_demand_prod_loe_plot()
            self._helper_ppa_component_plot()
        
    def plot_5(self):
        col1, col2 = st.columns([1, 3])
        with col1:
            self._helper_opt_price_metric()
            self._helper_opt_gen_cap_metric()
            self._helper_renewable_bat_util_ratio_df()
            self._helper_ppa_component_df()
        with col2:
            self._helper_usage_vs_prod_plot() 
            self._helper_demand_prod_loe_plot() 
            self._helper_ppa_component_plot()
            
    def plot_6(self):
        col1, col2 = st.columns([1, 3])
        with col1:
            self._helper_opt_price_metric()
            self._helper_renewable_util_ratio_df()
            self._helper_ppa_component_with_pen_df()
        with col2:
            self._helper_usage_vs_prod_plot()
            self._helper_demand_prod_pen_plot()
            self._helper_ppa_component_with_pen_plot()
        
    def plot_7(self):
        col1, col2 = st.columns([1, 3])
        with col1:
            self._helper_opt_price_metric()
            self._helper_opt_gen_cap_metric()
            self._helper_renewable_util_ratio_df()
            self._helper_ppa_component_with_pen_df()
        with col2:
            self._helper_usage_vs_prod_plot()
            self._helper_demand_prod_pen_plot()
            self._helper_ppa_component_with_pen_plot()
            
    def plot_8(self):
        col1, col2 = st.columns([1, 3])
        with col1:
            self._helper_opt_price_metric()
            self._helper_opt_bat_cap_metric()
            self._helper_renewable_bat_util_ratio_df()
            self._helper_ppa_component_with_pen_df()
        with col2:
            self._helper_usage_vs_prod_plot()
            self._helper_demand_prod_loe_pen_plot()
            self._helper_ppa_component_with_pen_plot()
            
    def plot_9(self):
        col1, col2 = st.columns([1, 3])
        with col1:
            self._helper_opt_price_metric()
            self._helper_opt_gen_cap_metric()
            self._helper_opt_bat_cap_metric()
            self._helper_renewable_bat_util_ratio_df()
            self._helper_ppa_component_with_pen_df()
        with col2:
            self._helper_usage_vs_prod_plot()
            self._helper_demand_prod_loe_pen_plot() 
            self._helper_ppa_component_with_pen_plot()
            
            
    def plot_10(self):
        col1, col2 = st.columns([1, 3])
        with col1:
            self._helper_opt_price_metric()
            self._helper_renewable_bat_util_ratio_df()
            self._helper_ppa_component_with_pen_df()
        with col2:
            self._helper_usage_vs_prod_plot()
            self._helper_demand_prod_loe_pen_plot()
            self._helper_ppa_component_with_pen_plot()
            
    def plot_11(self):
        col1, col2 = st.columns([1, 3])
        with col1:
            self._helper_opt_price_metric()
            self._helper_opt_gen_cap_metric()
            self._helper_renewable_bat_util_ratio_df()
            self._helper_ppa_component_with_pen_df()
        with col2:
            self._helper_usage_vs_prod_plot()
            self._helper_demand_prod_loe_pen_plot()
            self._helper_ppa_component_with_pen_plot()
            
########### Unused Function 1 ##################

    # Another version of demand_prod_loe plot with diff line colors 
    # def _helper_demand_prod_loe_plot(self):
    #     T = self.T
    #     bat_soc = self.ex_bat.get_bat_soc()
        
    #     st.divider()
                    
    #     bat_soc_df = pd.DataFrame({'Timestep': self.X,
    #             'BESS LOE': np.array(bat_soc)[:T],
    #             })
    #     st.write('BESS Level of Energy (LOE)')
    #     st.line_chart(bat_soc_df,
    #         x="Timestep",
    #         y=["BESS LOE"], 
    #         x_label = 'Time',
    #         y_label = 'Energy (MWh)')
        
    #     st.divider()
    #     st.write('Demand vs. PPA Renewable Provision vs. BESS LOE')
        
    #     demand_vs_production_vs_soc_df = pd.DataFrame(
    #             {'Timestep': self.X,
    #             'Production': self.agg_prod[:T],
    #             'Demand': self.agg_demand[:T],
    #             'BESS LOE': np.array(bat_soc)[:T],
    #             }
    #         )
            
    #     st.line_chart(demand_vs_production_vs_soc_df,
    #         x="Timestep",
    #         y=["Demand", "Production", "BESS LOE"], 
    #         x_label = 'Time',
    #         y_label = 'Energy (MWh)')
            
########### Unused Function 2 ##################
    ## Not using this plot as the idea is to not include bat in PPA energy sourcse consideration
    # def _helper_ppa_component_with_pen_bat_plot(self):
    #     demands = np.sum(self.agg_demand) * self.perc
        
    #     asset_names = self.gen_names
    #     asset_prods = self.input_proddatas
    #     asset_betas = self.beta

    #     mkt_purchase = self.v_buy
        
    #     bat_u_d = self.ex_bat.get_u_d()
        
    #     res_dict = {}
    #     for i in range(len(asset_names)):
    #         res_dict[asset_names[i]] = np.round(np.sum(asset_prods[i] * asset_betas[i]) / demands, 3)
        
    #     res_dict['Market Purchase'] = np.round(np.sum(mkt_purchase) / demands, 3)
    #     res_dict['Battery Discharging'] = np.round(np.sum(bat_u_d) / demands, 3)
        
    #     perc_asset_df = pd.DataFrame(list(res_dict.items()), columns=["Energy Source", "Percentage"])
        
    #     st.divider()
    #     st.write('Decompositon of Total PPA Energy Source')
    #     st.bar_chart(perc_asset_df, x="Energy Source", y="Percentage")
    
    ## Not using this plot as the idea is to not include bat in PPA energy sourcse consideration
    # def _helper_ppa_component_with_pen_bat_df(self):
    #     st.divider()
    #     st.write('PPA Energy Source Component')
    #     demands = np.sum(self.agg_demand) * self.perc
        
    #     asset_names = self.gen_names
    #     asset_prods = self.input_proddatas
    #     asset_betas = self.beta

    #     mkt_purchase = self.v_buy
        
    #     bat_u_d = self.ex_bat.get_u_d()
        
    #     res_dict = {}
    #     for i in range(len(asset_names)):
    #         res_dict[asset_names[i]] = np.round(np.sum(asset_prods[i] * asset_betas[i]) / demands, 3)
        
    #     res_dict['Market Purchase'] = np.round(np.sum(mkt_purchase) / demands, 3)
    #     res_dict['Battery Discharging'] = np.round(np.sum(bat_u_d) / demands, 3)
        
    #     st.dataframe(pd.DataFrame.from_dict(res_dict, orient="index", columns=["weight"]))

########### Unused Function 3 ##################
    # def _helper_mkt_rev_plot(self):
    #     T = self.T
    #     market_revenue_df = pd.DataFrame(
    #         { 
    #             "Timestep": self.X,
    #             "Market revenue ($)": self.mkt_rev[:T]
    #         }
    #     )
        
    #     st.divider()    
    #     st.write('Revenue from selling excess production to the market (no selling if negative price)')
    #     st.line_chart(market_revenue_df,
    #         x="Timestep", x_label = 'Time',
    #         y=["Market revenue ($)"],)