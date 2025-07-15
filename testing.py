
import streamlit as st
import pandas as pd
import numpy as np

# Input:
# Optimization output:
    # 1. gen: each 3 cols [prod, usage, beta]
    # 2. then: [agg_hourly_gen, mkt_purchase, total_prov]
    # 3. cus: 1 col for each [demand]
    # 4. last col: total demand
    
# bat_opt_output: [change, loe]

# ppa_price / ppa_price_and_gen_capacity

# How did I deal with battery/no battery case?

class Tester():

    def __init__(self, case, matching_level, no_gen, no_cus, lcoe,
                 opt_res_df, ppa_price_txt, 
                 new_asset_lcoe = -1, new_asset_pap_price = -1,
                 spot_price = np.array([0]*8760), pen_factor = 0, pen_cap = 0,
                 bat_change = np.array([0]*8760), bat_params = [0]*6):
        
        self.case = case
        self.matching_lvl = matching_level
        self.no_gen = no_gen
        self.no_cus = no_cus
        self.lcoe = lcoe
        
        self.new_asset_lcoe = new_asset_lcoe
        self.new_asset_pap_price = new_asset_pap_price
        
        self.opt_res_df = opt_res_df
        self.ppa_price_txt = ppa_price_txt
        
        self.spot_price = spot_price
        
        self.pen_factor = pen_factor
        self.pen_cap = pen_cap
        
        self.bat_change = bat_change
        self.bat_params = bat_params
        
        
    def test_total_re_gen(self):
        # case = self.case
        no_gen = self.no_gen
        opt_res_df = self.opt_res_df
        bat_change = self.bat_change
        [lcos, bat_capacity, bat_rte, bat_charging_rate, bat_max_cycle, bat_n] = self.bat_params

        total_gen = opt_res_df['Agg Hourly Renewable Provision (MWh)'].round(2)

        temp_index = np.arange(1, no_gen*3, 3)
        temp_gen_df = opt_res_df.iloc[:, temp_index].copy()
 
        temp_gen_df['bat'] = bat_change
        temp_gen_df['bat'] = temp_gen_df['bat'].apply(lambda x: abs(x * bat_rte) if x < 0 else -x)
        
        row_sums = temp_gen_df.sum(axis = 1).round(2)

        res = row_sums.eq(total_gen)
        
        if bool(res.all().item()):
            print('[PASS] Total RE Provision Matched with Aggregated Quantity')
        else:
            print('[ERROR] Total RE Provision NOT Matched with Aggregated Quantity')
            print('Number of unmatched cases:', 8760 - row_sums.eq(total_gen).sum().item())
            print('')
            print('Showing First 10 Lines of Mismatched Values')
            cnt = 0
            for i in range(8760):
                if row_sums[i] != total_gen[i]:
                    cnt += 1
                    print('Row', i, '| Row sums', row_sums[i], '|', 'Total gen value', total_gen[i])
                if cnt >10:
                    break
            print('')

        
    def test_matching_level(self):
        case = self.case
        opt_res_df = self.opt_res_df
        matching_level = self.matching_lvl
        bat_change = self.bat_change
        [lcos, bat_capacity, bat_rte, bat_charging_rate, bat_max_cycle, bat_n] = self.bat_params
     
        total_gen = opt_res_df['Agg Hourly Renewable Provision (MWh)'].sum()
        total_gen -= np.sum(np.where(bat_change >= 0, 0, bat_change*(-1)))
        total_bat = np.sum(np.where(bat_change >= 0, 0, bat_change * -bat_rte))
   
        
        total_mkt = opt_res_df['Hourly Purchase from Mkt (MWh)'].sum()
        total_prov = total_gen + total_bat + total_mkt 
    
        
        total_demand = opt_res_df['Hourly Total Consumption (MWh)'].sum()
        
        if np.round(total_prov / total_demand, 2) == matching_level:
            print('[PASS] Total Energy Provision Matched with Demand at Target CFE%')

        else:
            print('[ERROR] Total Energy Provision NOT Matched with Demand at Target CFE%')
            print('Cur level', np.round(total_prov / total_demand, 5))
            return False


    # def test_production_quantity(case, no_gen, no_cus):
    #     pass


    # lcoe be a list of prices, manually input
    def test_optimal_price(self):
        
        case = self.case
        opt_res_df = self.opt_res_df
        opt_price_txt = self.ppa_price_txt
        no_gen = self.no_gen
        lcoe = self.lcoe
        pen_factor = self.pen_factor
        pen_cap = self.pen_cap
        spot_price = self.spot_price
        
        [lcos, bat_capacity, bat_rte, bat_charging_rate, bat_max_cycle, bat_n] = self.bat_params
        
        new_asset_lcoe = self.new_asset_lcoe
        new_asset_pap_price = self.new_asset_pap_price
        
        new_asset_rev = 0
        # Case switch for new asset
        if new_asset_pap_price >= 0:
            total_prod = np.array(opt_res_df['New Generator Hourly Production (MWh)'])
            used_prod = np.array(opt_res_df['New Generator PPA Contribution (MWh)'])
            excess_prod = total_prod - used_prod
            new_asset_rev = new_asset_pap_price * np.sum(excess_prod)
            new_asset_cost = new_asset_lcoe * np.sum(total_prod)
            
            temp_index = np.arange(1, (no_gen-1)*3, 3)
            temp_gen_df = opt_res_df.iloc[:, temp_index]
            

        else:
            temp_index = np.arange(1, no_gen*3, 3)
            temp_gen_df = opt_res_df.iloc[:, temp_index]
            new_asset_cost = 0
            
        gen_sums = temp_gen_df.sum().to_numpy()
        re_cost = np.sum(lcoe * gen_sums)

        bat_cost = bat_capacity * bat_max_cycle * lcos * 365
        
        total_mkt = opt_res_df['Hourly Purchase from Mkt (MWh)'].fillna(0).to_numpy()

        mkt_cost = np.sum(total_mkt * np.clip(pen_factor*spot_price, 0, pen_cap)) 
        total_prov = opt_res_df['Hourly Total Energy Provision (MWh)'].sum()
        
        opt_price = np.round((mkt_cost + re_cost + new_asset_cost + bat_cost - new_asset_rev) / total_prov, 4)
        res_price = opt_price_txt
        

        if opt_price == res_price:
            print('[PASS] Optimal Price Matched')

        else:
            print('[ERROR] Optimal price NOT Matched')
            print('computed:', opt_price, 'In result:', res_price)
            print('re_cost', re_cost+new_asset_cost)
            print('bat_cost', bat_cost)
            print('mkt cost', mkt_cost)
            print('res excess rev', new_asset_rev)
            print('total prov', total_prov)