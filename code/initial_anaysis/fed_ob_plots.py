"""
Created on July 25, 2023
@author: Giselle Labrador Badia (@gisslab)

This script plots the evolution of the Federal Reserve's QE purchases prices and the Ob auction prices.
"""
# %% Import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
# import other modules
import fed_mbs as fmbs
import auction_prices_timeseries_plots as aptp

# %%



if __name__ == '__main__':

    # %%
    fed = pd.read_csv(fmbs.fed_data + '/clean_data/fed_bl_forward_coupon_monthly.csv',
                      sep='|')
    fed.head()
    # %%
    ob = pd.read_csv(f'{aptp.auction_data_folder}/ob_bl_merge_monthly_coupon.csv',
                        sep=',')
    ob.head()

    # %%

    # now we want only 1 forward month in fed "forwardmonths" column
    fed = fed[fed['forwardmonths'] == 1]

    # eliminate PX_Last column from Ob since it is 2 months forward
    ob = ob.drop(columns=['PX_Last'])

    # %%

    # merge both dataframes on FirstMonthYear
    group = ['FirstMonthYear', 'Coupon']

    df = pd.merge(fed, ob, on= group, how='outer')

    # convert date times
    df['FirstMonthYear'] = pd.to_datetime(df['FirstMonthYear'])

    # only until april 2021
    df = df[df['FirstMonthYear'] < '2021-05-01']

    # %%

    # Plots using the function from auction_prices_timeseries_plots.py
    df_coupon2_5 = df[df['Coupon'] == 2.5].copy()
        #! why FED begins in march? 
    # Its because february is missing 
    # make february be mean between january and march in 2020 using FirstMonthYear column
    sum_fed_price = df_coupon2_5.loc[df.FirstMonthYear == '2020-01-01' ,'fed_price_mean'].values[0] + df_coupon2_5.loc[df.FirstMonthYear == '2020-03-01' ,'fed_price_mean'].values[0]
    df_coupon2_5.loc[df.FirstMonthYear == '2020-02-01' ,'fed_price_mean'] = sum_fed_price/2
    # same with PxLast
    sum_px_last = df_coupon2_5.loc[df.FirstMonthYear == '2020-01-01' ,'PX_Last'].values[0] + df_coupon2_5.loc[df.FirstMonthYear == '2020-03-01' ,'PX_Last'].values[0]
    df_coupon2_5.loc[df.FirstMonthYear == '2020-02-01' ,'PX_Last'] = sum_px_last/2

    
    # %%
    # plot 1: Fed vs OB prices

    f,a = aptp.plot(df_coupon2_5, var = 'winner_bid_mean',
              title = 'Fed vs OB prices for 2.5 coupon',
              horizontal_lines=[],
              legend = True,
              legendlabel='Ob',
              varrate= '',
              initial_stat="Price",
              save = False,
              )

    # add Bloomberg too
    f,a = aptp.plot(df_coupon2_5, var = 'PX_Last',
            fig = f, ax = a,
            title = 'Fed vs OB prices for 2.5 coupon',
            horizontal_lines=[],
            legend = True,
            legendlabel='Bloomberg',
            varrate= '',
            initial_stat="Price",
            empty_label = False,
            color = 'green',
            save = False
    )   
    # same with fed_price
    aptp.plot(df_coupon2_5, var = 'fed_price_mean',
            fig = f, ax = a,
            title = 'Fed vs OB prices for 2.5 coupon',
            horizontal_lines=[],
            legend = True,
            legendlabel='Fed',
            varrate= '',
            initial_stat="Price",
            empty_label = False,
            color = 'orange',
            save = False
    )


    # %%

    # plot 2: 
    # same but normalized by Bloomberg
    f,a = aptp.plot(df_coupon2_5, var = 'winner_bid_mean',
                title = 'Fed vs OB prices for 2.5 coupon',
                horizontal_lines=[0],
                legend = True,
                legendlabel='Ob',
                varrate= '',
                initial_stat="Net price",
                save = False,
                normalization_var='PX_Last',
                empty_label=True
    )
    # same with fed
    aptp.plot(df_coupon2_5, var = 'fed_price_mean',
            fig = f, ax = a,
            title = 'Fed vs OB prices for 2.5 coupon',
            horizontal_lines=[0],
            legend = True,
            legendlabel='Fed',
            varrate= '',
            initial_stat="Net price",
            empty_label = True,
            color = 'orange',
            normalization_var='PX_Last'
    )


# %%
# df_coupon2_5 for january 2020 abd february 2020

df_coupon2_5_janfeb = df_coupon2_5[(df_coupon2_5['FirstMonthYear'] == '2020-01-01') | (df_coupon2_5['FirstMonthYear'] == '2020-02-01')]
df_coupon2_5_janfeb
# %%
