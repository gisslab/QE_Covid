"""
Created on Thu June 1, 2023
@author: Giselle Labrador Badia (@gisslab)

This module is used to analyze the auction prices data from the OB (Optimal Blue) bid evel data data. 

input: 
     file : csv file with the auction prices data time series. 
     path : '/project/houde/mortgages/QE_Covid/data/data_auction/clean_data'

output:
    graphs : prices with date of auctions in x axis and bids in y axis.
    path : '/project/houde/mortgages/QE_Covid/results/figures


"""



#%%
# * libraries

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# * settings

auction_data_folder = '/project/houde/mortgages/QE_Covid/data/data_auction/clean_data'

auction_save_folder = '/project/houde/mortgages/QE_Covid/results/figures'

auction_filename = 'combined_auctions_jan2018-jul2022_cleaned'

bl_data_folder = '/project/houde/mortgages/QE_Covid/data/data_TBA/bloomberg/'


maturity = 30
""" Maturity of the auctioned loans in years."""

loantype = 1
""" Type of the auctioned loans. 1 = Conforming"""

noterange_list = [(1,7),(2, 2.75), (2.75, 3.5), (3, 3.75), (3.75, 5.25),(4, 4.75), (5, 7)]
""" List of tuples with the min and max note rates to filter the data. Index 0 is all the note rates. """


collapsed_note_rate_list = [2.5, 3, 3.75, 4.25, 4.5,  5,  5.5,  6]

#%%

# * functions

def read_data(file, path):
    """
    This function reads the data from the auction data folder and returns a dataframe.
    """

    filepath = f'{path}/{file}.csv'
    print('Reading data from: ', filepath)
    try: 
        df_auc = pd.read_csv(filepath,
                            sep='|',
                            parse_dates=['CommittedDate', 'BorrowerClosingDate']
                            )

        return df_auc
    except Exception as e:
        print('Error reading the data: ',filepath, ", ", e)
        return None
    


def collapse_note_rates(df, list_bins):
    """
    This function collapses the note rates into bins and returns a dataframe with the collapsed note rates.
    """

    df['NoteRate_bins'] = pd.cut(df['NoteRate'], bins = list_bins)
    
    # collapse the prices by weighted mean 
    df = df.groupby(['CommittedDate', 'NoteRate_bins']).agg({'Price': 'mean'}).reset_index()

    return df

def filter_bins_rates(df, min_rate, max_rate):
    """
    This function filters the dataframe by min note rate and max note rate.
    # ! This function is depracated. Funtionality in auction_prices_analysis.py. 
    """

    df = df[(df['NoteRate'] >= min_rate) & (df['NoteRate'] <= max_rate)]

    df['total_loan_amount_day'] = df.groupby(['CommittedDate'])['LoanAmount_sum'].transform('sum')

    # colllapse by day weighting by LoanAmount_sum
    vars = ['w_winner_bid_mean',
            # 'winner_bid_median', 
            # 'winner_bid_std',
            # 'winner_bid_coeff_var', 'winner_bid_p90_p10' 
            ]
    for var in vars:
        df[var] = df[var] * df['LoanAmount_sum']/ df['total_loan_amount_day']

    # collapse by day
    df = df.groupby(['CommittedDate']).agg({'w_winner_bid_mean': 'mean', 
                                            'winner_bid_median': 'mean',
                                            'winner_bid_std': 'mean',
                                            'winner_bid_coeff_var': 'mean',
                                            'winner_bid_p90_p10': 'mean'
    }).reset_index()

    # column names 
    df.columns = ['CommittedDate', 
                  'w_winner_bid_mean', 
                  'winner_bid_median', 
                  'winner_bid_std', 
                  'winner_bid_coeff_var', 
                  'winner_bid_p90_p10']

    return df


def plot(df, var, maturity, initial_stat = "Mean",
          vertical_lines = ["2020-03-01","2020-04-01", "2020-04-15"],
          fig = None, ax = None, color = 'tab:blue',
          save = True, , empty_label = False,
          legend = False, legendlabel = "_nolegend_",
          title = False):
    """
    This function plots the time series of the variable var.
    """
    additional = '' if empty_label else 'of the bid prices'
    yadd = '' if empty_label else 'Highest bid'
    # plot the time series
    if fig is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(df['Trading_Date'], df[var], color = color, label = legendlabel)
    # x axis labels, only 10 dates
    ax.set_xticks(df['Trading_Date'][::int(len(df)/12)])
    # rotate the x axis labels
    plt.xticks(rotation=45)
    # add title
    if title:
        plt.title(f'{initial_stat} {additional} for loans with maturity {maturity} years')
    # y axis name
    plt.ylabel(f'{yadd} {initial_stat.lower()}')
    plt.xlabel('Date')

    plt.subplots_adjust(top=0.925, 
                    bottom=0.20, 
                    left=0.12, 
                    right=0.96, 
                    hspace=0.01, 
                    wspace=0.01)

    for vl in vertical_lines:
        plt.axvline(x=pd.to_datetime(vl), color='r', linestyle='--', linewidth=1, alpha = 0.7, label = None)

    if legend:
        plt.legend(loc="upper left")

    if save:
        fig.savefig(f'{auction_save_folder}/{var}_mat{maturity}_loan{loantype}_timeseries_nr_{noterate_range[0]}_{noterate_range[1]}.pdf')

    return fig, ax

  
#%%

if __name__ == '__main__':

    
    # %%
    # ************* Data ************* #

    # * auction OB data

    # choosing note rate range
    noterate_range = noterange_list[0]
    # building path 
    filename_timeseries = f'{auction_filename}_mat{maturity}_loan{loantype}_timeseries_nr_{noterate_range[0]}_{noterate_range[1]}'

    print('Note rate range: ', noterate_range)

    df_ts = read_data(file = filename_timeseries, path = auction_data_folder)
    df_ts = df_ts[df_ts['CommittedDate'] >= '2020-01-01']
    # rename to Trading_Date
    df_ts = df_ts.rename(columns = {'CommittedDate': 'Trading_Date'})

        # %%
    df_ts.columns
    # %%
    # print("Unique note rates: ", df_ts['NoteRate'].nunique())
    # df_ts.value_counts('NoteRate')

    # %%
    # df_ts = collapse_note_rates(df_ts, collapsed_note_rate_list)
    # df_ts['NoteRate_bins'].value_counts()

    # %%
    # ts = filter_bins_rates(df_ts, min_rate = 3, max_rate = 3.75)
    ts = df_ts
    # if count < 1 delete, so small number of observations do not create additional noise. 
    ts = ts[ts['winner_bid_count'] >= 5]


    # %%

    # * bloomberg data

    # bloomberg_daily_trading_prices_w_forwards bloomberg_daily_trading_prices

    df_bl = pd.read_csv(f'{bl_data_folder}/clean_data/bloomberg_daily_trading_prices.csv',
                    sep='|'
                    )
    df_bl.columns

    # %%
    # convert to date time
    df_bl['Trading_Date'] = pd.to_datetime(df_bl['Trading_Date'])
    df_bl.Trading_Date.describe()
    # %%
    # trading date 2020
    df_bl_2020 = df_bl[df_bl['Trading_Date'].dt.year == 2020] # same dates that other dataset
    df_bl_2020 = df_bl_2020[df_bl_2020['Trading_Date'] <= '2020-04-30']

    # coopon rate that is between noterate_range
    df_bl_2020 = df_bl_2020[(df_bl_2020['Coupon'] >= noterate_range[0]) & (df_bl_2020['Coupon'] < noterate_range[1])]

    # ? ticker FNCL and maybe FGLMC
    df_bl_2020 = df_bl_2020[df_bl_2020['Ticker'].isin(['FNCL', 'FGLMC'])]

    # %%
    df_bl_2020.describe()
    # %%
    # group by to get one price PX_Last per day (avg across months forwards and ticker )
    df_bl_2020 = df_bl_2020.groupby(['Trading_Date']).agg({'PX_Last': 'mean'}).reset_index()



    # %%
    # ******** Plots ******** #
    # %%
    #* count

    var = 'winner_bid_count'
    plot(ts, var, maturity,  initial_stat = "Count", legendlabel="OB")

    # %%
    # * loan amount sum
    var = 'LoanAmount_sum'
    plot(ts, var, maturity, initial_stat = "Loan amount sum", empty_label = True)
    # # %%
    # # * min 

    # var = 'winner_bid_min'
    # plot(ts, var, maturity, initial_stat = "Min")

    # # %%
    # # * max 

    # var = 'winner_bid_max'
    # plot(ts, var, maturity, initial_stat = "Max")

    # %%

    # * min max

    f, a = plot(ts, 'winner_bid_min', maturity, initial_stat = "Min/Max", save=False, legend=True, legendlabel = 'Min')
    f, a = plot(ts, 'winner_bid_max', maturity, initial_stat = "Min/Max", fig = f, ax = a, color = 'tab:orange', legend=True, legendlabel = 'Max')
    
    # %%
    # * mean (w if weighted)
    var = 'w_winner_bid_mean'
    # add bloomberg
    f, a = plot(df_bl_2020, var = 'PX_Last', maturity = maturity, initial_stat = "",  color = 'tab:orange', legend=True, legendlabel = 'Bloomberg', save=False)
    f, a = plot(ts, var, maturity, initial_stat = "Mean", fig = f, ax = a, color = 'tab:blue', legend=True, legendlabel = 'OB', save=True)

    # %%
    # * median 
    var = 'winner_bid_median'
    f, a = plot(df_bl_2020, var = 'PX_Last', maturity = maturity, initial_stat = "",  color = 'tab:orange', legend=True, legendlabel = 'Bloomberg', save=False)
    plot(ts, var, maturity, initial_stat = "Median", fig = f, ax = a, color = 'tab:blue', legend=True, legendlabel = 'OB', save=True)
    
    # %%
    # * days to auction
    var = 'DaysToAuction_mean'
    plot(ts, var, maturity, initial_stat = "Days to auction", empty_label = True)

    # %%
    # * dummy sell any
    var = 'dummy_sell_any_mean'
    plot(ts, var, maturity, initial_stat = "Rate sell", empty_label = True)

    # %%
    # * dummy sell winner
    var = 'dummy_sell_winner_mean'
    plot(ts, var, maturity, initial_stat = "Rate sell to winner", empty_label = True)

    # %%
    
    # * number of enterprise bidders
    var = 'Number of Enterprise Bidders_mean'
    plot(ts, var, maturity, initial_stat = "Number of enterprise bidders", empty_label = True)

    # %%
    # * number of bulk bidders
    var = 'Number of Bulk Bidders_mean'
    plot(ts, var, maturity, initial_stat = "Number of bulk bidders", empty_label = True)

    # %%
    # * number of participants
    var = 'Number of Participants_mean'
    plot(ts, var, maturity, initial_stat = "Number of participants", empty_label = True)

    # # %% 
    # # * std

    # var = 'winner_bid_std'
    # plot(ts, var, maturity, initial_stat = "Std")

    # # %%
    # # * coeff var
    # # 
    # var = 'winner_bid_coeff_var' 
    # plot(ts, var, maturity, initial_stat = "CV")

    # # %%
    # # * p90-p10

    # var = 'winner_bid_p90_p10'
    # plot(ts, var, maturity, initial_stat = "P90-P10")








    





# %%
