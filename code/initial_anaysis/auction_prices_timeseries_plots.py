"""
Created on Thu June 1, 2023
@author: Giselle Labrador Badia (@gisslab)

This module is used to analyze the auction prices data from the OB (Optimal Blue) bid evel data data.

Includes functions to read and clean Bloomberg TBA price data.

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
import auction_prices_analysis as ap


# * settings

#TODO: Nicer modulee to manage paths

auction_data_folder = '/project/houde/mortgages/QE_Covid/data/data_auction/clean_data'

auction_save_folder = '/project/houde/mortgages/QE_Covid/results/figures'

table_folder = '/project/houde/mortgages/QE_Covid/results/tables'

auction_filename = 'combined_auctions_jan2018-jul2022_cleaned'

bl_data_folder = '/project/houde/mortgages/QE_Covid/data/data_TBA/bloomberg/clean_data/'

bloomberg_filename = 'bloomberg_daily_trading_prices' # bloomberg_daily_trading_prices_w_forwards


maturity = 30
""" Maturity of the auctioned loans in years."""

loantype = 1
""" Type of the auctioned loans. 1 = Conforming"""

collapsed_note_rate_list = [2.5, 3, 3.75, 4.25, 4.5,  5,  5.5,  6]

interval = (2.5,4) # Coupons ot note rate

#%%

# * functions

def read_data(file, path, datetime_vars = ['CommittedDate', 'BorrowerClosingDate', 'FirstMonthYear']):
    """
    This function reads the data from the path data folder and returns a dataframe.
    """

    filepath = f'{path}/{file}.csv'
    print('Reading data from: ', filepath)
    try: 
        df_auc = pd.read_csv(filepath,
                            sep='|',
                            parse_dates= datetime_vars
                            )

        return df_auc
    except Exception as e:
        print('Error reading the data: ',filepath, ", ", e)
        return None
    
def tide_auction_data(df, 
                      min_date = '2019-07-01', max_date = '2021-12-31', 
                      min_daily_count = 10,
                      vartime = 'FirstMonthYear', #'CommittedDate', # ,
                      interval = (1,7), 
                      varrate = 'Coupon',
                      setrates = [] ):
    """
    Receives auction data daily time series cleaned by auction_prices_analysis module and returns cleaner data ready to be used in plot function.

    df : dataframe with auction data.
    min_date : minimum date to filter the data.
    max_date : maximum date to filter the data.
    min_daily_count : minimum number of observations per day to keep the day.
    vartime : time unit, e.g. FirstMonthYear, Trading_Date.
    interval : Filter by <varrate> range in interval (inclusive), e.g. [2,4] for coupon range.
    setrates : Keep only the <varrate> in the list, e.g. [2.5, 3.0, 3.5, 4.0] for coupon list.
    """
    print("Type of vartime var in pd", type(df[vartime][0]))
    # Convert to datetime
    df[vartime] = pd.to_datetime(df[vartime])

    df = df[ (df[vartime] >= pd.to_datetime(min_date)) & (df[vartime] < pd.to_datetime(max_date)) ]

    #Convert Loan Amount to millions, it is in thousands
    df['LoanAmount_sum'] = df['LoanAmount_sum']/(10**3)
    # rename to Trading_Date to plot with the TBA data
    df = df.rename(columns = { 
                                # vartime: 'Trading_Date'
                                's_coupon': 'Coupon' 
                                })

    # if count < 1 delete, so small number of observations do not create additional noise.
    if varrate in df.columns: 
        df = df[df['winner_bid_count'] >= 1]
    else : df = df[df['winner_bid_count'] >= min_daily_count]

    # NoteRate range if NoteRate is a column 
    if varrate in df.columns:
        print("NoteRate range: ", interval)
        df = df[(df[varrate] >= interval[0]) & (df[varrate] <= interval[1])] 

    if setrates != []:
        df = df[df[varrate].isin(setrates)]

    return df


def tide_collapse_bloomberg_data(df_, 
                                interval = (1,7), # coupon range
                                min_date = '2019-07-01', max_date = '2021-12-31', #'2020-05-01'
                                tickers = ['FNCL', 'FGLMC'],
                                service_fee = 0.75,
                                group_by = ['FirstMonthYear', 'Coupon'], #Trading_Date
                                bycoupon = True,
                                forward_months = [2],
                                ):
    """
    Receives bloomberg data daily cleaned and returns cleaner time series at the monthly/daily level (specified in group_by).

    df_ : dataframe with bloomberg data.
    interval : tuple with coupon.
    min_date : minimum date to filter the data.
    max_date : maximum date to filter the data.
    tickers : list of tickers to filter the data (FNC, FGLMC, GNMA).
    service_fee : service fee to be substracted from the note rate, only when bycoupon = False.
    group_by : list of variables to group by, usually one time and one coupon, e.g., ['FirstMonthYear', 'Coupon'].
    bycoupon : boolean, if True filter by coupon, if False filter assuming interval is note rate (note rate - service_fee).
    forward_months : number of months forward to filter the data.
    """
    df = df_.copy()

    # coupon rate that is between interval
    # * rule for note rates
    if bycoupon:
        df = df[(df['Coupon'] >= interval[0]) 
                & (df['Coupon'] <= interval[1])]
    else: 
        df = df[(df['Coupon'] >= interval[0]- service_fee - 0.25) 
                & (df['Coupon'] < interval[1] - service_fee + 0.25)]

    print("Coupons in range: ", df['Coupon'].unique())

    # filter by forward months
    # df = df.loc[df['Forward_Trading_Months'].isin(forward_months), :]

    # # ? ticker FNCL and maybe FGLMC
    df = df[df['Ticker'].isin(tickers)]

    # create Month Year variable
    # df['MonthYear'] = (
    #     df["Trading_Date"].dt.month_name() + "-" + df["Trading_Date"].dt.year.astype(str)
    # )
    # create date time column withfirst day of Committed date: e.g. 2020-01-22 -> 2020-01-01
    df['FirstMonthYear'] = df['Trading_Date'].dt.to_period('M').dt.to_timestamp()

    df = df[(df['Trading_Date'] < pd.to_datetime(max_date)) & (df['Trading_Date'] >= pd.to_datetime(min_date))]

    # group by to get one price PX_Last, per day, coupon (avg across months forwards and ticker )
    df = df.groupby(group_by).agg({'PX_Last': 'mean'}
                                    ).reset_index()

    print("Number of observations: ", df.shape[0])

    return df

def merge_bl_ob(df_bl, df_ob, on = ['FirstMonthYear', 'Coupon']):
    """
    Merge bloomberg and OB data on arguments passed on.
    """

    df = pd.merge(df_bl, df_ob, on = on, how = 'right')

    return df



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
    """

    df = df[(df['NoteRate'] >= min_rate) & (df['NoteRate'] <= max_rate)]

    # df['total_loan_amount_day'] = df.groupby(['CommittedDate'])['LoanAmount_sum'].transform('sum')

    # colllapse by day weighting by LoanAmount_sum
    # vars = ['w_winner_bid_mean',
    #         # 'winner_bid_median', 
    #         # 'winner_bid_std',
    #         # 'winner_bid_coeff_var', 'winner_bid_p90_p10' 
    #         ]
    # for var in vars:
    #     df[var] = df[var] * df['LoanAmount_sum']/ df['total_loan_amount_day']

    # collapse by day
    df = df.groupby(['Trading_Date']).agg({'w_winner_bid_mean': 'mean', 
                                            'winner_bid_median': 'mean',
                                            'winner_bid_std': 'mean',
                                            'winner_bid_coeff_var': 'mean',
                                            'winner_bid_p90_p10': 'mean'
    }).reset_index()

    # column names 
    df.columns = ['Trading_Date', 
                  'w_winner_bid_mean', 
                  'winner_bid_median', 
                  'winner_bid_std', 
                  'winner_bid_coeff_var', 
                  'winner_bid_p90_p10']

    return df


def plot(df, var, 
          maturity = 30, 
          initial_stat = "Mean",
          vertical_lines = ["2020-03-01"], #["2020-03-01","2020-04-01", "2020-04-15"],
          horizontal_lines = [0],
          fig = None, ax = None, color = 'tab:blue',
          save = True, 
          empty_label = False,
          legend = False, legendlabel = "_nolegend_",
          title = False,
          vartime = 'FirstMonthYear', #'CommittedDate', # ,
          varrate = 'Coupon',
          filenameend = '',
          normalization_var = '', #'PX_Last',
          interval = interval,
          linestyle='-'
          ):
    """
    Plots the time series of the variable <var> (y axis) by <varrate> where <vartime> is the x axis . <varrate> variable defines the color.

    df : dataframe with the data to plot.
    var : variable to plot.
    initial_stat : Label that goes in title and ylabel.
    vertical_lines : list of dates to plot vertical lines.
    horizontal_lines : list of values to plot horizontal lines.
    fig : figure to plot on.
    ax : axis to plot on.
    color : color of the line.
    save : boolean, if True saves the figure.
    empty_label : boolean, if True does not add additional text to the y axis label.
    legend : boolean, if True adds legend.
    legendlabel : label of the legend variable.
    title : boolean, if True adds title.
    vartime : variable to plot in the x axis.
    varrate : variable to plot in the color.
    filenameend : string to add to the filename when saving.
    normalization_var : variable to normalize the variable to plot.
    interval : tuple with the interval of the variable to plot.
    linestyle : linestyle of the plot.

    """
    agglevel = 'daily' if vartime == 'Trading_Date' else 'monthly'
    agglevel = 'cp' + agglevel if varrate == 'Coupon' else 'nr' + agglevel

    additional = '' if empty_label else 'of the bid prices'
    yadd = '' if empty_label else 'Highest bid'
    # plot the time series
    if fig is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # each coupon different colors
    if varrate != '':
        for coupon in df[varrate].unique():
            df_ = df[df[varrate] == coupon].copy()
            df_.sort_values(vartime, inplace=True)
            if normalization_var != '':
                df_[var] = df_[var] -  df_[normalization_var]
            ax.plot(df_[vartime], df_[var], label = f'{coupon}', alpha = 0.8, linewidth=2.0, linestyle = linestyle) # color = color, 


    else: 
        df.sort_values(vartime, inplace=True)
        df_ = df.copy()

        if normalization_var != '':
            df_[var] = df_[var] -  df_[normalization_var]
        ax.plot(df_[vartime], df_[var], color = color, label = legendlabel, alpha = 0.8, linewidth=2.0, linestyle = linestyle) # color = color,

    # add horizontal lines
    for hl in horizontal_lines:
        ax.axhline(y=hl, color='black', linestyle='--', linewidth=1, alpha = 0.5, label = None)

    # x axis labels, only 10 dates
    # ax.set_xticks(df[vartime][::int(len(df)/12)])
    # rotate the x axis labels
    plt.xticks(rotation=45)
    # add title
    if title:
        plt.title(f'{initial_stat} {additional} for loans with maturity {maturity} years')
    # y axis name
    plt.ylabel(f'{yadd} {initial_stat}') # {initial_stat.lower()}
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
        plt.legend(loc="upper right", title = varrate)

    if save:
        # if var interval variable exists
        if interval != None: 
            intervals = f'{interval[0]}_{interval[1]}_'
        else: intervals = ''

        filepath = f'{auction_save_folder}/{var}_mat{maturity}_loan{loantype}_timeseries_{agglevel}_{intervals}{filenameend}.pdf'
        print('Saving plot to: ', filepath)
        fig.savefig(filepath)

    return fig, ax

def create_table_stats(df,
                       cols = ['LoanAmount', 'NoteRate', 'Price', 
                                'DaysToAuction', 'Number of Participants', 'Number of Bulk Bidders',
                                'dummy_sell_any', 'dummy_sell_winner'],
                        mat = 30,
                        loantype = 1,
                        additional_name = '',
                        stats = ['count','mean', 'std', 'min',  '25%', 'median', '75%', 'max']):
    """
    Creates a table with summary statistics for the variables in cols, returns a dataframe and save the latex. 
    """
    
    # create table df with summary statistics from describe mean median min max 25 % 75 % std

    df_table = df[cols].describe().T[['count','mean', 'std', '50%', 'min',  '25%', '75%', 'max']]
    df_table = df_table.rename(columns={'50%': 'median'})
    # rename variable names 
    df_table = df_table.rename(index={'LoanAmount': 'Loan amount', 'NoteRate': 'Note rate', 'Price': 'Price',
                                        'DaysToAuction': 'Days to auction', 'Number of Participants': 'Number of participants', 'Number of Bulk Bidders': 'Number of bulk bidders',
                                        'dummy_sell_any': 'Sell rate', 'dummy_sell_winner': 'Rate sell to winner'})
    # round all 
    df_table = df_table.round(2)
    # # add count, ommit nan
    # df_table['count'] = df[cols].count()
    # reorder columns
    df_table = df_table[stats]

    # to tex
    filepath = f'{table_folder}/auctions_level_mat{mat}_loan{loantype}_{additional_name}.tex'
    print('Saving table to: ', filepath)
    df_table.to_latex(filepath)

    return df_table

def build_values_note_rate_from_TBA(df_tba,
                                    df_noterate_timeseries,
                                    maturity = 30,
                                    loantype = 1,
                                    interval = [2.5, 3.5]):
    """
    This function builds the values of the note rate from the TBA data.
    """
    # filter df_nr by noterate range, NoteRate variable
    df_noterate_timeseries.loc[ (df_noterate_timeseries['NoteRate'] >= interval[0]) &
                                                     (df_noterate_timeseries['NoteRate'] <= interval[1]), :]
    

    # for each note rate calculate the value of the note rate from the TBA data 

    df_noterate_timeseries.loc[:, 'value_note_rate'] = df_noterate_timeseries.apply(
                                            lambda x: compute_note_rate_value(x, df_tba),
                                            axis = 1)


    #! for now eliminate the nan values that are days in which there are no reported TBA prices
    df_noterate_timeseries = df_noterate_timeseries.dropna(subset = ['value_note_rate'])

    # collapse all values by mean weighting by LoanAmount_sum by Trading_Date
    df_noterate_timeseries.loc[:,'LoanAmount_sum_tot'] = df_noterate_timeseries.groupby('Trading_Date')['LoanAmount_sum'].transform('sum')
    df_noterate_timeseries['weight'] = df_noterate_timeseries['LoanAmount_sum'] / df_noterate_timeseries['LoanAmount_sum_tot']
    df_noterate_timeseries['w_value_note_rate'] = df_noterate_timeseries['value_note_rate'] * df_noterate_timeseries['weight']
 

    # create weighted measure of value_note_rate by date
    df_noterate_timeseries['value_day'] = df_noterate_timeseries.groupby('Trading_Date')['w_value_note_rate'].transform('sum')


    #! Depending on how you weight by date you get different results
    #! e.g. by the amounts of the loans for each note rate
    # or all the same

    # rename  w_value_note_rate to PX_Last
    df_noterate_timeseries = df_noterate_timeseries.rename(columns={'w_value_note_rate': 'PX_Last'})

    print(df_noterate_timeseries[["weight", "PX_Last", "value_note_rate"]].describe())

    # collapse by date
    # df_noterate_timeseries = df_noterate_timeseries.groupby('Trading_Date').sum( numeric_only = True).reset_index()
    # group by Note rate mean, PX_Last sum , by Trading_Date, weight sum, value_note_rate mean 

    df_noterate_timeseries = df_noterate_timeseries.groupby('Trading_Date').agg({'NoteRate': 'mean', 
                                                                                'PX_Last': 'sum',
                                                                                'weight': 'sum', 
                                                                                'value_note_rate': 'mean',
                                                                                'value_day': 'mean'}
                                                                                ).reset_index() 


    return df_noterate_timeseries



def compute_note_rate_value(row, df_tba, service_fee = 0.75):
    """
    This function computes the value of the note rate from the TBA data. Only when Ob is aggregated by note rate.
    """
    # filter by day Trading_Date
    # .loc[row_indexer,col_indexer] = value instead
    # df_tba_ = df_tba[df_tba['Trading_Date'] == row['Trading_Date']]
    df_tba_ = df_tba.loc[df_tba['Trading_Date'] == row['Trading_Date'], :]

    # filter df_nr by noterate range, NoteRate variable
    df_tba_ = df_tba_[(df_tba_['Coupon'] >= row['NoteRate'] - 0.25 - service_fee) &
                    (df_tba_['Coupon'] <= row['NoteRate'] + 0.25 - service_fee)] 
    # df_tba_ = df_tba_.loc[(df_tba_['Coupon'] >= row['NoteRate'] - 0.25 - service_fee) &
    #                 (df_tba_['Coupon'] <= row['NoteRate'] + 0.25 - service_fee), :]

    if df_tba_.empty:
        # if empty return nan
        # print('empty', row['Trading_Date'], row['NoteRate'])
        value_note_rate = np.nan
    # just average the coupons to calculate value 
    value_note_rate = df_tba_['PX_Last'].mean()
    return value_note_rate




#%%
# * main function



def main():
    
    # %%
    # *********************************** Data ********************************* #

    """
    Number of observations:  338
    Saving data to:  /project/houde/mortgages/QE_Covid/data/data_auction/clean_data/timeseries/combined_auctions_jan2018-jul2022_cleaned_mat30_loan1_timeseries_s_coupon_MonthYear_auctype.csv

    Number of observations:  174
    Saving data to:  /project/houde/mortgages/QE_Covid/data/data_auction/clean_data/timeseries/combined_auctions_jan2018-jul2022_cleaned_mat30_loan1_timeseries_s_coupon_MonthYear.csv

    Number of observations:  24
    Saving data to:  /project/houde/mortgages/QE_Covid/data/data_auction/clean_data/timeseries/combined_auctions_jan2018-jul2022_cleaned_mat30_loan1_timeseries_MonthYear.csv
    """

    # ******************************* auction OB data ********************************** #

    var_time = 'MonthYear'
    var_rate = 's_coupon'
    # aucttype = 'cash_window' # cash_window

    # choosing note rate range
    interval = (2.5,4) #ap.couponrange_list[2]
    set_coupons = [ 2.5, 3.0, 3.5, 4.0] #1.5, 2.0,
    set_coupons_extended = [1.5, 2.0,  2.5, 3.0, 3.5, 4.0, 4.5] #1.5, 2.0,
    interval_extended = (min(set_coupons_extended), max(set_coupons_extended))

    # * dataframe 1: by auction type, coupon
    # building path 
    filename_timeseries_all = f'{auction_filename}_mat{maturity}_loan{loantype}_timeseries_{var_rate}_{var_time}_auctype'
    # filename_timeseries = f'{auction_filename}_mat{maturity}_loan{loantype}_timeseries_{var_rate}_{interval[0]}_{interval[1]}_{var_time}_auctype'


    print('Note rate range: ', interval)

    df_ts_all = read_data(file = f'timeseries/{filename_timeseries_all}',
                        path = auction_data_folder,
                        datetime_vars=['FirstMonthYear'])
    
    # * dataframe 2: by coupon 
    
    filename_timeseries_coup = f'{auction_filename}_mat{maturity}_loan{loantype}_timeseries_{var_rate}_{var_time}'


    df_ts = read_data(file = f'timeseries/{filename_timeseries_coup}',
                        path = auction_data_folder,
                        datetime_vars=['FirstMonthYear'])

    
    # * dataframe 3: by month year

    filename_timeseries_month = f'{auction_filename}_mat{maturity}_loan{loantype}_timeseries_{var_rate}_{interval[0]}_{interval[1]}_{var_time}_agg'

    df_ts_month = read_data(file = f'timeseries/{filename_timeseries_month}',
                                path = auction_data_folder,
                                datetime_vars=['FirstMonthYear'])
    
    # %%
    df_ts.columns

    # %%
    df_ts_all.columns
    # %%
    # * cleaning time series data
    # ts = filter_bins_rates(df_ts, min_rate = 3, max_rate = 3.75)
    # ts = tide_auction_data(df_ts)
    ts_all = tide_auction_data(df_ts_all, 
                               interval= interval,
                               setrates= set_coupons)
    
    ts = tide_auction_data(df_ts,
                            interval= interval,
                            setrates= set_coupons)

    ts_extended = tide_auction_data(df_ts,
                        interval= interval_extended,
                        setrates= set_coupons_extended)

    ts_all.head()
    # %%
    # ts_all_agg = filter_bins_rates(ts_all, min_rate = interval[0], max_rate = interval[1])
    # %%

    # *************************** bloomberg data ********************************** #

    # bloomberg_daily_trading_prices_w_forwards bloomberg_daily_trading_prices

    df_bl = read_data(file = bloomberg_filename, 
                      path = bl_data_folder, 
                      datetime_vars=['Trading_Date', 'Settlement_Date'])
    df_bl.head()

    # %%
    df_bl_2020 = tide_collapse_bloomberg_data(df_bl, 
                                              interval= interval,
                                              forward_months = [1, 2])
    df_bl_2020.head()
    # 0 Coupons in range:  [2.5 3.  3.5 4.  4.5 5.  5.5 6.  6.5]
    # 3 [2.5 3.  3.5]
    # 5 Coupons in range:  [3.5 4.  4.5]
    # %%
    df_bl_2020_ts = tide_collapse_bloomberg_data(df_bl, interval= interval, group_by=['FirstMonthYear'])

    # # %%
    # # only when is aggregated by note rate
    # df_tba = build_values_note_rate_from_TBA(df_tba = df_bl_2020,
    #                                 df_noterate_timeseries = ts_all,
    #                                 interval = interval)
    # df_tba.head()
    # # %%
    # df_tba.value_note_rate.describe()
    # # %%
    # # counut nan
    # df_tba.value_note_rate.isna().sum()

    # %%
    # *********************** merge OB and BL data ********************************** #

    # * auction type, year, month coupon level 
    ts_ob_bl = merge_bl_ob(df_bl_2020, ts_all)

    # * by coupon, year, month
    ts_ob_bl_collapsed = merge_bl_ob(df_bl_2020, ts) # not by auction type

    # save 
    ts_ob_bl.to_csv(f'{auction_data_folder}/ob_bl_merge_monthly_coupon_auctiontype.csv', index = False)
    ts_ob_bl_collapsed.to_csv(f'{auction_data_folder}/ob_bl_merge_monthly_coupon.csv', index = False)

    # %%
    # only Coupon 3.0 
    # ts_ob_bl_ = ts_ob_bl[ts_ob_bl['Coupon'] == 3.0]
    # ts_ob_bl_.head(40)
    # %%

    # * filter by auction type
    aucttype = 'auction' #'cash_window'  'auction'
    ts_ob_bl_1 = ts_ob_bl[ts_ob_bl['auction_type'] == aucttype].copy()
    

    # %% 
    # collapse coupons by averaging the prices by day, auction type
    # create normalize price winner_bid_mean - PX_Last
    ts_ob_bl_1['winner_bid_mean_n'] = ts_ob_bl_1['winner_bid_mean'] - ts_ob_bl_1['PX_Last']
    # now collapse by day weighting by LoanAmount_sum
    ts_ob_bl_1['total_loan_amount_day'] = ts_ob_bl_1.groupby(['FirstMonthYear'])['LoanAmount_sum'].transform('sum')
    ts_ob_bl_1['w_winner_bid_mean_n'] = ts_ob_bl_1['winner_bid_mean_n'] * ts_ob_bl_1['LoanAmount_sum']/ ts_ob_bl_1['total_loan_amount_day']
    ts_ob_bl_1_collapse = ts_ob_bl_1.groupby(['FirstMonthYear']).agg({'w_winner_bid_mean_n': 'sum'}).reset_index()

    # ************************************************************************************************
    # ****************************** Plot ************************************************************
    # ************************************************************************************************


    # %%
    #* count

    var = 'winner_bid_count'
    plot(ts_ob_bl_1, var, maturity,  initial_stat = "Count", legendlabel="OB", 
         empty_label = True, legend = True, filenameend=aucttype)
    
    plot(ts, var, maturity,  initial_stat = "Count", legendlabel="OB", 
        empty_label = True, legend = True)

    # %%
    # * loan amount sum
    var = 'LoanAmount_sum'
    plot(ts_ob_bl_1, var, maturity, initial_stat = "Loan amount total (millions $)", empty_label = True,
        legend = True, filenameend=aucttype, legendlabel = 'OB')
    
    plot(ts, var, maturity, initial_stat = "loan amount total (millions $)", empty_label = True,
        legend = True, legendlabel = 'OB')
    

    # %%
    # ? Note : Use  ts, and df_tba when larger aggregation is needed
    # * mean (w if weighted)
    var = 'winner_bid_mean'    # df_tba df_bl_2020_ts ts_all_agg ts 
    # add bloomberg
    f, a = plot(ts_ob_bl_1, var, normalization_var= 'PX_Last', maturity = maturity,
                initial_stat = "(mean) $ ", legend=True, filenameend=aucttype + '_netbid')

    # %%
    # ts_ob_bl_collapsed
    f, a = plot(ts_ob_bl_collapsed, var, normalization_var= 'PX_Last', maturity = maturity,
                initial_stat = "(mean) $ ", legend=True, filenameend= '_netbid')

    # %%
    f, a = plot(ts_ob_bl_1, var, maturity = maturity,
            initial_stat = "(mean) $", legend=True, filenameend=aucttype, 
            horizontal_lines=[])

    # %%
    # * median 
    var = 'winner_bid_median'
    f, a = plot(ts_ob_bl_1, var, normalization_var= 'PX_Last', maturity = maturity,
                initial_stat = "(median) $", legend=True, filenameend=aucttype + '_netbid')
    
    # %%

    f, a = plot(ts_ob_bl_collapsed, var, normalization_var= 'PX_Last', maturity = maturity,
                initial_stat = "(median) $", legend=True, filenameend='_netbid')
    
    # ******************* aggregated prices by day ********************************** #
    # %%
    # ts_ob_bl_1_collapse
    # * mean (w if weighted)
    var = 'w_winner_bid_mean_n'    # df_tba df_bl_2020_ts ts_all_agg ts
    f, a = plot(ts_ob_bl_1_collapse, var, maturity = maturity, varrate='',
                initial_stat = "(mean) $ ", legend=True, filenameend=aucttype + '_netbid')

    
    # ************************ Distress signs in the OB auctions    ********************************** #
    # %%
    # * days to auction
    var = 'DaysToAuction_mean'
    f, a = plot(ts_ob_bl_1, var, maturity = maturity,
                initial_stat = "Days to auction", legend=True, 
                filenameend=aucttype,  empty_label = True)

    plot(df_ts_month, var, maturity, varrate = '', 
            initial_stat = "Days to auction", empty_label = True, legend = True)


    # %%
    # # * dummy sell any
    # var = 'dummy_sell_any_mean'
    # plot(ts_ob_bl, var, maturity, initial_stat = "Rate sell", empty_label = True, filenameend=aucttype )

    # %%
    # * dummy sell winner
    var = 'dummy_sell_winner_mean'
    plot(ts_ob_bl_1, var, maturity, initial_stat = "Rate sell to winner", empty_label = True, 
        horizontal_lines= [],
        filenameend=aucttype, legend = True)

    plot(df_ts_month, var, maturity, varrate = '', initial_stat = "Rate sell to winner", empty_label = True, legend = False)

    # %%
    # * number of participants
    var = 'Number of Participants_mean'
    plot(ts_ob_bl_1, var, maturity, initial_stat = "Number of participants", empty_label = True, legend = True, 
         horizontal_lines= [], filenameend=aucttype)

    plot(df_ts_month, var, maturity, varrate = '', initial_stat = "Number of participants", empty_label = True, legend = False)

    # %%
    
    # * number of enterprise bidders
    var = 'Number of Enterprise Bidders_mean' 
    plot(df_ts_month, var, maturity, varrate = '',initial_stat = "Number of enterprise bidders",
            empty_label = True,        
            save = False)

    # %%
    # * number of bulk bidders
    var = 'Number of Bulk Bidders_mean'
    plot(df_ts_month, var, maturity, varrate = '', initial_stat = "Number of bulk bidders", empty_label = True, save = True)

    # by coupon
    plot(ts, var, maturity, initial_stat = "Number of bulk bidders", empty_label = True, save = True, legend="True")

    # %%
    # * bulk bidders fraction

    var = 'bulk_bidders_fraction_mean'
    plot(df_ts_month, var, maturity,  varrate = '', initial_stat = "Bulk bidders fraction", empty_label = True, save = True)

    # by coupon
    plot(ts, var, maturity, initial_stat = "Bulk bidders fraction", empty_label = True, save = True, legend="True")

    # %%
    # * Enterprise sold
    #! maybe this makes more sense for all bids not only auction type, not separate coupons
    f,a = plot(df_ts_month, 'sold_FannieBid_mean', maturity, initial_stat = "Fraction sold", empty_label = True, 
                color = 'tab:blue', legend=True, legendlabel = 'Fannie Mae', save=False, varrate = '',)
    f,a = plot(df_ts_month, 'sold_FreddieBid_mean', maturity, initial_stat = "fraction sold", empty_label = True,
                varrate = '', fig = f, ax = a, color = 'tab:orange', legend=True, legendlabel = 'Freddie Mac', save=True)
    # f,a = plot(ts, 'sold_GinnieBid_mean', maturity, initial_stat = "fraction sold", fig = f, ax = a, color = 'tab:green', legend=True, legendlabel = 'Ginnie Mae', save=True)

    # %%
    # * Enterprise sold coupon 2.5
    coupon = 2.5
    ts_coupon = ts[ts['Coupon'] == coupon].copy()
    #! maybe this makes more sense for all bids not only auction type, not separate coupons
    f,a = plot(ts_coupon, 'sold_FannieBid_mean', maturity, initial_stat = "Fraction sold", empty_label = True, 
                color = 'tab:blue', legend=True, legendlabel = 'Fannie Mae', save=False, varrate = '', filenameend=f'c{coupon*10}')
    f,a = plot(ts_coupon, 'sold_FreddieBid_mean', maturity, initial_stat = "fraction sold", empty_label = True,
                varrate = '', fig = f, ax = a, color = 'tab:orange', legend=True, legendlabel = 'Freddie Mac', save=True, filenameend=f'c{coupon*10}')
    # %% 
    # * sold GSE
    plot(df_ts_month, 'sold_GSE_mean', maturity, initial_stat = "Fraction sold to GSE", empty_label = True, 
                color = 'tab:blue', save=True, varrate = '')
    # %%
     # * sold GSE by coupon
    plot(ts, 'sold_GSE_mean', maturity, initial_stat = "Fraction sold to GSE", empty_label = True, legend = True,
                color = 'tab:blue', save=True)
    
    # %%
    # 
    # * GSE prices mean (w if weighted)
    
    ts_ob_bl_collapsed['price_diff'] = ts_ob_bl_collapsed['price_fanny_mean'] - ts_ob_bl_collapsed['price_freddie_mean']
    # calculate loan amount diff
    ts_ob_bl_collapsed['sold_FF_diff'] = ts_ob_bl_collapsed['sold_FannieBid_mean'] - ts_ob_bl_collapsed['sold_FreddieBid_mean']
    ts_ob_bl_collapsed_c25 = ts_ob_bl_collapsed[ts_ob_bl_collapsed['Coupon'] == 2.5].copy()

    

    f,a = plot(ts_ob_bl_collapsed_c25, 'price_fanny_mean', maturity, initial_stat =  "Highest bid (mean) difference", empty_label = True, normalization_var= 'PX_Last' ,
                color = 'tab:blue', legend=True, legendlabel = 'Fannie Mae', save=False, varrate = '',)
    f,a = plot(ts_ob_bl_collapsed_c25, 'price_freddie_mean', maturity, initial_stat =   "Highest bid (mean) difference", empty_label = True, normalization_var= 'PX_Last' ,
                varrate = '', fig = f, ax = a, color = 'tab:orange', legend=True, legendlabel = 'Freddie Mac', save=True,
                filenameend='byGSE_c25')
    # %%
    # * difference F F prices

    plot(ts_ob_bl_collapsed, 'price_diff', maturity, initial_stat =  "Highest bid (mean) difference", empty_label = True, 
                color = 'tab:blue', save=True, varrate = "Coupon", filenameend='diffFF')
    
    plot(ts_ob_bl_collapsed_c25, 'price_diff', maturity, initial_stat =  "Highest bid (mean) difference", empty_label = True, 
                color = 'tab:blue', save=True, varrate = "Coupon", filenameend='diffFF_c25')
    # moving average
    # %%
    # * moving average
    nma = 3
    ts_ob_bl_collapsed_c25['price_diff_ma'] = ts_ob_bl_collapsed_c25['price_diff'].rolling(window=nma).mean()
    f,a = plot(ts_ob_bl_collapsed_c25, 'price_diff_ma', maturity, initial_stat = "highest bid (mean) difference", empty_label = True,   
                color = 'tab:blue', save=True, varrate = '', filenameend=f'diffFF_ma{nma}_c25')
    # add sold diff -> this you have to do by using differenty axis #Todo
    # plot(ts_ob_bl_collapsed_c25, 'sold_FF_diff', maturity, initial_stat = "", empty_label = True,

    # %%
    # * Number of banks in the auction
    var = 'number_banks_mean'
    plot(df_ts_month, var, maturity, varrate = '', initial_stat = "Number of banks in auction",
            empty_label = True,        
            save = False)
    # %%
    # * fraction of banks in the auction
    var = 'fraction_banks_mean'
    plot(df_ts_month, var, maturity, varrate = '', initial_stat = "Fraction bank bidders", empty_label = True, save = True)

    # by coupon
    plot(ts, var, maturity, initial_stat = "fraction bank bidders", empty_label = True, save = True, legend="True")

    
    # %%
    # ***************************** Coupons note rates over time ******************************************* #
    
    coupon = 2.5
    ts_coupon = ts_ob_bl_collapsed[ts_ob_bl_collapsed['Coupon'] == coupon].copy()

    # * by coupon: 2.5, 3.0, 3.5, 4.0
    f,a = plot(ts_coupon, 'NoteRate_min', maturity, initial_stat =  "Note rate", empty_label = True,
                color = 'tab:blue', save=True, varrate = "", filenameend=f'c{coupon*10}', linestyle='--', horizontal_lines = [])
    
    f,a = plot(ts_coupon, 'NoteRate_max', maturity, initial_stat =  "Note rate", empty_label = True, 
                color = 'tab:blue', fig = f, ax = a, save=True, varrate = "", filenameend=f'c{coupon*10}', linestyle='--', horizontal_lines = [])
    
    f,a = plot(ts_coupon, 'NoteRate_mean', maturity, initial_stat =  "Note rate", empty_label = True, 
                color = 'tab:blue', fig = f, ax = a, save=True, varrate = "", filenameend=f'c{coupon*10}',  horizontal_lines = [])

    # %%
    # ****************************************** Other plots:  ******************************************* #
    # %%
    ts_extended.columns

    # %%
    # *************************** Loan amount area graphs all coupons ********************************** #
    df_coupons = ts_extended[['Coupon', 'FirstMonthYear', 'LoanAmount_sum']].copy()

    date_init_ = '2019-10-01'

    date_end_ = '2021-06-30'

    df_coupons = df_coupons[(df_coupons.FirstMonthYear >= date_init_) & (df_coupons.FirstMonthYear <= date_end_)]

    # uses ts_extended
    # graph style taken from fed ob script

    for c in df_coupons.Coupon.unique():
        for m in df_coupons.FirstMonthYear.unique():

            # if row c, m is not in df_coupons, then add it with 0 in fed_trade_amount
            if df_coupons[(df_coupons.Coupon == c) & (df_coupons.FirstMonthYear == m)].shape[0] > 0:
                continue
            else:
 
                df_coupons = pd.concat([df_coupons, 
                                        pd.DataFrame({'Coupon': [c], 'FirstMonthYear': [m], 'LoanAmount_sum': [0]})], 
                                       ignore_index=True)
                # print("Added " , c, m)


    # create dictionary with key = coupon and value = array of trade per month

    dict_amount_per_coupon = {}

    for c in df_coupons.Coupon.unique():
        # order by FirstMonthYear
        df_coupons = df_coupons.sort_values('FirstMonthYear')
        current_coupon = df_coupons[df_coupons.Coupon == c]
        dict_amount_per_coupon[c] = current_coupon.LoanAmount_sum.values
        # print length
        print(c, " - ", len(dict_amount_per_coupon[c]))

    # order by coupon
    dict_amount_per_coupon = dict(sorted(dict_amount_per_coupon.items()))

    # %%
    fig, ax = plt.subplots()

    monthyear = df_coupons.FirstMonthYear.sort_values().unique()

    ax.stackplot(monthyear, 
                dict_amount_per_coupon.values(),
                labels=dict_amount_per_coupon.keys(), alpha=0.7)
    
    plt.subplots_adjust(top=0.925, 
                bottom=0.20, 
                left=0.12, 
                right=0.96, 
                hspace=0.01, 
                wspace=0.01)

 # create list of ticks and increse the onth by 3 for i in range(min_d.month, max_d.month + 1, 3)]
    all_dates_df =df_coupons.FirstMonthYear.sort_values().unique()
    # convert to datetime
    all_dates_df = [pd.to_datetime(x) for x in all_dates_df]
    # now format dates to only show year-month
    list_ticks = [x.strftime('%Y-%m') for x in all_dates_df]
    # leave only first quarter
    list_ticks = [x for i,x in enumerate(list_ticks) if i % 3 == 0]
    all_dates_df = [x for i,x in enumerate(all_dates_df) if i % 3 == 0]
    print(list_ticks)

    # pass ticks to xticks
    plt.xticks(all_dates_df, list_ticks, rotation=45)
    
     # reorder legend last is first 
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='upper left',title='Coupon')

    ax.set_title('Monthly trade amount by coupon')
    ax.set_ylabel('Trade amount (million $)')
    ax.set_xlabel('Year-Month')
    plt.savefig(f'{auction_save_folder}/ob_monthly_trade_amount_by_coupon_all_area.png', dpi=300)

    # %%
    # ***************************************** Note rate version ******************************************* #
    # * Produce the same graph but by note rates

    #  /project/houde/mortgages/QE_Covid/data/data_auction/clean_data/timeseries/combined_auctions_jan2018-jul2022_cleaned_mat30_loan1_timeseries_NoteRate_MonthYear.csv
    var_rate = 'NoteRate'

    filename_timeseries_coup = f'{auction_filename}_mat{maturity}_loan{loantype}_timeseries_{var_rate}_{var_time}'


    df_ts = read_data(file = f'timeseries/{filename_timeseries_coup}',
                        path = auction_data_folder,
                        datetime_vars=['FirstMonthYear'])


    ts_extended = tide_auction_data(df_ts,
                        varrate = var_rate,
                        interval= interval_extended,
                        setrates= [])
    # n

    # %%
    df_noterates = ts_extended[['NoteRate', 'FirstMonthYear', 'LoanAmount_sum']].copy()

    df_noterates = df_noterates[(df_noterates.FirstMonthYear >= date_init_) & (df_noterates.FirstMonthYear <= date_end_)]

    # uses ts_extended

    for c in df_noterates.NoteRate.unique():
        for m in df_noterates.FirstMonthYear.unique():

            # if row c, m is not in df_coupons, then add it with 0 in fed_trade_amount
            if df_noterates[(df_noterates.NoteRate == c) & (df_noterates.FirstMonthYear == m)].shape[0] > 0:
                continue
            else:
 
                df_noterates = pd.concat([df_noterates, 
                                        pd.DataFrame({'NoteRate': [c], 'FirstMonthYear': [m], 'LoanAmount_sum': [0]})], 
                                       ignore_index=True)


    # create dictionary with key = noterate and value = array of trade per month

    dict_amount_per_noterate = {}

    for c in df_noterates.NoteRate.unique():
        # order by FirstMonthYear
        df_noterates = df_noterates.sort_values('FirstMonthYear')
        current_noterate = df_noterates[df_noterates.NoteRate == c]
        dict_amount_per_noterate[c] = current_noterate.LoanAmount_sum.values
        # print length
        print(c, " - ", len(dict_amount_per_noterate[c]))

    # order by noterate
    dict_amount_per_noterate = dict(sorted(dict_amount_per_noterate.items()))

    # %%
    fig, ax = plt.subplots()

    monthyear = df_noterates.FirstMonthYear.sort_values().unique()

    ax.stackplot(monthyear,
                dict_amount_per_noterate.values(),
                labels=dict_amount_per_noterate.keys(), alpha=0.7)
    
    plt.subplots_adjust(top=0.925,
                bottom=0.20,
                left=0.12,
                right=0.96,
                hspace=0.01,
                wspace=0.01)
    
    # create list of ticks and increse the onth by 3 for i in range(min_d.month, max_d.month + 1, 3)]
    all_dates_df = [pd.to_datetime(x) for x in monthyear]
    list_ticks = [x.strftime('%Y-%m') for x in all_dates_df]
    list_ticks = [x for i,x in enumerate(list_ticks) if i % 3 == 0]
    all_dates_df = [x for i,x in enumerate(all_dates_df) if i % 3 == 0]
    print(list_ticks)

    # pass ticks to xticks
    plt.xticks(all_dates_df, list_ticks, rotation=45)

    # ax.legend(loc='upper left',title='Note rate')
    # legend outside 
    # ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1),title='Note rate')

    # reorder legend last is first 
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='upper left', bbox_to_anchor=(1.05, 1),title='Note rate')
    # legend name Coupon

    ax.set_title('Monthly trade amount by note rate')
    ax.set_ylabel('Trade amount (million $)')
    ax.set_xlabel('Year-Month')
    plt.savefig(f'{auction_save_folder}/ob_monthly_trade_amount_by_noterate_all_area_legend_colors.png', dpi=300)    # %%

    # %%
    # * version of graph with note rates is is a color map


    # ****************************** Table Auctions  ********************************* #

    # * read bid level data
    # %%
    # combined_auctions_jan2018-jul2022_cleaned_mat30_loan1
    filename = f'{auction_filename}_mat{maturity}_loan{loantype}'

    df = read_data(file = filename, path = auction_data_folder, 
                   datetime_vars=['BorrowerClosingDate', 'CommittedDate' ])


    # %%
    df.columns
    
    # %%
    # dates range see
    df['CommittedDate'].describe()
    # %%
    # * bids distribution
    # plot distribution of bids , var is Price, soft color, transparecy , no vertical line, bo bakground vertical lines
    df['Price'].hist(bins=40, color = 'tab:blue', alpha = 0.5, edgecolor='black', linewidth=1.2, grid=False)
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    plt.title('Distribution of bids')
    plt.savefig(f'{auction_save_folder}/distribution_of_bids.png', dpi=300)
    plt.show()  


    # %%


    # *********************** read auction level data ******************************* #
    # %%
    
    # read 
    filename = f'{auction_filename}_mat{maturity}_loan{loantype}_auction_level'
    df = read_data(file = filename, path = auction_data_folder,
                   datetime_vars=['BorrowerClosingDate', 'CommittedDate' ])
    df.columns

    # %%
    table = create_table_stats(df, additional_name='all')

    # %%
    # repeat for January and February only
    df1 = df[df['CommittedDate'] < pd.to_datetime('2020-03-01')]
    table1 = create_table_stats(df1, additional_name='jan-feb', 
                                stats = ['count', 'mean', 'std', 'min', 'max'])
    table1

    # %%
    # march april
    df2 = df[(df['CommittedDate'] >= pd.to_datetime('2020-03-01')) 
                & (df['CommittedDate'] < pd.to_datetime('2020-05-01'))]
    table2 = create_table_stats(df2, additional_name='mar-apr',
                                stats = ['count', 'mean', 'std', 'min', 'max'])
    table2

    # %%
    # * Note Rate distribution
    # plot distribution of bids , var is Price, soft color, transparecy , no vertical line, bo bakground vertical lines
    df['NoteRate'].hist(bins=40, color = 'tab:blue', alpha = 0.5, edgecolor='black', linewidth=1.2, grid=False)
    plt.xlabel("Note rate")
    plt.ylabel('Frequency')
    plt.title('Distribution of note rates')
    plt.savefig(f'{auction_save_folder}/distribution_of_noterates.png', dpi=300)

    
    # *********************** end of main ******************************* #



# %%



if __name__ == '__main__':

    main()