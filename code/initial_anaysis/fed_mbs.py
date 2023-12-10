"""
Created on Thu June 1, 2023
@author: Giselle Labrador Badia (@gisslab)

This module explores the FED purchases of MBS by cusip, coupons and maturities.

"""

#%%
# * libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
import os
import auction_prices_timeseries_plots as apts


#%%

# * settings

# set directory
fed_data = "/project/houde/mortgages/QE_Covid/data/data_fed"

auction_save_folder = '/project/houde/mortgages/QE_Covid/results/'

filename = "MBS_data_new.csv"

date_init = '2019-10-01'

date_end = '2021-06-30'

agencies = ['FNMA', 'FHLMC']

maturity = 30

cols = ['tradedate', 
       'contractualsettlementdate', 
       'transactioncategory',
       'operationtype', 
       'agency', 
       'couponinpercent', 'terminyears', 'cusip',
       'price', 'totalamounttransferredinmillions', 'counterparty',
       'tradeamount', 
        # 'auctionstatus', #! only in matched (3) and using only (2)
        # 'operationid', #! only in matched (3) and using only (2)
        # 'operationdate', #! only in matched (3) and using only (2)
    #    'operationdirection', 'auctionmethod', 'classtype', #! only in matched (3) and using only (2)
    #    'settlementdate', #! only in matched (3) and using only (2)
    #    'totalamtacceptedcurrentmillions', 'totalamtacceptedoriginalmillions',
    #    'totalamtsubmittedcurrentmillions', 'totalamtsubmittedoriginalmillion',
    #    'totalamtacceptedparmillions', 'totalamtsubmittedparmillions',
    #    'inclusionexclusion', 
        # 'securitydescription', #! only in matched (3) and using only (2)
    #    'specpoolcurrentfaceamtaccepted', 'specpooloriginalfaceamtaccepted',
    #    'tbaparamountaccepted', 'note', 
    '_merge'
       ]

# Is this auction level data? Transaction level data? (it does not seems to have bid level data)

# some info about how populated columns are in teh FED's data
# RangeIndex: 86793 entries, 0 to 86792
# Data columns (total 18 columns):
#  #   Column                            Non-Null Count  Dtype  
# ---  ------                            --------------  -----  
#  0   tradedate                         86793 non-null  object 
#  1   contractualsettlementdate         86653 non-null  object 
#  2   transactioncategory               79128 non-null  object 
#  3   operationtype                     86793 non-null  object 
#  4   agency                            79128 non-null  object 
#  5   couponinpercent                   79128 non-null  float64
#  6   terminyears                       79128 non-null  float64
#  7   cusip                             79156 non-null  object 
#  8   price                             79128 non-null  float64
#  9   totalamounttransferredinmillions  61167 non-null  float64
#  10  counterparty                      61167 non-null  object 
#  11  tradeamount                       79128 non-null  float64
#  12  auctionstatus                     7838 non-null   object 
#  13  operationid                       7838 non-null   object 
#  14  operationdate                     7838 non-null   object 
#  15  settlementdate                    7698 non-null   object 
#  16  securitydescription               7838 non-null   object 
#  17  _merge                            86793 non-null  object 


# os.chdir(fed_data)

# * functions

def read_data(file, path, cols = cols, maturity = maturity):
    """
    Reads the data from the specified path using the specified columns.
    """
    print(f'Reading {file} from {path}')

    try:
        df = pd.read_csv(f'{path}/{file}', usecols = cols)
    except Exception as e:
        print(f'Could not read {file} from {path}')
        print(e)
        df = None
    return df

def process(df_raw, 
            date_init = date_init,
            date_end = date_end):
    """
    Processes the data by filtering by date, maturity and agency.
    """
    df = df_raw.copy()

    # delete elements with missing dates
    df = df.loc[df['tradedate'].notnull() & df['contractualsettlementdate'].notnull(), :]

    # rename couponinpercent
    df.rename(columns = {'couponinpercent': 'Coupon'}, inplace = True)

    # dates to datetime format
    df['tradedate'] = pd.to_datetime(df['tradedate'])
    df['contractualsettlementdate'] = pd.to_datetime(df['contractualsettlementdate'])

    # calclate forward months
    df['forwarddays'] = (df['contractualsettlementdate'] - df['tradedate']) / np.timedelta64(1, 'D')
    # df['forwardmonths'] = (df['contractualsettlementdate'] - df['tradedate']) / np.timedelta64(1, 'M')
        # Forward Trading Months
    df['forwardmonths'] = [relativedelta(x['contractualsettlementdate'], x['tradedate']).months for i,x in df.iterrows()]


    # in some cases totalamounttransferredinmillions is null, but tradeamount is not, so we use tradeamount
    df['tradeamount_millions'] = df['tradeamount'] / (10**6)
    # dates 
    df = df[(df['tradedate'] >= date_init) & (df['tradedate'] <= date_end)]

    #  terminyears = 30 
    df = df[df['terminyears'] == maturity]

    df = df.loc[df['agency'].isin(agencies), :]

    # create Month year column, alternatively by contractualsettlementdate
    df['MonthYear'] = (
        df["tradedate"].dt.month_name() + "-" + df["tradedate"].dt.year.astype(str)
    )

    # create date time column withfirst day of Committed date: e.g. 2020-01-22 -> 2020-01-01
    df['FirstMonthYear'] = df['tradedate'].dt.to_period('M').dt.to_timestamp()

    return df

def collapse(df, on = ['FirstMonthYear', 'Coupon', 'forwardmonths']):
    """
    Collapses the data by the specified arguments. This is for the FED data, variables are: price, totalamounttransferredinmillions, tradeamount_millions, cusip.
    """

    df_g = df.groupby(on).agg(
        {
            "totalamounttransferredinmillions": ["sum"],
            "tradeamount_millions": ["sum"],
            "price": ["mean", "median"],
            "cusip": ["count"],
        }
    ).reset_index()

    # rename columns
    df_g.columns = on + [
                    'fed_amount_transfer',
                    'fed_trade_amount',
                    'fed_price_mean',
                    'fed_price_median',
                    'fed_cusip_count'
                    ]

    return df_g

def merge_bl_mbs(df_bl, df_fed, on = ['FirstMonthYear', 'Coupon', 'forwardmonths']):
    """
    Merge bloomberg and OB data on arguments passed on.
    """
    df_bl_ = df_bl.rename(columns = {'Forward_Trading_Months': 'forwardmonths'})

    df = pd.merge(df_bl_, df_fed, on = on, how = 'right')

    return df



def plot_simple_coupon(df, agencies = agencies, on = ['tradedate', 'Coupon'], addname = 'daily'):
    """
    Plots price and amount by coupon for the specified agencies. Uses both tradedate and contractualsettlementdate.
    """

    # df = df.loc[df['agency'].isin(agencies), :]
    # * daily graph of purchases by coupon, totalamounttransferredinmillions, price
    # add ylabels
    # agency label: concatenate agencies
    agency = '_'.join(agencies)
    # get frm column name trade amount in text
    var_amount_trade = [col for col in df.columns if 'trade_amount' in col][0]
    # price
    # var price mean 
    var_price_mean = [col for col in df.columns if 'price_mean' in col][0]

    g = df.groupby(on)[var_amount_trade
                    ].sum().unstack().plot()
    g.set_ylabel('Total amount transferred in millions')
    g.set_xlabel('Trade Date')
    g.figure.savefig(f'{auction_save_folder}/figures/{agency}_purchases_tradedate_amount_{addname}.png')
    print(f'Saved figure: {auction_save_folder}/figures/{agency}_purchases_tradedate_amount_{addname}.png')

    # * price
    g = df.groupby(on)[
                    var_price_mean].mean().unstack().plot()
    g.set_ylabel('Avg price')
    g.set_xlabel('Trade Date')
    g.figure.savefig(f'{auction_save_folder}/figures/{agency}_purchases_tradedate_price_{addname}.png')

    #daily graph of purchases by coupon, totalamounttransferredinmillions, price

    # g = df.groupby(on)[
    #                 'totalamounttransferredinmillions'].sum().unstack().plot()
    # g.set_ylabel('Total amount transferred in millions')
    # g.set_xlabel('Contractual Settlement Date')
    # g.figure.savefig(f'{auction_save_folder}/figures/{agency}_purchases_contractualsettlementdate_amount.png_{addname}')

    # # * price
    # g = df.groupby(on)[
    #                 'price'].mean().unstack().plot()
    # g.set_ylabel('Avg price')
    # g.set_xlabel('Contractual Settlement Date')
    # g.figure.savefig(f'{auction_save_folder}/figures/{agency}_purchases_contractualsettlementdate_price.png_{addname}')


#%%

# * main


if __name__ == '__main__':

    # %%
    # read data
    df_raw = read_data(file = filename, path = f'{fed_data}/raw_data/')

    # %%
    df_raw.agency.value_counts()
    # FNMA     49229  -> Fannie Mae, the Federal National Mortgage Association (FNMA)
    # GNMA2    18844 -> Ginnie Mae, Government National Mortgage Association.
    # FHLMC    10507 -> Freddie Mac,  Federal Home Loan Mortgage Corp. (FHLMC).
    # GNMA       548 -> Ginnie Mae, Government National Mortgage Association.

    # %% 
    # see matched (3)
    df_raw._merge.value_counts()
    # Master only (1)    78955
    # Using only (2)      7665
    # Matched (3)          173

    df_raw[df_raw._merge == 'Matched (3)'].head(10)
    # the only matched seemed to me sales 
    # %%
    df_raw.transactioncategory.value_counts()
    # Purchase    77412
    # Sale         1716
    # %%
    # * process data
    df = process(df_raw)

    # %%
    df.head(15)
    # %%
    df.columns 

    # %%
    # * banks that sell or buy to FED
    print("numer of unique banks: ", df.counterparty.nunique())
    df.counterparty.value_counts()

    # %%
    ## nans in counterparty
    print("Number of nans in counterparty: ", df.counterparty.isnull().sum(), "from ", df.shape[0], "observations")
    # %%
    df.describe()

    # %%
    # See pool of forward days by forward months
    df.groupby('forwardmonths')['forwarddays'].describe()
    # It's better to use only one month forward to compare with TBA market. 
    # See that most elements in sample are one month forward (28 to 60 days)

    # %%
    # see rows where totalamounttransferredinmillions is null, but tradeamount is not
    df1 = df[df['totalamounttransferredinmillions'].isnull() & df['tradeamount'].notnull()]
    # not sure why this happens, most of the cases Purchases. 
    #? why is trade amount greater than total amount transferred?
    # because is transfer amount is the amount of the transaction, but trade amount is the amount of the trade

    # %%
    # how many forward months greated than 1.9 (around 60 days)
    x = 25
    y = 45
    df_2_months = df[(df['forwarddays'] >=  x) & (df['forwarddays'] <= y)]
    # 2 months is the max forward months
    print(f"Number of observations with forward months greater than {x} days: ", df_2_months.shape[0])
    print(f"Percentage of FED auction with forward months greater than {x} das: ", round((df_2_months.shape[0] / df.shape[0]), ndigits = 5)* 100, "%")


    # %%

    # * collapse FED
    df_fed_monthly = collapse(df)
    df_fed_monthly.head(10)

    # %%
    df_fed_monthly.describe()

    # %%
    # ******************** Bloomberg data ******************** #

    # %%
    df_bl = apts.read_data(file = apts.bloomberg_filename, path = apts.bl_data_folder, 
                           datetime_vars= ['Trading_Date', 'Settlement_Date'],)
    
    df_bl['forwarddays'] = (df_bl['Settlement_Date'] - df_bl['Trading_Date']) / np.timedelta64(1, 'D')

    df_bl.describe()
    # %% 
    df_bl_clean = apts.tide_collapse_bloomberg_data(df_bl,
                                                    min_date= date_init,
                                                    max_date= date_end,
                                                    forward_months= [0,1,2],
                                                    group_by= ['Coupon', 'FirstMonthYear', 'Forward_Trading_Months']
                                                    )


    # %%
    # ******************* merge bloomberg and FED ******************* #

    df_merged = merge_bl_mbs(df_bl_clean, df_fed_monthly)

    # save df merge 
    df_merged.to_csv(f'{fed_data}/clean_data/fed_bl_forward_coupon_monthly.csv', sep='|', index=False)

    # %%
    df_merged.info()

    # %%
    df_merged.describe()
    
    # %%
    # plot were PX_last is not null
    df_merged[df_merged.PX_Last.notnull()]

    # %%
    df_null = df_merged[df_merged.PX_Last.isnull()]
    df_null.describe()
    #! For coupons 1.5 and 2.0 there are no Bloomberg prices in some months 

    # %%
    # *********************************************************************************************
    # ****************************** Plot ************************************************************
    # *********************************************************************************************

    # %%
    # plot_simple_coupon(df_fed_monthly,
    #                     on = ['FirstMonthYear', 'Coupon'], 
    #                     addname='monthly')
    # %%

    # only plot 1 month forward and coupons 2.5, 3.0, 4.0
    df_merged_1m = df_merged[(df_merged.forwardmonths == 1)]
    # df_merged_1m = df_merged_1m[df_merged_1m.Coupon.isin([2.5, 3.0, 4.0])]

    # %% 
    # * df_bl distribution of forward days
    df_bl.forwarddays.hist(bins=40, color = 'tab:blue', alpha = 0.5, edgecolor='black', linewidth=1.2, grid=False)
    # paint
    plt.xlabel('Forward days (settlement date - trade date)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Bloomberg forward days')
    plt.savefig(f'{auction_save_folder}/figures/distribution_of_forwarddays_bloomberg.png', dpi=300)

    # %%
    # * plot histogram of forward days
    df.forwarddays.hist(bins=40, color = 'tab:blue', alpha = 0.5, edgecolor='black', linewidth=1.2, grid=False)
    # paint 
    plt.xlabel('Forward days (settlement date - trade date)')
    plt.ylabel('Frequency')
    plt.title('Distribution of FED auctions forward days (Fannie Mae and Freddie Mac)')
    plt.savefig(f'{auction_save_folder}/figures/distribution_of_forwarddays_FED.png', dpi=300)
    plt.show()  

    # * Plot price and amount by coupon

    # %%

    # seleected coupons
    df_merged_1m_coup = df_merged_1m[df_merged_1m.Coupon.isin([2.5])]
    # use plot in apts
    # * trade 
    apts.plot(df_merged_1m_coup,
                maturity = maturity,
                vertical_lines=[],
                var = 'fed_trade_amount',
                initial_stat = "Trade amount (million $)", 
                empty_label = True,
                legend = True,
                interval= None,
                filenameend="coup2.5"                
    )

    # %%
    # * price mean
    apts.plot(df_merged_1m_coup,
                maturity = maturity,
                vertical_lines=[],
                horizontal_lines=[0],
                var = 'fed_price_mean',
                initial_stat = "Mean net price ($)",
                empty_label = True,
                legend = True,
                interval= None,
                normalization_var= 'PX_Last',
                filenameend="normalized_coup2.5"    
    )   
    # %%
    # * price mean
    apts.plot(df_merged_1m_coup,
                maturity = maturity,
                vertical_lines=[],
                horizontal_lines=[],
                var = 'fed_price_mean',
                initial_stat = "Mean price ($)",
                empty_label = True,
                legend = True,
                interval= None, 
                filenameend="coup2.5"
    )   

    # %%
    # by coupon 
    apts.plot(df_merged_1m,
            maturity = maturity,
            vertical_lines=[],
            horizontal_lines=[0],
            var = 'fed_price_mean',
            initial_stat = "Mean price ($)",
            empty_label = True,
            legend = True,
            interval= None, 
            normalization_var= 'PX_Last',
            filenameend=""
    )   

    # %%
    # **************************** scatter price by month ************************************ #
    # by coupon
    # df_merged_1m = df_merged_1m[df_merged_1m.Coupon.isin([ 4.0])]

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink']
    i = 0 
    for c in df_merged_1m.Coupon.sort_values().unique():
        print(c)

        df_merged_1m_ = df_merged_1m[df_merged_1m.Coupon == c]

        # if only one sample then plot a point
        if df_merged_1m_.shape[0] == 1:
            plt.scatter(df_merged_1m_.FirstMonthYear, df_merged_1m_.fed_price_mean, alpha = 0.7, label = c, color = colors[i])
        else: 
            plt.plot(df_merged_1m_.FirstMonthYear, df_merged_1m_.fed_price_mean, alpha = 0.7, label = c, color = colors[i])

        i += 1
    
    plt.subplots_adjust(top=0.925, 
            bottom=0.20, 
            left=0.12, 
            right=0.96, 
            hspace=0.01, 
            wspace=0.01)
    
    all_dates_df = [pd.to_datetime(x) for x in monthyear]
    list_ticks = [x.strftime('%Y-%m') for x in all_dates_df]
    # leave only first quarter
    list_ticks = [x for i,x in enumerate(list_ticks) if i % 3 == 0]
    all_dates_df = [x for i,x in enumerate(all_dates_df) if i % 3 == 0]
    print(list_ticks)

    plt.xticks(all_dates_df, list_ticks, rotation=45)

    # get handler to change order legend
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[::-1], labels[::-1], loc='upper left',title='Coupon')

    plt.title('Mean price by month')
    plt.xlabel('Year-Month')
    plt.ylabel('Mean price ($)')
    plt.xticks(rotation=45)

    plt.savefig(f'{auction_save_folder}/figures/fed_monthly_price_mean_by_coupon.pdf')



    # %% 

    # **************************** Area graph ************************************ #
    # ***** Begin area graph for trade amount by coupon ***** #
    # 
    # Create nice figure were we plot the percent of trade amount in the month by coupon, all forward months
    g = ['FirstMonthYear']
    df_coupons = df_merged.groupby(['Coupon', 'FirstMonthYear'])['fed_trade_amount'].sum().reset_index()
    df_coupons.columns = [ 'Coupon', 'FirstMonthYear', 'fed_trade_amount']

    # to billions divide by 1000
    df_coupons['fed_trade_amount'] = df_coupons['fed_trade_amount'] / 1000

    print("Min date: ", df_coupons.FirstMonthYear.min())
    print("Max date: ", df_coupons.FirstMonthYear.max())

    # df_coupons['trade_amount_total'] = df_coupons.groupby(g)['fed_trade_amount'].transform('sum')

    # df_coupons['trade_amount_percent'] = df_coupons['fed_trade_amount'] / df_coupons['trade_amount_total']

    # put a zero for coupons that are not in the month
 
    # if row coupon, FirstMonthYear is not in df_coupons, then add it with 0 in fed_trade_amount
    # add rows with 0 for coupons that are not in the month

    for c in df_coupons.Coupon.unique():
        for m in df_coupons.FirstMonthYear.unique():

            # if row c, m is not in df_coupons, then add it with 0 in fed_trade_amount
            if df_coupons[(df_coupons.Coupon == c) & (df_coupons.FirstMonthYear == m)].shape[0] > 0:
                continue
            else:
 
                df_coupons = pd.concat([df_coupons, 
                                        pd.DataFrame({'Coupon': [c], 'FirstMonthYear': [m], 'fed_trade_amount': [0]})], 
                                       ignore_index=True)
                # print("Added " , c, m)


    # graphs that shows as a 100 % monthly area that represents the percent of trade amount by coupon
    # * plot
    # using stackplot 
    # create dictionary with key = coupon and value = array of trade per month

    dict_amount_per_coupon = {}

    for c in df_coupons.Coupon.unique():
        # order by FirstMonthYear
        df_coupons = df_coupons.sort_values('FirstMonthYear')
        current_coupon = df_coupons[df_coupons.Coupon == c]
        dict_amount_per_coupon[c] = current_coupon.fed_trade_amount.values
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

    

    # pass ticks to xticks
    plt.xticks(all_dates_df, list_ticks, rotation=45)
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='upper left',title='Coupon')
    # legend name Coupon
    ax.set_title('Monthly trade amount by coupon')
    ax.set_ylabel('Trade amount (billions $)')
    ax.set_xlabel('Year-Month')
    # plt.xticks(rotation=45)
    plt.savefig(f'{auction_save_folder}/figures/fed_monthly_trade_amount_by_coupon.pdf')

    # ***** End area graph for trade amount by coupon ***** #



# %%
