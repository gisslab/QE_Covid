"""
Created on Dec 5,2024
@author: Giselle Labrador Badia (@gisslab)

Build measure on exposure based on the Amount of QE purchases of MBS by the Federal Reserve. 

Input:
    - Uses scripts: initial_analysis/auction_prices_analysis.py and fed_mbs.py
    - Data on MBS purchases by the Federal Reserve
    - Data on loan auctions from Ob

Output:
    - Dataframe with exposure measure by month, coupon, forward months and counterparty

"""

# %% Import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mda
import matplotlib.ticker as mtick
# import other modules

# add paths
import sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../initial_anaysis/')

import auction_prices_analysis as apa
import fed_mbs as fmbs

# %%
init_date = '2019-01-01'
end_date = '2021-05-01'

save_fig = "/project/houde/mortgages/QE_Covid/results/figures/"
save_tab = "/project/houde/mortgages/QE_Covid/results/tables/"
# %%

# In Ob: investors names
# ['CALIBER HOME LOANS, INC.', 'JP MORGAN CHASE BANK N.A.',
#        'WELLS FARGO BANK, NA,', 'PENNYMAC LOAN SERVICES, LLC',
#        'TEXAS CAPITAL BANK', 'FLAGSTAR BANK, FSB', nan,
#        'TOWNE MORTGAGE COMPANY', 'DITECH FINANCIAL LLC', 'NEWREZ LLC',
#        'NORTHPOINTE BANK', 'HOME POINT FINANCIAL CORPORATI ON',
#        'NATIONSTAR MORTGAGE, LLC', 'U. S. BANK, NA',
#        'THE MONEY SOURCE INC.', 'PLANET HOME LENDING, LLC',
#        'LAKEVIEW LOAN SERVICING, LLC', 'AMERIHOME MORTGAGE COMPANY,LLC',
#        'GUILD MORTGAGE COMPANY LLC',
#        'FIRST GUARANTY MORTGAGE CORPORATION',
#        'DBA FREEDOM HOME MORTGAGE CORP', 'ON Q FINANCIAL, INC.',
#        'RUSHMORE LOAN MANAGEMENT SERVI CES, LLC', 'FIRSTBANK',
#        'CITIZENS BANK N.A.',
#        'GATEWAY MORTGAGE,A DIVISION OF GATEWAY FIRST BANK',
#        'ARC HOME LLC', 'SUNTRUST MORTGAGE, INC.', 'TRUIST BANK',
#        'PHH MORTGAGE CORPORATION', 'CITIMORTGAGE, INC.',
#        'CMG MORTGAGE, INC.', 'PLAZA HOME MORTGAGE, INC.', 'M&T BANK',
#        'IMPAC MORTGAGE', 'AMERICAN FINANCIAL RESOURCES, INC.', 'NEXBANK',
#        'WINTRUST MORTGAGE', 'THE HUNTINGTON NATIONAL BANK',
#        'SUN WEST MORTGAGE CO., INC.', 'VILLAGE CAPITAL & INVESTMENT, LLC',
#     #    'NATIONSTAR MORTGAGE LLC'], dtype=object)


# Morgan Stanley & Co. LLC                              
# Credit Suisse AG, New York Branch                     267385.000000
# Goldman Sachs & Co. LLC                               165578.100000
# Citigroup Global Markets Inc.                         9 # CITIMORTGAGE, INC.	
# Bofa Securities, Inc.                                 148372.700000
# J.P. Morgan Securities LLC                             97448.000064
# Nomura Securities International, Inc.                  79254.999872
# Barclays Capital Inc.                                  78578.000000
# Wells Fargo Securities, LLC                            11
# Daiwa Capital Markets America Inc.                     50863.000000
# J.P. Morgan Chase                                      12
# Wells Fargo Securities                                 17914.000000
# BNP Paribas Securities Corp.                           15209.000000
# Mizuho Securities USA LLC                              11742.000000
# Jefferies LLC                                           9382.000000
# BNP-Paribas                                             2459.000000
# BofA Securities, Inc.                                    251.000000
# Merrill Lynch, Pierce, Fenner & Smith Incorporated         1.000000

dict_bank_id = {'Citigroup Global Markets Inc.': 9,
                'J.P. Morgan Securities LLC': 12,
                'Wells Fargo Securities': 11,
                }


def cleaning_names_fed(df):
    """
    Clean names of counterparty in fed data
    """

    # Bofa Securities, Inc. -> BofA Securities, Inc.
    df['counterparty'] = df['counterparty'].replace('Bofa Securities, Inc.', 'BofA Securities, Inc.')

    # J.P. Morgan Chase <- J.P. Morgan Securities LLC , same parent company
    df['counterparty'] = df['counterparty'].replace( 'J.P. Morgan Securities LLC','J.P. Morgan Chase')

    # Wells Fargo Securities <- Wells Fargo Securities, LLC
    df['counterparty'] = df['counterparty'].replace( 'Wells Fargo Securities, LLC','Wells Fargo Securities')

    # BNP-Paribas <- BNP Paribas Securities Corp.
    df['counterparty'] = df['counterparty'].replace( 'BNP Paribas Securities Corp.','BNP-Paribas')

    # Merrill Lynch, Pierce, Fenner & Smith Incorporated -> BofA Securities, Inc.
    df['counterparty'] = df['counterparty'].replace( 'Merrill Lynch, Pierce, Fenner & Smith Incorporated','BofA Securities, Inc.')

    return df

# %%

# * dict investors -> id (Ob)


if __name__ == '__main__':

    # %%
    # read data
    # fed = pd.read_csv(fmbs.fed_data + '/raw_data/' + fmbs.filename, sep=',')
    fed = fmbs.read_data(file =  fmbs.filename, path = f'{ fmbs.fed_data}/raw_data/')
    fed.info()
    # %%
    # clean data
    fed = fmbs.process(fed, 
                        date_init = init_date,
                        date_end = end_date
                        )

    # %%
    # check dates in data, min and max
    print("Min date: ", fed['tradedate'].min())
    print("Max date: ", fed['tradedate'].max())

    # same with FirstMonthYear
    
    # %%
    fed = cleaning_names_fed(fed)
    
    # %%
    fed.columns

    # %%
    fed.transactioncategory.value_counts()

    # %% 
    # only 'PURCHASE'
    fed = fed[fed['transactioncategory'] == 'Purchase']

    # %%
    fed.counterparty.unique()
    # %%
    fed.counterparty.value_counts()

    # %%
    ## nans in counterparty
    print("Number of nans in counterparty: ", fed.counterparty.isnull().sum(), "from ", fed.shape[0], "observations")


    # %%
    # * collapse on on = ['FirstMonthYear', 'Coupon', 'forwardmonths', 'counterparty']
    fed_collapse = fmbs.collapse(fed, 
                        on = ['FirstMonthYear', 'Coupon', 'forwardmonths', 'counterparty']
                        )
    
    fed_collapse_c = fmbs.collapse(fed, 
                        on = ['FirstMonthYear', 'Coupon', 'counterparty']
                        )
    
        # add column with id
    fed_collapse['counterparty_id'] = fed_collapse['counterparty'].map(dict_bank_id)
    fed_collapse_c['counterparty_id'] = fed_collapse_c['counterparty'].map(dict_bank_id)
    fed_collapse_copy = fed_collapse.copy()
    
    # %%
    fed_collapse.columns

    # %%
    # fed collapse dates
    print("Min date: ", fed_collapse['FirstMonthYear'].min())
    print("Max date: ", fed_collapse['FirstMonthYear'].max())

    # %%
    # make counterparty nan to be 'Missing'
    fed_collapse['counterparty'] = fed_collapse['counterparty'].fillna('Missing')

    # %%
    # see by counterparty who has more purchases
    fed_collapse.groupby('counterparty').sum()['fed_trade_amount'].sort_values(ascending=False)

    # %%
    # now export to latex table
    tab = fed_collapse.groupby('counterparty').sum()['fed_trade_amount'].sort_values(ascending=False)
    tab = tab.reset_index()
    # rename columns for latex
    tab.columns = ['Counterparty', 'Amount (millions $)']
    # export to latex
    tab.to_latex(f'{save_tab}/fed_mbs_amount_by_counterparty.tex', index=False)
    # round to 0 decimals
    tab['Amount (millions $)'] = tab['Amount (millions $)'].round(0)
    tab
    # %%
    # create table with percentage of purchases by counterparty in that month-year
    fed_collapse['fed_trade_amount_month'] = fed_collapse.groupby(['FirstMonthYear', 'counterparty'])['fed_trade_amount'].transform('sum')
    fed_collapse['percentage'] = fed_collapse['fed_trade_amount_month']/fed_collapse.groupby('FirstMonthYear')['fed_trade_amount'].transform('sum')
    fed_collapse['percentage'].describe()


    # %% See Citie Group: Citigroup Global Markets Inc. by month
    fed_collapse[fed_collapse['counterparty'] == 'Citigroup Global Markets Inc.'].groupby('FirstMonthYear').sum()['fed_trade_amount'].plot()

    # %% 
    # ********************************************* 
    # *  Plot City and JP Mprgan Chase BofA Securities, Inc.

    # convert to datetime
    fed_collapse['FirstMonthYear'] = pd.to_datetime(fed_collapse['FirstMonthYear'])

    fig, ax = plt.subplots(figsize=(8, 5))
    # plot after july 2019
    fed_collapse = fed_collapse[fed_collapse['FirstMonthYear'] >= '2019-07-01']
    fed_collapse = fed_collapse[fed_collapse['FirstMonthYear'] <= '2021-03-01']

    # plot with dots in circles
    fed_collapse[fed_collapse['counterparty'] == 'Citigroup Global Markets Inc.'].groupby('FirstMonthYear').sum()['fed_trade_amount'].plot(label='Citigroup')
    fed_collapse[fed_collapse['counterparty'] == 'J.P. Morgan Chase'].groupby('FirstMonthYear').sum()['fed_trade_amount'].plot(label='J.P. Morgan Chase')
    fed_collapse[fed_collapse['counterparty'] == 'Wells Fargo Securities'].groupby('FirstMonthYear').sum()['fed_trade_amount'].plot(label='Wells Fargo')
    fed_collapse[fed_collapse['counterparty'] == 'BofA Securities, Inc.'].groupby('FirstMonthYear').sum()['fed_trade_amount'].plot(label='BofA')
    # scatter with same colors circles
    fed_collapse[fed_collapse['counterparty'] == 'Citigroup Global Markets Inc.'].groupby('FirstMonthYear').sum()['fed_trade_amount'].plot(style='o', color='tab:blue', label='_nolegend_')
    fed_collapse[fed_collapse['counterparty'] == 'J.P. Morgan Chase'].groupby('FirstMonthYear').sum()['fed_trade_amount'].plot(style='o', color='tab:orange', label='_nolegend_')
    fed_collapse[fed_collapse['counterparty'] == 'Wells Fargo Securities'].groupby('FirstMonthYear').sum()['fed_trade_amount'].plot(style='o', color='tab:green', label='_nolegend_')
    fed_collapse[fed_collapse['counterparty'] == 'BofA Securities, Inc.'].groupby('FirstMonthYear').sum()['fed_trade_amount'].plot(style='o', color='tab:red', label='_nolegend_')
    # line in march 2020

    ax.grid(axis = "y", color = "grey", alpha = 0.2, linewidth = 0.5)
    ax.axvline(x='2020-03-01', color='black', linestyle='--')
    ax.spines[['right', 'top']].set_visible(False)
    ax.legend(loc='upper left')
    ax.set_ylabel('Amount (millions $)')
    ax.set_xlabel('Month')
    ax.set_title('Amount of MBS purchased by Fed by month')
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))
    # ax.xaxis.set_major_formatter(mda.DateFormatter('%Y-%m'))


    # ax.set_xticks([ '2020-01-01','2021-01-01'])
    # tick labels
    # ax.set_xticklabels(['Jan 2020', 'Jan 2021'], rotation=0)

    #7 ticks
    # ax.xaxis.set_major_locator(plt.MaxNLocator(7))
    ax.set_xticks([ '2019-07-01','2019-10-01','2020-01-01','2020-04-01','2020-07-01','2020-10-01','2021-01-01'])
    ax.set_xticklabels(['  Jul', '  Oct', '      Jan 2020', '  Apr', '   Jul',  '   Oct', '      Jan 2021'], rotation=0)
    # shift to right labels
    # ax.tick_params(axis='x', pad=10)

    plt.show()

    # save
    fig.savefig(f'{save_fig}/fed_mbs_amount_by_month_example_retail.pdf')

    # %% 
    # * Repat now for five 4 investors 
    # Morgan Stanley & Co. LLC                 305283.000000
    # Credit Suisse AG, New York Branch        251325.000000
    # Citigroup Global Markets Inc.            147147.000000
    # Goldman Sachs & Co. LLC                  135437.000000
    fed_collapse = fed_collapse[fed_collapse['FirstMonthYear'] <= '2021-03-01']
    fig, ax = plt.subplots(figsize=(8, 5))

    fed_collapse[fed_collapse['counterparty'] == 'Morgan Stanley & Co. LLC'].groupby('FirstMonthYear').sum()['fed_trade_amount'].plot(label='Morgan Stanley')
    fed_collapse[fed_collapse['counterparty'] == 'Credit Suisse AG, New York Branch'].groupby('FirstMonthYear').sum()['fed_trade_amount'].plot(label='Credit Suisse')
    fed_collapse[fed_collapse['counterparty'] == 'Citigroup Global Markets Inc.'].groupby('FirstMonthYear').sum()['fed_trade_amount'].plot(label='Citigroup')
    fed_collapse[fed_collapse['counterparty'] == 'Goldman Sachs & Co. LLC'].groupby('FirstMonthYear').sum()['fed_trade_amount'].plot(label='Goldman Sachs')
    # # scatter with same colors circles
    fed_collapse[fed_collapse['counterparty'] == 'Morgan Stanley & Co. LLC'].groupby('FirstMonthYear').sum()['fed_trade_amount'].plot(style='o', color='tab:blue', label='_nolegend_')
    fed_collapse[fed_collapse['counterparty'] == 'Credit Suisse AG, New York Branch'].groupby('FirstMonthYear').sum()['fed_trade_amount'].plot(style='o', color='tab:orange', label='_nolegend_')
    fed_collapse[fed_collapse['counterparty'] == 'Citigroup Global Markets Inc.'].groupby('FirstMonthYear').sum()['fed_trade_amount'].plot(style='o', color='tab:green', label='_nolegend_')
    fed_collapse[fed_collapse['counterparty'] == 'Goldman Sachs & Co. LLC'].groupby('FirstMonthYear').sum()['fed_trade_amount'].plot(style='o', color='tab:red', label='_nolegend_')

    # live more space bottom
    # plt.subplots_adjust(bottom=0.3)
    # line in march 2020
    ax.grid(axis = "y", color = "grey", alpha = 0.2, linewidth = 0.5)
    ax.axvline(x='2020-03-01', color='black', linestyle='--')
    ax.spines[['right', 'top']].set_visible(False)
    ax.legend()
    ax.set_ylabel('Amount (millions $)')
    ax.set_xlabel('Month')
    ax.set_title('Amount of MBS purchased by Fed by month')
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))
    # onlly 5 ticks in y axis
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    # ax.xaxis.set_major_formatter(mda.DateFormatter('%Y-%m'))

    # x ticks set 
    ax.set_xticks([ '2020-01-01','2021-01-01'])
    ax.set_xticklabels(['Jan 2020', 'Jan 2021'], rotation=0)
    plt.show()

    # save
    fig.savefig(f'{save_fig}/fed_mbs_amount_by_month_example_larg4.pdf')

    # %%
    # * 


    # %%
    # **********************************************************************************************************************
    # * ob data
    # ob = pd.read_csv(f'{apa.auction_save_folder}/{apa.auction_filename}_mat{apa.maturity}_loan{apa.loantype}.csv', sep='|')

    # %%
    # datetime
    ob['FirstMonthYear'] = pd.to_datetime(ob['FirstMonthYear'])

                     
    # %%
    ob.columns
    # %%
    # issuername - HedgeInvestorKey in Ob
    g = ob.groupby('issuername').agg({'HedgeInvestorKey': 'first'}).sort_values(by='HedgeInvestorKey', ascending=False).reset_index()
    # %%
    g
    # %%
    # merge on HedgeInvestorKey OB and counterparty in fed, and merge on FirstMonthYear
    # merge on FirstMonthYear


    # %%
    # type of columns
    fed_collapse.info()

    # %%
    ob['FirstMonthYear'].info()

    # %%
    # **********************************************************************************************************************
    # * Loan level
    df_auc = pd.read_csv(f'{apa.auction_save_folder}/{apa.auction_filename}_mat{apa.maturity}_loan{apa.loantype}_auction_level.csv', sep='|')

    # %%
    df_auc.columns

    # %%
    # now create time series at seller, investor level
    var_time = 'MonthYear'
    var_rate = 's_coupon'
    df_time_series2 = apa.to_time_series(df_auc, 
                            bynote=True,
                            var_time= var_time, 
                            var_rate= var_rate,
                            add_name= f'{var_time}_{var_rate}_seller_investor',
                            groupby_other = ["HedgeClientKey","CommittedInvestorKey"])
    


    # %%
    # head
    # FirstMonthYear to datetime
    # delete column with repeated name of FirstMonthYear
    # df_time_series2 = df_time_series2.drop(columns=['FirstMonthYear'])

    # rename first column
    df_time_series2['FirstMonthYear'] = pd.to_datetime(df_time_series2['FirstMonthYear'])
    df_time_series2.head()

    # %%
    fed_collapse_c.rename(columns = {'counterparty_id':'CommittedInvestorKey'}, inplace = True)
    fed_collapse_c.head()

    # %%
    df_time_series2.rename(columns = {'s_coupon':'Coupon'}, inplace = True)

    # %%
    
    df = pd.merge(fed_collapse_c, df_time_series2, on= ['FirstMonthYear','CommittedInvestorKey', 'Coupon'], how='right')

    # %%
    df.head()

    # %%
    # create exposure measure

    # fed_trade_amount fil nan with 0
    df['fed_trade_amount'] = df['fed_trade_amount'].fillna(0)

    # if bank has fed_amount_transfer > 0 then exposure = 1
    df['exposure'] = np.where(df['fed_trade_amount'] > 0, 1, 0)

    #%%
    # creating variable with all the banks etc fed_trade_amount_march_2020 
    df['fed_trade_amount_march_2020'] = df.loc[df['FirstMonthYear'] == '2020-03-01', 'fed_trade_amount']
    # zero for other months
    df['fed_trade_amount_march_2020'] = df['fed_trade_amount_march_2020'].fillna(0)
    # if is equal any other month make equal to ['FirstMonthYear','CommittedInvestorKey', 'Coupon'] fed_trade_amount march 2020
    df['fed_trade_amount_march_2020'] = df.groupby(['CommittedInvestorKey', 'Coupon'])['fed_trade_amount_march_2020'].transform('max') 

    df["exposure_march_2020"] = np.where(df['fed_trade_amount_march_2020'] > 0, 1, 0)
    # %%
    df1 = df[df['CommittedInvestorKey'] == 9]
    df1[['fed_trade_amount_march_2020','fed_trade_amount','CommittedInvestorKey', 'Coupon','FirstMonthYear']].head(50)
    # %%
    # generate FirstMonthYear int period equivalent

    df.period.describe()
    # %%
    # onlt until april 2021
    df = df[df['FirstMonthYear'] < '2021-05-01']

    # %%
    #  save to csv
    df.to_csv(f'{apa.auction_save_folder}/ob_fed_exposure_measure.csv', index=False)
# %%
