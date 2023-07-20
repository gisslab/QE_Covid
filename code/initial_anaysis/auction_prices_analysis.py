"""
Created on Thu June 1, 2023
@author: Giselle Labrador Badia (@gisslab)

This module is used to analyze the auction prices data from the OB (Optimal Blue) bid evel data data. 

input: 
     file : csv file with the auction prices data from the OB (Optimal Blue) bid evel data data.
     path : auction_save_folder

output:
    - csv file with auction/loan level data and measures of the prices (bids).
    - csv with time series of the auction prices measures.


"""

#%%
# * libraries

import pandas as pd
import numpy as np
from datetime import datetime

# * paths

# auction_data_folder = '/project/houde/mortgages/data/raw/ob_auctions/auction_data_2018-09-01'
auction_data_folder = '/project/houde/mortgages/QE_Covid/data/data_auction/clean_data'
# auction_save_folder = '/project/houde/mortgages/data/intermediate/ob_auctions'
auction_save_folder = '/project/houde/mortgages/QE_Covid/data/data_auction/clean_data'

auction_filename = 'combined_auctions_jan2018-jul2022_cleaned'

ob_hmda_orig_folder = '/project/houde/mortgages/QE_Covid/data/data_auction/clean_data/hmda-ob-mbs_origination_data_apr2023.dta'
""" Merged file with the ob, hmda, mbs data. """
# * settings

relevant_vars_ob_hmda_mbs = ['AuctionId', 
                             'LoanSequenceNumberEMBS', 
                             's_coupon']

relevant_vars =  [
       'Auction ID', 
       'HedgeClientKey',
        #'HedgeLoanKey', 
        #'MonthlyIncome',
       #'CountyName', 
       'BorrowerClosingDate', 
       #'ZipCode', 
       'NoteRate',
       #'OriginalLoanAmount', 
       'LoanAmount',
        #'LTV', 'CLTV', 'DTIRatio', 'FICO',
       #'StateName', 
       'ProductLoanTerm', 
       #Occupancy', 'LoanPurpose',
       #'PropertyType', 'WaiveEscrows', 'DocumentationType',
       #'AutomatedUnderwritingSystem', 'NumberOfUnits',
       'ProductType',
       #'SpecialProductType', 
       'CommittedDate', 
       'HedgeInvestorKey',
       #'DeliveryMethod',
       'Price', 
       #'1(Observation in New Data)',
       'CommittedInvestorKey', 
       #'StateCode', 'County FIPS', 'State FIPS',
       'DaysToAuction', 
       #'i_ARM', 'i_HighBalance', 'IntroductoryRatePeriod',
       'LoanType', 
       # 'i_BulkBid', 
       'i_FannieBid',
       'i_FreddieBid', 
       #'i_GinnieBid',
       #'i_Retail',
       'FannieBid', 
       'FreddieBid', 
       #'GinnieBid', 
       #'i_IncomeZero',
       #'i_IncomeNegative', 'i_HighIncome', 'i_SecondLien',
       #'SecondLienPercentage', 
       'Overall Rank', 
       #'Bid Rank', 'Reserve Rank',
       'Number of Participants', 
       'Number of Bulk Bidders',
       #'Number of Reserve Price Bidders', 
       'Number of Enterprise Bidders',
       'Highest Bulk Bid', 
       #'Second Highest Bulk Bid', 'Highest Reserve Bid',
        #'Second Highest Reserve Bid',
        's_coupon'
       ]
""" Relevant variables from the data."""

maturity = 30 
""" Maturity of the auctioned loans in years."""

loantype = 1
""" Type of the auctioned loans. 1 = Conforming"""

dateinit = '2020-01-01'
""" Initial date of the auctioned loans."""

dateend = '2021-12-31'
""" End date of the auctioned loans."""

noterange_list = [(1,7),(2, 2.75), (2.75, 3.5), (3, 3.75), (3.75, 5.25),(4, 4.75), (5, 7), (3.5, 3.5), (4.5, 4.5)]
""" List of tuples with the ranges of the note rate to be used in the analysis. The elements are:
[(1,7),(2, 2.75), (2.75, 3.5), (3, 3.75), (3.75, 5.25),(4, 4.75), (5, 7),(3.5, 3.5), (4.5, 4.5) ] """


# * stat functions

def coeff_var(x):
    return x.std()/x.mean()

def p90_p10(x):
    return x.quantile(.9)/ x.quantile(.1) 


stats_auction = [
                'min', 
                 'max',
                'mean', 
                'median',
                'std',
                'count',
                coeff_var,
                p90_p10
                ]
""" Statistics to be calculated for the auction data."""

def winsorize_series(s, lower, upper):
#    clipped = s.clip(lower=s.quantile(lower), upper=s.quantile(upper), axis=1)
    quantiles = s.quantile([lower, upper])
    q_l = quantiles.loc[lower]
    q_u = quantiles.loc[upper]

    out = np.where(s.values <= q_l,q_l, 
                                      np.where(s >= q_u, q_u, s)
                  )

    return out
""" Function to winsorize a series between two quantiles."""

#%%
# * main functions


def read_data():
    """
    Reads the data from the auction data folder and returns a dataframe.
    """

    filepath = f'{auction_save_folder}/{auction_filename}.csv'
    try: 
        df_auc = pd.read_csv(filepath,
                            sep='|'
                            )
        return df_auc
    except Exception as e:
        print('Error reading the data: ',filepath, ", ", e)
        return None

def read_get_security_characteristics():
    """
    Reads from the merge file the security characteristics and returns a dataframe.
    """
        
        # read by chunks to avoid memory error
    df_ob_hmda_orig = pd.read_stata(ob_hmda_orig_folder, chunksize=100000, convert_categoricals=False)

    cols = relevant_vars_ob_hmda_mbs
    # for each chunk keeps only where Aution_ID is not null
    df = pd.DataFrame()
    i = 0
    for chunk in df_ob_hmda_orig:
        dfchunk = chunk[chunk['AuctionId'].notnull() & chunk['LoanSequenceNumberEMBS'].notnull()][cols]
        df = pd.concat([df, dfchunk])
        i += 1
        if i % 100 == 0:
            print("chunk ", i)

    # save the data
    filename = 'ob_hmda_mbs_security_coupon.csv'
    print("Saving the data to csv: ", filename)
    df.to_csv(f'{auction_save_folder}/{filename}', index=False)

    return df

def clean_data(df):
    """
    Filters the data to relevant variables, dates and product type, save and returns a dataframe.
    """

    df = df.loc[:, relevant_vars]

    # create maturity variable = x if "x Yr" in Product Type
    df['Maturity'] = df['ProductType'].str.extract('(\d+) Yr', expand=False).copy() #.astype(int)
    print(df.value_counts('Maturity'))
    # define nan = -1 
    df['Maturity'] = df['Maturity'].fillna(-1).astype(int)

    # Use Maturity = 30 and LoanType = 1   
    # Loan type = 1 (Conf) , Loan Type = 2 (FHA) Loan Type = 3 (VA)  Loan Type = 4 (Rural) 
    df = df[(df['Maturity'] == maturity) & (df['LoanType'] == loantype)]

    # Alternatively choose only 30 year fixed rate loans: Conf 30 Yr Fixed
    # df = df[df['ProductType'] == 'Conf 30 Yr Fixed']

    # dates between interval
    df['CommittedDate'] = pd.to_datetime(df['CommittedDate'])
    df = df[(df['CommittedDate'] >= dateinit) & (df['CommittedDate'] <= dateend)]
    df['BorrowerClosingDate'] = pd.to_datetime(df['BorrowerClosingDate'])

    # create Month year column
    df['MonthYear'] = (
        df["CommittedDate"].dt.month_name() + "-" + df["CommittedDate"].dt.year.astype(str)
    )

    # convert loan ammount to thousands
    df['LoanAmount'] = df['LoanAmount']/1000

    # create winner investor id per auction_id  #! Winner is highest bid not the investor got the loan
    df['WinnerHedgeInvestorKey'] = df[df['Overall Rank'] == 1]['HedgeInvestorKey']
    df['WinnerHedgeInvestorKey'] = df.groupby(['Auction ID']).transform('max')['WinnerHedgeInvestorKey']

    # create price of commited investor
    df['CommittedPrice'] = df[df['CommittedInvestorKey'] == df['HedgeInvestorKey']]['Price']
    
    # * probability sell
    # when investor key = commited key, means that sells to that bidder, create variable dummy sell
    df['dummy_sell'] = np.where(df['CommittedInvestorKey'] == df['HedgeInvestorKey'], 1, 0)

    # when sells to anyone in the auction (transform create variable = 1 when dummy sell = 1 for any of the bids in Auction_ID)
    df['dummy_sell_any'] = df.groupby(['Auction ID'])['dummy_sell'].transform('max')

    # when well to highest bid, winner WinnerHedgeInvestorKey
    df['dummy_sell_winner'] = np.where(df['WinnerHedgeInvestorKey'] == df['CommittedInvestorKey'], 1, 0)

    df['dummy_committedseller'] = np.where(df['CommittedInvestorKey'] == df['HedgeClientKey'],1,0)

    # nan 99999 is one of Committed
    df['dummy_committednan'] = np.where(df['CommittedInvestorKey'] == 99999,1,0)

    # create sold to Fannie, Freddie, Ginnie
    df['sold_FannieBid'] = np.where(df['CommittedInvestorKey'] == 17,1,0)
    df['sold_FreddieBid'] = np.where(df['CommittedInvestorKey'].isin([22,23]), 1, 0)
    df['sold_GinnieBid'] = np.where(df['CommittedInvestorKey'] == 51, 1, 0)

    # sold any GSE
    df['sold_GSE'] = np.where(df['sold_FannieBid'] | df['sold_FreddieBid'] | df['sold_GinnieBid'], 1, 0)
    # assuming always CommitedInvestorKey bid and is one of the GSEs 

    # bulk bidders percentage
    df['bulk_bidders_fraction'] = df['Number of Bulk Bidders']/df['Number of Participants']


    # * end, summary and save 
    print("Summary of the data: ", df.describe())
    print("Number of observations: ", df.shape[0])
    print("Number of unique auctions: ", df['Auction ID'].nunique())

    # save data
    df.to_csv(f'{auction_save_folder}/{auction_filename}_mat{maturity}_loan{loantype}.csv', sep='|', index=False)
    return df


def create_measures_collapse(df):
    """
    Creates bid metrics like min, max, mean, median from the setting variable
    stats_auction, std; saves and returns a collapsed dataframe at the loan/auction level.
    """
    
    df = df.groupby(['Auction ID']).agg({'Price': stats_auction,
                                        'CommittedPrice': 'first',
                                        'CommittedInvestorKey': 'first',
                                        'WinnerHedgeInvestorKey': 'first',
                                        'HedgeClientKey' : 'first', #! check this
                                        'CommittedDate': 'first',
                                        'MonthYear': 'first',
                                        'BorrowerClosingDate': 'first',
                                        'DaysToAuction': 'first',
                                        'Number of Enterprise Bidders': 'first',
                                        'Number of Bulk Bidders': 'first',
                                        'Number of Participants' : 'first',
                                        'FannieBid': 'max',
                                        'FreddieBid': 'max',
                                        # 'GinnieBid': 'first',
                                        'i_FannieBid': 'max',
                                        'i_FreddieBid': 'max',
                                        # 'i_GinnieBid': 'first',
                                        'NoteRate': 'first',
                                        'LoanAmount': 'first',
                                        'ProductType': 'first',
                                        'dummy_sell_any': 'max',
                                        'dummy_sell_winner': 'max',
                                        'dummy_committedseller': 'max',
                                        'dummy_committednan': 'max',
                                        'sold_FannieBid': 'max',
                                        'sold_FreddieBid': 'max',
                                        'sold_GinnieBid': 'max',
                                        'sold_GSE': 'max',
                                        'bulk_bidders_fraction': 'max',
                                        's_coupon': 'first',
                                        }).reset_index()

    
    # rename to eliminate multiindex and other small details
    df.columns = ['_'.join(col).strip() for col in df.columns.values]
    df.columns = [col.replace('_first', '') for col in df.columns.values]
    df.columns = [col.replace('_max', '') for col in df.columns.values]
    df = df.rename(columns={'Auction ID_': 'Auction ID'})

    # print("Summary of the data: ", df.describe())
    print("Number of observations: ", df.shape[0])
    print("Column names: ", df.columns)

    # save data
    filename = f'{auction_save_folder}/{auction_filename}_mat{maturity}_loan{loantype}_auction_level.csv'
    df.to_csv(filename, sep='|', index=False)
    print("Data saved at: ", filename)
    return df


def to_time_series(df, bynote=False, 
                   add_name = '', 
                   var_time = 'CommittedDate', # alternatice 'MonthYear'
                   var_rate = 'NoteRate'): # alternative 's_coupon'
    """
    Creates a time series dataframe from the auction level dataframe.
    """
    df = df.copy()
    
    group = [var_time, var_rate] if bynote else [var_time]

    # loan amount group and transform by day and noterate
    df['loan_amount_time_noterate'] = df.groupby(group)['LoanAmount'].transform('sum')

    # all price variables
    vars_iter = [var for var in df.columns if 'Price' in var]

    # create weighted price variables by loan amount by day and noterate
    df['loan_weight'] = df['LoanAmount'] / df['loan_amount_time_noterate']

    for var in vars_iter:
        print(var)
        df[f"{var}_weighted"] = df[var] * df['loan_weight']

    stats_max_price = stats_auction
        
    # Winsorizing continuous variables at 0.1% and 99.9% levels (to avoid outliers)
    #? Question: Winsorized by month or all data? - all data
    #? Question: winsorized loans size as well? - YES
    wind_list = ['Price', 
                'CommittedPrice', 
                'Price_weighted', 
                'CommittedPrice_weighted', 
                'LoanAmount']

    for f in wind_list:
        print("Windsorized: " , f)
        df[f] = winsorize_series(df[f], 0.001, 0.999) 
    

    df = df.groupby(group).agg({
                                'Price_weighted': 'sum', # max,
                                'Price': stats_max_price,
                                'CommittedPrice': 'first',
                                'CommittedPrice_weighted': 'sum', # accepted price
                                'Auction ID': 'count',
                                'BorrowerClosingDate': 'first',
                                'DaysToAuction': 'mean',
                                'Number of Enterprise Bidders': 'mean',
                                'Number of Bulk Bidders': 'mean',
                                'Number of Participants' : 'mean',
                                'FannieBid': 'mean',
                                'FreddieBid': 'mean',
                                # 'GinnieBid': 'first',
                                'i_FannieBid': 'mean',
                                'i_FreddieBid': 'mean',
                                # 'i_GinnieBid': 'first',
                                'LoanAmount': ['mean', 'sum'],
                                'ProductType': 'first',
                                'dummy_sell_any': 'mean',
                                'dummy_sell_winner': 'mean',
                                'dummy_committedseller': 'mean',
                                'dummy_committednan': 'mean',
                                'sold_FannieBid': 'mean',
                                'sold_FreddieBid': 'mean',
                                'sold_GinnieBid': 'mean',
                                'sold_GSE': 'mean',
                                'bulk_bidders_fraction': 'mean',
                                # 'loan_weight' : 'sum'
                                }).reset_index()
    
    

    # rename to eliminate multiindex and other small details
    df.columns = ['_'.join(col).strip() for col in df.columns.values]
    df.columns = [col.replace('_first', '') for col in df.columns.values]
    df.columns = [col.replace('Price_weighted', 'w_winner_bid') for col in df.columns.values]
    df.columns = [col.replace('Price', 'winner_bid') for col in df.columns.values]
    df = df.rename(columns={'Auction ID_count': 'number_auctions',
                            'CommittedDate_': 'CommittedDate',
                            'w_winner_bid_sum' : 'w_winner_bid_mean'
                            })
    if bynote:
        df = df.rename(columns={'NoteRate_': 'NoteRate',
                                's_coupon_': 's_coupon'
                                })
    # summary of the data
    # print("Summary of the data: ", df.describe())
    print("Number of observations: ", df.shape[0])
    # print("Column names: ", df.columns)

    # save data
    filename = f'{auction_save_folder}/{auction_filename}_mat{maturity}_loan{loantype}_timeseries_{add_name}.csv'
    print("Saving data to: ", filename)
    df.to_csv(filename , sep='|', index=False)

    return df


# %%

# * main

if __name__ == '__main__':

    # %%
    # * saving coupon 

    df_coupon = read_get_security_characteristics() 

    # %%
    filename = 'ob_hmda_mbs_security_coupon.csv'
    df_coupon = pd.read_csv(f'{auction_save_folder}/{filename}', sep=',')
    # %%
    df_coupon.head()

    # %%
    df_coupon.info()

    # %%
    # * change AuctionID name for merge
    df_coupon = df_coupon.rename(columns={'AuctionId': 'Auction ID'})

    # %%
    # %%

    # * Reading and processing OB data
    # %%
    df = read_data()

    # %%
    df.columns

    # %%
    # * merge on Auction ID
    df = df.merge(df_coupon, on='Auction ID', how='left')

    df.columns


    # %%
    # %%
    df1 = clean_data(df)
    # %%
    df1[['Auction ID','CommittedInvestorKey', 'HedgeInvestorKey',  'WinnerHedgeInvestorKey','HedgeClientKey',
        'Price']].head(25)

    # %%
    df1[['Auction ID','CommittedInvestorKey', 'HedgeInvestorKey',  'WinnerHedgeInvestorKey',
         'dummy_sell', 'dummy_sell_any', 'dummy_sell_winner', 'Number of Participants']].head(25)

    # %%
    # print only where dummy_sell_any is 0
    df1[df1['dummy_sell_any']==0][
        ['Auction ID','CommittedInvestorKey', 'HedgeInvestorKey', 'HedgeClientKey', 'WinnerHedgeInvestorKey',
         'Price','sold_GSE']].head(45)
    #? Who is investor 65, accepts and it is not a GSE?
    # %%
    df1[['Auction ID','CommittedInvestorKey', 'HedgeInvestorKey', 'Number of Participants', 'Number of Bulk Bidders','bulk_bidders_fraction' ]].head(25)

    # %%   
    cols = ['Auction ID','CommittedInvestorKey', 'HedgeInvestorKey', 
            #'HedgeClientKey', 'WinnerHedgeInvestorKey',
            'Price','sold_GSE', 's_coupon', 'NoteRate']

    df1[cols].describe()        

    # %%   
    cols_describe = ['dummy_committedseller', 'dummy_sell_any', 'dummy_sell_winner', 'sold_GSE']
    df1[cols_describe].describe()
    # only means
    df1[cols_describe].mean()
    # %%
    # ***************** data at auction level *****************


    df_auc = create_measures_collapse(df1)

    # %%
    # ***************** read data at auction level *****************

    df_auc = pd.read_csv(f'{auction_save_folder}/{auction_filename}_mat{maturity}_loan{loantype}_auction_level.csv', sep='|')

    # %%
    df_auc.columns

    # %%
    cols_describe = ['dummy_committedseller', 'dummy_sell_any', 'dummy_sell_winner', 'sold_GSE', 'sold_FannieBid', 'sold_FreddieBid', 'sold_GinnieBid', 's_coupon']
    # df_auc[['Auction ID','dummy_sell_any','dummy_sell_winner', 'sold_FannieBid', 'sold_FreddieBid', 'sold_GinnieBid' ]].describe()
    df_auc[cols_describe].describe()

    # %%
    df_auc[['Auction ID','CommittedInvestorKey',  'CommittedPrice','Price', 'NoteRate', 's_coupon' ]].describe()

    # %%
    # * not sold to anyone stats
    # df_auc[['Auction ID','dummy_sell_any','dummy_sell_winner', 'sold_FannieBid', 'sold_FreddieBid', 'sold_GinnieBid' ]].describe()
    df_auc[df_auc['dummy_sell_any']==0][cols_describe].describe()

    # %%
    # * sold to anyone stats
    df_auc[df_auc['dummy_sell_any']==1][cols_describe].describe()

    # %%
    # * case in which sold to anyone = 0 (Committed is not in bids) and sold to Winner = 1 (highest bid is Commited) -> imposible
    #! what is 99999 key -> nan , CommittedInvestorKey
    auc_ID = df_auc[(df_auc['dummy_sell_any']==0) & (df_auc['dummy_sell_winner']==1)]['Auction ID'].unique()
    df1[df1['Auction ID'].isin(auc_ID)][[
        'Auction ID','CommittedInvestorKey', 'HedgeInvestorKey', 'HedgeClientKey', 
        'WinnerHedgeInvestorKey','Price','dummy_sell_winner']].head(35)
    

    # %%
    # * how many missing s_coupon

    percent_missing_coupon = df_auc['s_coupon'].isna().sum()/ df_auc.shape[0]
    print("Percent missing s_coupon: ", percent_missing_coupon)

    # %%
    # * testing winsorize
    print(df_auc['Price'].mean(), "number of elements: ", df_auc['Price'].shape[0])
    winsorize_series(df_auc['Price'], 0.05, 0.95).mean()

    # %%

    # *********** Time Series ***********

    # var_time = 'CommittedDate'
    var_time = 'MonthYear'
    # var_rate = 'NoteRate'
    var_rate = 's_coupon'


    # %%
    df_time_series = to_time_series(df_auc, 
                                    bynote=True,
                                    var_time= var_time, 
                                    var_rate= var_rate,
                                    add_name= f'{var_rate}_{var_time}'
        )

    # # %%

    # * By rates or coupons 


    noterange_list = [(1,7),(2, 2.75), (2.75, 3.5), (3, 3.75), (3.75, 5.25),(4, 4.75), (5, 7), (3.5, 3.5), (4.5, 4.5)]


    print("*********  Note rates intervals  *********")
    for (min_nr, max_nr) in noterange_list:

        print("********* ", min_nr, " - ", max_nr, "  *********")
        df_auc_1 = df_auc[ (df_auc[var_rate] >= min_nr) & (df_auc[var_rate] <= max_nr)].copy()
        df_time_series_1 = to_time_series(df_auc_1, 
                                        bynote=False, 
                                        var_time='MonthYear',
                                        add_name= f'{var_rate}_{min_nr}_{max_nr}_{var_time}'
                                        )

    # # %%
    # # * Separate cash windows (GSE bid accepted) and auction transactions (non GSE investors bid accepted)
    # # ? Questions is what to do when non GSE key bid appears as the buyer (Committed) but it is not one of the bids. 

    cash_windows = df_auc[df_auc['sold_GSE'] == 1].copy() 
    # # ? sold GSE are where Committed is GSE (does not have to be in the bids), is this right?


    # # %%
    # auction_trans = df_auc[(df_auc['sold_GSE'] == 0) & df_auc['dummy_sell_any']==1].copy()
    # # ? Maybe not need to remove transactions where the buyer is not one of the bidders. 
    # # ? option is to only remove where the buyer is not the seller (Commited and HedgeClientKey are different)

    # # %%
    # df_time_series_1[['w_winner_bid_mean', 'winner_bid_mean']].head(30)

    # * end of main


# %%
