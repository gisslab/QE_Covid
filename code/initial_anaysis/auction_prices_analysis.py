"""
Created on Thu June 1, 2023
@author: Giselle Labrador Badia (@gisslab)

This module is used to analyze the auction prices data from the OB (Optimal Blue) bid level data data. 

input: 
     file : csv file with the auction prices data from the OB (Optimal Blue) bid evel data data.
     path : '/project/houde/mortgages/QE_Covid/data/data_auction/clean_data'

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

crosswalk_folder = '/project/houde/mortgages/QE_Covid/data/crosswalks'
crosswalk_investors_filename = 'gnma_issuer_investor_crosswalk_full.dta'

table_folder = '/project/houde/mortgages/QE_Covid/results/tables'

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

dateinit = '2019-07-01'
""" Initial date of the auctioned loans."""

dateend = '2021-12-31'
""" End date of the auctioned loans."""

noterange_list = [(1,7),(2, 2.75), (2.75, 3.5), (3, 3.75), (3.75, 5.25),(4, 4.75), (5, 7), (3.5, 3.5), (4.5, 4.5)]
""" List of tuples with the ranges of the note rate to be used in the analysis. The elements are:
[(1,7),(2, 2.75), (2.75, 3.5), (3, 3.75), (3.75, 5.25),(4, 4.75), (5, 7),(3.5, 3.5), (4.5, 4.5) ] """

couponrange_list = [(1,5), (1,1.5), (2, 2.5), (2.5, 3), (3, 3.5), (3.5, 4), (4, 4.5)]
""" List of tuples with the ranges of the coupon to be used in the analysis. The elements are:
[(1,5),(1,1.5), (2, 2.5), (2.5, 3), (3, 3.5), (3.5, 4), (4, 4.5)] """

no_banks = [
            'PENNYMAC LOAN SERVICES, LLC',
            'CALIBER HOME LOANS, INC.',
            'HOME POINT FINANCIAL CORPORATI ON',
            'NATIONSTAR MORTGAGE, LLC',
            'LAKEVIEW LOAN SERVICING, LLC',
            'DITECH FINANCIAL LLC',
            'NEWREZ LLC',
            'PLANET HOME LENDING, LLC',
            'TOWNE MORTGAGE COMPANY',
            'THE MONEY SOURCE INC.',
            'AMERIHOME MORTGAGE COMPANY,LLC',
            'RUSHMORE LOAN MANAGEMENT SERVI CES, LLC',
            'PHH MORTGAGE CORPORATION',
            # 'SUNTRUST MORTGAGE, INC.', # DIVISION OF SUNTRUST BANK
            # 'GETEWAY MORTGAGE,A DIVISION OF GATEWAY FIRST BANK',
            'CMG MORTGAGE, INC.', # 
            'CITIMORTGAGE, INC.', #DIVISION OF CITIBANK
            'PLAZA HOME MORTGAGE, INC.',
            'WESTSTAR MORTGAGE CORPORATION',
            'FIRST GUARANTY MORTGAGE CORPORATION',
            'NATIONSTAR MORTGAGE LLC',
            'ARC HOME LLC',
            'WINTRUST MORTGAGE',
            'DBA FREEDOM HOME MORTGAGE CORP',
            'GUILE MORTGAGE COMPANY LLC',
            'ON Q FINANCIAL, INC.',
            'AMERICAN FINANCIAL RESOURCES, INC.',
            'UNITED SECURITY FINANCIAL CORP',
            'SUN WEST MORTGAGE CO., INC.',
            'VILLAGE CAPITAL & INVESTMENT, LLC',
            'IMPAC MORTGAGE',
            ]
""" List of mortagaes servicers that are not banks. """

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
""" Statistics to be calculated on the auction data."""

def winsorize_series(s, lower, upper):
    """ 
    Function to winsorize a series between two quantiles.
    """
#    clipped = s.clip(lower=s.quantile(lower), upper=s.quantile(upper), axis=1)
    quantiles = s.quantile([lower, upper])
    q_l = quantiles.loc[lower]
    q_u = quantiles.loc[upper]

    out = np.where(s.values <= q_l,q_l, 
                                      np.where(s >= q_u, q_u, s)
                  )

    return out


#%%
# * main functions


def read_data():
    """
    Reads the data from the auction data folder and returns a dataframe.
    """

    filepath = f'{auction_data_folder}/{auction_filename}.csv'
    try: 
        df_auc = pd.read_csv(filepath,
                            sep='|'
                            )
        return df_auc
    except Exception as e:
        print('Error reading the data: ',filepath, ", ", e)
        return None
    
def adding_investors_information(df):
    """
    Adds investors information to the dataframe and returns the merged dataframe.
    """

    # read crosswalks file
    df_crosswalk = pd.read_stata(f'{crosswalk_folder}/{crosswalk_investors_filename}')

    # merge to int
    df_crosswalk['o_CommittedInvestorKey'] = df_crosswalk['o_CommittedInvestorKey'].astype(int)
    df['HedgeInvestorKey'] = df['HedgeInvestorKey'].astype(int)


    df = df.merge(df_crosswalk, left_on = 'HedgeInvestorKey', right_on='o_CommittedInvestorKey',  how='left')

    print("Number of investors keys in Ob", df["HedgeInvestorKey"].nunique())
    print("Number of investors keys in crosswalk", df_crosswalk["o_CommittedInvestorKey"].nunique())
    df_not_in_crosswalk = df[~df['HedgeInvestorKey'].isin(df_crosswalk['o_CommittedInvestorKey'])]
    # print("Unmatched investor keys", df_not_in_crosswalk["CommittedInvestorKey"].unique())

    # * classify investors by investor's name (InvestorName) no banks list
    # print("Investors names: ", df['issuername'].unique())
    df['bank'] = np.where(~df['issuername'].isin(no_banks),1, 0)
    banks = df[df['bank'] == 1]['issuername'].unique()
    print("Banks names: ", banks)
    print("Number of banks: ", len(banks))
    print("Number of non-banks: ", df[df['bank'] == 0]['issuername'].nunique())

    # create number of banks by loan
    df["name_if_bank"] = np.where(df['bank'] == 1, df['issuername'], np.nan)
    df['number_banks'] = df.groupby(['Auction ID']).transform('nunique')['name_if_bank']

    df['fraction_banks'] = df['number_banks']/df['Number of Participants']

    # ? Note: If 99999 is -1, 99999 is nan, 17, 22, 23, 51 are GSEs. Then there are missing investors: 58 - 43 - 6 = 9

    return df



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

    # create date time column withfirst day of Committed date: e.g. 2020-01-22 -> 2020-01-01
    df['FirstMonthYear'] = df['CommittedDate'].dt.to_period('M').dt.to_timestamp()


    # convert loan ammount to thousands -> hence if it shows 10^6 it is billion : 10^9
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
    
    # remove auction with coupons that are not in the list from 1 to 5.5 adding 0.5 (removing odd coupons, e.g. 2.972)
    coupons = [x/2 for x in range(2, 11)]
    df = df.loc[df['s_coupon'].isin(coupons), :]


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
                                        'HedgeClientKey' : 'first', 
                                        'CommittedDate': 'first',
                                        'MonthYear': 'first',
                                        'FirstMonthYear': 'first',
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
                                        'number_banks' : 'first',
                                        'fraction_banks' : 'first'
                                        }).reset_index()
    

    
    # rename to eliminate multiindex and other small details
    df.columns = ['_'.join(col).strip() for col in df.columns.values]
    df.columns = [col.replace('_first', '') for col in df.columns.values]
    df.columns = [col.replace('_max', '') for col in df.columns.values]
    df = df.rename(columns={'Auction ID_': 'Auction ID'})

    # create auction classification
    df.loc[(df['sold_GSE'] == 0) & (df['dummy_sell_any']==1), 'auction_type'] = 'auction'
    df.loc[df['sold_GSE'] == 1.0, 'auction_type'] = 'cash_window'

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
                   var_rate = 'NoteRate',
                   groupby_other = []): # alternative 's_coupon'
    """
    Creates a time series dataframe from the auction level dataframe.
    """
    df = df.copy()
    
    group = [var_time, var_rate] + groupby_other if bynote else [var_time]  + groupby_other 

    # loan amount group and transform by day and noterate
    df['loan_amount_group'] = df.groupby(group)['LoanAmount'].transform('sum')

    # all price variables
    vars_iter = [var for var in df.columns if 'Price' in var]

    # TODO: Fix weighted aggregation of prices, still given smaller than mean numbers e.g. 20 when count is small. 

    # create weighted price variables by loan amount by day and noterate
    df['loan_weight'] = df['LoanAmount'] / df['loan_amount_group']

    for var in vars_iter:
        # print(var)
        df[f"{var}_weighted"] = df[var] * df['loan_weight']

    stats_max_price = stats_auction
        
    # Winsorizing continuous variables at 0.1% and 99.9% levels (to avoid outliers)
    #? Question: Winsorized by month or all data? - all data
    #? Question: winsorized loans size as well? - YES
    wind_list = ['Price', 
                'CommittedPrice', 
                'Price_weighted', 
                'CommittedPrice_weighted', 
                'LoanAmount'
                ]

    for f in wind_list:
        # print("Windsorized: " , f)
        df[f] = winsorize_series(df[f], 0.001, 0.999) 
    
    # additional price variables - to the price is it was sold to Fannie, Freddie, Ginnie
    df['price_fanny'] = np.where(df['sold_FannieBid'] == 1, df['Price'], np.nan)
    df['price_freddie'] = np.where(df['sold_FreddieBid'] == 1, df['Price'], np.nan)

    # * collapse

    dict_collapse = {
                        'Price_weighted': 'sum', # max,
                        'Price': stats_max_price,
                        'CommittedPrice': 'first',
                        'CommittedPrice_weighted': 'sum', # accepted price
                        'FirstMonthYear': 'first',
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
                        'price_fanny': 'mean',
                        'price_freddie': 'mean',
                        'number_banks' : 'mean',
                        'fraction_banks' : 'mean',
                        # 'loan_weight' : 'sum'
                    }
    
    if "s_coupon" in group:
        # add to dictionary: min, mean, max, median of NoteRate
        dict_collapse['NoteRate'] = ['min', 'mean', 'max', 'median']

    # * collapsing
    print("Collapsing data by: ", group)
    df = df.groupby(group).agg(dict_collapse).reset_index()


    # rename to eliminate multiindex and other small details
    df.columns = ['_'.join(col).strip() for col in df.columns.values]
    df.columns = [col.replace('_first', '') for col in df.columns.values]
    df.columns = [col.replace('Price_weighted', 'w_winner_bid') for col in df.columns.values]
    df.columns = [col.replace('Price', 'winner_bid') for col in df.columns.values]
    df = df.rename(columns={'Auction ID_count': 'number_auctions',
                            'w_winner_bid_sum' : 'w_winner_bid_mean',
                            'Committedw_winner_bid_sum' : 'committed_bid_mean',
                            })
    # remove '_' character only when last character
    df.columns = [col[:-1] if col[-1] == '_' else col for col in df.columns.values]

    # summary of the data
    # print("Summary of the data: ", df.describe())
    print("Number of observations: ", df.shape[0])
    # print("Column names: ", df.columns)

    # save data
    filename = f'{auction_save_folder}/timeseries/{auction_filename}_mat{maturity}_loan{loantype}_timeseries_{add_name}.csv'
    print("Saving data to: ", filename)
    df.to_csv(filename , sep='|', index=False)

    return df

def create_table_coupons(df_auc, table_folder):
    """
    Summary table of the auctions and note rates by coupon. Save latex table in 'table_folder' and returns summary dataframe.
    """

    df_coupon = df_auc.groupby('s_coupon').agg({'Auction ID' : 'count',
                                                'NoteRate': ['min', 'mean', 'median', 'max']
                                                }
                                                ).reset_index()
    
    df_coupon = df_coupon.rename(columns={'Auction ID': 'auctions', 
                                        's_coupon': 'coupon',
                                        'NoteRate': 'note rate'})
    
    df_coupon= df_coupon.round(3)

    df_coupon.to_latex(f'{table_folder}/auctions_coupon__mat{maturity}_loan{loantype}.tex',
                        index=False, 
                        float_format="%.3f")
    
    return df_coupon 


# %%

# * main (to run in jupyter notebook)

def main():
    """
    Main function to execute, it can be run in jupyter notebook interactive.
    """
    # %%
    # * create, save coupon 

    # df_coupon = read_get_security_characteristics() 

    # %%
    filename = 'ob_hmda_mbs_security_coupon.csv'
    df_coupon = pd.read_csv(f'{auction_data_folder}/{filename}', sep=',')
    # %%
    df_coupon.head()

    # %%
    df_coupon.info()

    # %%
    # * change AuctionID name for merge
    df_coupon = df_coupon.rename(columns={'AuctionId': 'Auction ID'})

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

    # * adding investors' information
    df2 = adding_investors_information(df1)

    # save data with investor names at bid level
    df2.to_csv(f'{auction_save_folder}/{auction_filename}_mat{maturity}_loan{loantype}.csv', sep='|', index=False)


    # %%
    df2[['bank', 'number_banks', 'name_if_bank', "Number of Participants"]].head(20)
    
    #%%
    df2[['bank', 'number_banks', 'name_if_bank', "Number of Participants"]].describe()


    # ***************** data at auction level *****************

    df_auc = create_measures_collapse(df2)

    # %%
    # # ***************** read data at auction level clean *****************

    df_auc = pd.read_csv(f'{auction_save_folder}/{auction_filename}_mat{maturity}_loan{loantype}_auction_level.csv', sep='|')

    # %%
    df_auc.columns

    # %%
    cols_describe = ['dummy_committedseller', 'dummy_sell_any', 'dummy_sell_winner', 'sold_GSE', 'sold_FannieBid', 'sold_FreddieBid', 'sold_GinnieBid', 's_coupon']
    # df_auc[['Auction ID','dummy_sell_any','dummy_sell_winner', 'sold_FannieBid', 'sold_FreddieBid', 'sold_GinnieBid' ]].describe()
    df_auc[cols_describe].describe()

    # %%
    # df_auc[['Auction ID','CommittedInvestorKey',  'CommittedPrice','Price', 'NoteRate', 's_coupon' ]].describe()

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
    # auc_ID = df_auc[(df_auc['dummy_sell_any']==0) & (df_auc['dummy_sell_winner']==1)]['Auction ID'].unique()
    # df1[df1['Auction ID'].isin(auc_ID)][[
    #     'Auction ID','CommittedInvestorKey', 'HedgeInvestorKey', 'HedgeClientKey', 
    #     'WinnerHedgeInvestorKey','Price','dummy_sell_winner']].head(35)
    
    #%%
    # * quantifying how many Nan bidders in df_auc 99999.0
    nanbid = df_auc[df_auc['CommittedInvestorKey']==99999.0]
    print("Number of nan bidders: ", nanbid.shape[0], "Number of auctions: ", df_auc['Auction ID'].nunique())
    # eliminate nan bidders
    df_auc = df_auc[df_auc['CommittedInvestorKey']!=99999.0]

    # %%
    # * how many missing s_coupon

    percent_missing_coupon = df_auc['s_coupon'].isna().sum()/ df_auc.shape[0]
    print("Percent missing s_coupon: ", percent_missing_coupon)

    # %%
    # * table with coupons informations
     
    table_coupons = create_table_coupons(df_auc, table_folder)
    table_coupons

    # %%
    # # * testing winsorize
    # print(df_auc['Price'].mean(), "number of elements: ", df_auc['Price'].shape[0])
    # winsorize_series(df_auc['Price'], 0.05, 0.95).mean()

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
                                    add_name= f'{var_rate}_{var_time}_auctype',
                                    groupby_other = ['auction_type']
        )

    df_time_series1 = to_time_series(df_auc, 
                                    bynote=True,
                                    var_time= var_time, 
                                    var_rate= var_rate,
                                    add_name= f'{var_rate}_{var_time}',
                                    groupby_other = []
        )

    df_time_series2 = to_time_series(df_auc, 
                                bynote=False,
                                var_time= var_time, 
                                var_rate= var_rate,
                                add_name= f'{var_time}',
                                groupby_other = []
    )

    # %%

    # * By rates or coupons 


    list = [(2.5,4)]#couponrange_list # noterange_list


    print("*********  intervals  *********")
    for (min_, max_) in list:

        print("********* ", min_, " - ", max_, "  *********")
        df_auc_1 = df_auc[ (df_auc[var_rate] >= min_) & (df_auc[var_rate] <= max_)].copy()

        df_time_series3 = to_time_series(df_auc_1, 
                                        bynote=False, 
                                        var_time='MonthYear',
                                        var_rate= var_rate,
                                        add_name= f'{var_rate}_{min_}_{max_}_{var_time}_auctype',
                                        groupby_other = ['auction_type']
                                        )

        df_time_series4 = to_time_series(df_auc_1, 
                                bynote=False, 
                                var_time='MonthYear',
                                var_rate= var_rate,
                                add_name= f'{var_rate}_{min_}_{max_}_{var_time}_agg',
                                groupby_other = []
                                )
    
    # %% 
    # count nan in auction_type

    print("Percentage nan ", df_auc['auction_type'].isna().sum()/df_auc.shape[0])
    print("Percentage cash window ", df_auc['auction_type'].value_counts()['cash_window']/df_auc.shape[0])
    print("Percentage auction ", df_auc['auction_type'].value_counts()['auction']/df_auc.shape[0])

    #! should be less than 27% (number of auctions with nan auction_type)
    # %% 

    # * end of main



if __name__ == '__main__':

    main()

    


# %%
