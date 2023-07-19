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
import os


#%%

# * settings

# set directory
fed_data = "/project/houde/mortgages/QE_Covid/data/data_fed"

auction_save_folder = '/project/houde/mortgages/QE_Covid/results/'

filename = "MBS_data_new.csv"

cols = ['tradedate', 'contractualsettlementdate', 'transactioncategory',
       'operationtype', 'agency', 'couponinpercent', 'terminyears', 'cusip',
       'price', 'totalamounttransferredinmillions', 'counterparty',
       'tradeamount', 
    #    'auctionstatus', 'operationid', 'operationdate',
    #    'operationdirection', 'auctionmethod', 'classtype', 'settlementdate',
    #    'totalamtacceptedcurrentmillions', 'totalamtacceptedoriginalmillions',
    #    'totalamtsubmittedcurrentmillions', 'totalamtsubmittedoriginalmillion',
    #    'totalamtacceptedparmillions', 'totalamtsubmittedparmillions',
    #    'inclusionexclusion', 'securitydescription',
    #    'specpoolcurrentfaceamtaccepted', 'specpooloriginalfaceamtaccepted',
    #    'tbaparamountaccepted', 'note', 
    '_merge'
       ]
# os.chdir(fed_data)

# * functions

def read_data(file, path):
    """
    This function reads the data from the specified path.
    """
    print(f'Reading {file} from {path}')

    try:
        df = pd.read_csv(f'{path}/{file}')
    except Exception as e:
        print(f'Could not read {file} from {path}')
        print(e)
        df = None
    return df

# %%
# * main


if __name__ == '__main__':

    # %%
    # read data
    df_raw = read_data(file = filename, path = f'{fed_data}/raw_data/')

    # %%
    df  = df_raw.copy()

    # %%
    df_raw.agency.value_counts()

    # %%
    # dates to datetime format
    df['tradedate'] = pd.to_datetime(df['tradedate'])
    df['contractualsettlementdate'] = pd.to_datetime(df['contractualsettlementdate'])

    # dates between january 2020 april 2020

    df = df[(df['tradedate'] >= '2020-01-01') & (df['tradedate'] <= '2020-04-30')]

    #  terminyears = 30 

    df = df[df['terminyears'] == 30.0]

        # %%
    df = df[cols]

    # %%
    df.head(15)


    # %%
    df.columns 

    # %%
    df.describe()



    # %%
    agency = 'FNMA'
    df = df[df['agency'] == agency]

    # couponinpercent valuecounts
    df['couponinpercent'].value_counts()

    # %%

    df1 = df[(df['tradedate'] >= '2020-03-01') & (df['tradedate'] <= '2020-04-15')]

        # couponinpercent valuecounts
    df1['couponinpercent'].value_counts()

    # %%

    df1 = df[(df['tradedate'] < '2020-03-01')]

        # couponinpercent valuecounts
    df1['couponinpercent'].value_counts()


    # %%
    # * daily graph of purchases by coupon, totalamounttransferredinmillions, price
    # add ylabels

    g = df.groupby(['tradedate', 'couponinpercent'])[
                    'totalamounttransferredinmillions'].sum().unstack().plot()
    g.set_ylabel('Total amount transferred in millions')
    g.set_xlabel('Trade Date')
    g.figure.savefig(f'{auction_save_folder}/figures/{agency}_daily_purchases_tradedate_amount.png')
    # %%
    # price
    g = df.groupby(['tradedate', 'couponinpercent'])[
                    'price'].mean().unstack().plot()
    g.set_ylabel('Avg price')
    g.set_xlabel('Trade Date')
    g.figure.savefig(f'{auction_save_folder}/figures/{agency}_daily_purchases_tradedate_price.png')

    # %%
    # * daily graph of purchases by coupon, totalamounttransferredinmillions, price

    g = df.groupby(['contractualsettlementdate', 'couponinpercent'])[
                    'totalamounttransferredinmillions'].sum().unstack().plot()
    g.set_ylabel('Total amount transferred in millions')
    g.set_xlabel('Contractual Settlement Date')
    g.figure.savefig(f'{auction_save_folder}/figures/{agency}_daily_purchases_contractualsettlementdate_amount.png')

    # %%
    # price
    g = df.groupby(['contractualsettlementdate', 'couponinpercent'])[
                    'price'].mean().unstack().plot()
    g.set_ylabel('Avg price')
    g.set_xlabel('Contractual Settlement Date')
    g.figure.savefig(f'{auction_save_folder}/figures/{agency}_daily_purchases_contractualsettlementdate_price.png')



# %%
