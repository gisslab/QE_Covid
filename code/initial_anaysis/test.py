
#%%
import os
import pandas as pd
import numpy as np

#%%

auction_data_folder = '/project/houde/mortgages/data/raw/ob_auctions/auction_data_2018-09-01'
auction_save_folder = '/project/houde/mortgages/data/intermediate/ob_auctions'

#%%
# read f'{auction_save_folder}/combined_auctions_jan2018-jul2022_cleaned.csv'

### * Auction data testing * ###


df_auc = pd.read_csv(f'{auction_save_folder}/combined_auctions_jan2018-jul2022_cleaned.csv',
                     sep='|'
                     )


# %%
df_auc.iloc[50:90]

# %%

vars = ['Auction ID', 
        #'HedgeClientKey', 'HedgeLoanKey', 
        #'MonthlyIncome',
       #'CountyName', 
       'BorrowerClosingDate', 
       #'ZipCode', 
       'NoteRate',
       #'OriginalLoanAmount', 
       'LoanAmount',
        #'LTV', 'CLTV', 'DTIRatio', 'FICO',
       #'StateName', 'ProductLoanTerm', 'Occupancy', 'LoanPurpose',
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
      # 'StateCode', 'County FIPS', 'State FIPS',
       'DaysToAuction', 
       #'i_ARM', 'i_HighBalance', 'IntroductoryRatePeriod',
       #'LoanType', 'i_BulkBid', 
       'i_FannieBid',
       #  'i_FreddieBid', 'i_GinnieBid',
       #'i_Retail',
       'FannieBid', 
       # 'FreddieBid', 'GinnieBid', 'i_IncomeZero',
       #'i_IncomeNegative', 'i_HighIncome', 'i_SecondLien',
       #'SecondLienPercentage', 
       'Overall Rank', 
       #'Bid Rank', 'Reserve Rank',
       #'Number of Participants', 'Number of Bulk Bidders',
       #'Number of Reserve Price Bidders', 'Number of Enterprise Bidders',
    #    'Highest Bulk Bid', 'Second Highest Bulk Bid', 'Highest Reserve Bid',
    #    'Second Highest Reserve Bid'
       ]
df_auc[vars].iloc[100:140]


# %%

df = df_auc[vars]
df[df['i_FannieBid'] == 1].head(30)

fanniebids = df[df['i_FannieBid'] == 1]['Auction ID'].unique()
# %%
fanniebids_df = df[df['Auction ID'].isin(fanniebids)]
fanniebids_df.head(30)

# %%
df_auc.info()

# %%
df_auc.describe()

# %%
df_auc.columns

# %%
df_auc.LoanPurpose.value_counts()

# %%
df_auc.ProductType.value_counts()

# %%
# print Row, all columns 

for col in df_auc.columns[0:20]:
    print(col, " - ", df_auc.loc[0,col])

# %%
for col in df_auc.columns[21:41]:
    print(col, " - ", df_auc.loc[0,col])
# %%
for col in df_auc.columns[41:62]:
    print(col, " - ", df_auc.loc[0,col])

# %%


### * blomberg data testing * ###

bl_data_folder = '/project/houde/mortgages/QE_Covid/data/data_TBA/bloomberg/'

df_bl = pd.read_csv(f'{bl_data_folder}/clean_data/bloomberg_daily_trading_prices.csv',
                     sep='|'
                     )
# bloomberg_daily_trading_prices_w_forwards
# %%
df_bl.info()
# %%
df_bl.head(30)
# %%

df_bl.columns
# %%
df_bl.describe()
# %%
# conver to date time
df_bl['Trading_Date'] = pd.to_datetime(df_bl['Trading_Date'])
df_bl.Trading_Date.describe()
# %%
# trading date 2020
df_bl_2020 = df_bl[df_bl['Trading_Date'].dt.year == 2020]

# %%
df_bl_2020.Ticker.value_counts()
# %%
df_bl_2020.Forward_Trading_Months.value_counts()
# %%
