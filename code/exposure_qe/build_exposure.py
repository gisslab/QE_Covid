"""
Created on Dec 5,2024
@author: Giselle Labrador Badia (@gisslab)

Build measure on exposure based on the Amount of QE purchases of MBS by the Federal Reserve. 

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
    fed.columns


    # %%
    # * collapse on on = ['FirstMonthYear', 'Coupon', 'forwardmonths', 'counterparty']
    fed_collapse = fmbs.collapse(fed, 
                        on = ['FirstMonthYear', 'Coupon', 'forwardmonths', 'counterparty']
                        )
    
    # %%
    fed_collapse.columns

    # %%
    # see by counterparty who has more purchases
    fed_collapse.groupby('counterparty').sum()['fed_trade_amount'].sort_values(ascending=False)

    # %% See Citie Group: Citigroup Global Markets Inc. by month
    fed_collapse[fed_collapse['counterparty'] == 'Citigroup Global Markets Inc.'].groupby('FirstMonthYear').sum()['fed_trade_amount'].plot()

    # %%
    # * merge with ob data


# %%
