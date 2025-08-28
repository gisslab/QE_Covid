# QE_Covid: Quantitative Easing Effects on Secondary Auction Markets

This project analyzes the effects of Federal Reserve Quantitative Easing (QE) policies on secondary mortgage auction markets during the COVID-19 period. The study examines how Fed MBS purchases impacted auction prices and bidding behavior in the mortgage-backed securities market.

## Disclaimer

**Note:** This is an exploratory research project. The analysis and findings presented here are preliminary and intended for research purposes only. Results should not be interpreted as definitive conclusions about the effects of quantitative easing policies.

## Code Structure

```text
code/
├── initial_anaysis/
│   ├── auction_prices_analysis.py      # Main auction price analysis from OB bid-level data
│   ├── auction_prices_timeseries_plots.py  # Time series visualization of auction prices
│   ├── fed_mbs.py                      # Federal Reserve MBS purchase data processing
│   ├── fed_ob_plots.py                 # Fed and OB data plotting utilities
│   └── test.py                         # Testing scripts
└── exposure_qe/
    ├── build_exposure.py               # QE exposure measure construction
    └── dyn_did_exposure_purchases.do   # Dynamic difference-in-differences estimation (Stata)
```

## Outputs

- **results/figures/**: Time series plots and analysis charts
- **results/tables/**: Statistical tables and regression outputs  
- **reports/**: LaTeX reports with analysis findings

## Data Sources

- Mortgage auction data and secondary market transactions
- Federal Reserve MBS purchase data
