# pystocks
A quick and dirty library to scrape and process ETF data and to calculate factor-tilted ETF portfolios. Use at your own risk.

1. Contract scraper and validator
- api_contracts.ipynb

2. Fundamental data crawler for IBKR's Trader Workstation
- ibkr_ocr.ipynb

3. Historical price-series downloader and updater using IBKR's API
- api_series.ipynb

4. Factor return-series regression analyzes and MPT weights optimization
- data_prep.ipynb
- analysis.ipynb

# Requirements
Use Python 3.12.8
```
pip install -r requirements.txt
```

## Install nbstripout to Remove Unnecessary Jupyter Metadata
```
pip install nbstripout
nbstripout --install
```


## 2. Use nbdime for Better Diffs and Merging
```
pip install nbdime
nbdime config-git --enable
```

To manually compare two notebook versions:
```
nbdiff notebook_1.ipynb notebook_2.ipynb
```

To resolve conflicts interactively:

```
nbmerge notebook.ipynb
```
