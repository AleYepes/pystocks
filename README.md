# pystocks
A series of scripts to scrape IBKR workstation data and calculate factor-tilted ETF portfolios. Use at your own risk.

# Requirements
1. Install Trader Workstation (IBKR) https://www.interactivebrokers.com/campus/ibkr-api-page/twsapi-doc/#find-the-api
- This will be the source for all fundamental data and historical price series

2. Use Python 3.12.8
```
pip install -r requirements.txt
```

## Supplementary install for notebooks + git
1. Install nbstripout to Remove Unnecessary Jupyter Metadata
```
pip install nbstripout
nbstripout --install
```

2. Use nbdime for Better Diffs and Merging
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
