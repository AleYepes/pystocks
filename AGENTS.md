The goal is to build an ETF factor-series analysis pipeline to calculate efficient frontier portfolios. At a high level, it must:

1. collect ETF-related data from IBKR and supplementary external sources
2. preserve both raw source captures and canonical historical facts
3. construct stable analysis inputs and point-in-time panels
4. build factor return series and run factor-series regressions
5. run efficient frontier portfolio construction