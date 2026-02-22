example page URL for SPY (conId 756733):

https://www.interactivebrokers.ie/portal/?loginType=1&action=ACCT_MGMT_MAIN&clt=0&RL=1&locale=en_US#/quote/756733/fundamentals/landing?u=false&wb=0&exchange=ARCA&SESSIONID=6997f108.0000021f&impact_settings=true&supportsLeaf=true&widgets=objective,mstar,lipper_ratings,mf_key_ratios,risk_and_statistics,holdings,performance_and_peers,keyProfile,ownership,dividends,tear_sheet,news,fund_mstar,mf_esg,social_sentiment,securities_lending,sv,short_sale,ukuser

It seems that the API works primarily off of conIds. The following sections describe the different endpoints for tabs in the fundamentals page.

## Profile and Fees

Header:
https://www.interactivebrokers.ie/tws.proxy/fundamentals/mf_profile_and_fees/756733?sustainability=UK&lang=en

Response:
```
{
    "objective": "The Fund seeks to provide investment results that correspond generally to the price and yield performance of the S&P 500 Index (the Index). The Funds Portfolio consists of substantially all of the component common stocks that comprise the Index, which are weighted in accordance with the terms of the Trust Agreement.",
    "symbol": "SPY",
    "fund_and_profile": [
        {
            "name": "Fund Management Company",
            "name_tag": "Fund_Management_Company",
            "value": "State Street Global Advisors Trust Co"
        },
        {
            "name": "Distribution Details",
            "name_tag": "Distribution_Details",
            "value": "Paid",
            "value_tag": "paid_tag"
        },
        {
            "name": "Fiscal Date",
            "name_tag": "Fiscal_Date",
            "value": "9/30"
        },
        {
            "name": "Launch Opening Price",
            "name_tag": "Launch_Opening_Price",
            "value": "1993-01-22"
        },
        {
            "name": "Geographical Focus",
            "name_tag": "Geographical_Focus",
            "value": "United States of America"
        },
        {
            "name": "Asset Type",
            "name_tag": "Asset_Type",
            "value": "Equity"
        },
        {
            "name": "Fund Manager Benchmark",
            "name_tag": "Fund_Manager_Benchmark",
            "value": "S&P 500 TR"
        },
        {
            "name": "Management Approach",
            "name_tag": "Management_Approach",
            "value": "Passive"
        },
        {
            "name": "Total Expense Ratio",
            "name_tag": "Total_Expense_Ratio",
            "value": "0.09%"
        },
        {
            "name": "Inception Date",
            "name_tag": "Inception_Date",
            "value": "1993/01/22"
        },
        {
            "name": "Domicile",
            "name_tag": "Domicile",
            "value": "USA"
        },
        {
            "name": "Fund Category",
            "name_tag": "Fund_Category",
            "value": "Open-End Funds"
        },
        {
            "name": "Classification",
            "name_tag": "Classification",
            "value": "S&P 500 Index Funds"
        },
        {
            "name": "Fund Market Cap Focus",
            "name_tag": "Fund_Market_Cap_Focus",
            "value": "Large-cap"
        },
        {
            "name": "Scheme",
            "name_tag": "Scheme",
            "value": "US Mutual Fund Classification"
        },
        {
            "name": "Total Net Assets (Month End)",
            "name_tag": "Total_Net_Assets_Month_End",
            "value": "$708.92B (2026/01/30)"
        },
        {
            "name": "Objective Type",
            "name_tag": "Objective_Type",
            "value": "S&P 500 Index Objective Funds"
        },
        {
            "name": "Portfolio Manager",
            "name_tag": "Portfolio_Manager",
            "value": "Undisclosed"
        },
        {
            "name": "Management Expenses",
            "name_tag": "management_expenses",
            "value": "52.5%"
        }
    ],
    "reports": [
        {
            "name": "Annual Report",
            "as_of_date": 1759204800000,
            "fields": [
                {
                    "name": "Total Expense",
                    "value": "0.0893%",
                    "is_summary": true
                },
                {
                    "name": "Total Gross Expense",
                    "value": "0.0903%",
                    "is_summary": true
                },
                {
                    "name": "Total Net Expense",
                    "value": "0.09%",
                    "is_summary": true
                },
                {
                    "name": "Management Fees",
                    "value": "0.0469%",
                    "is_summary": true
                },
                {
                    "name": "Advisor Expenses",
                    "value": "0.0469%"
                },
                {
                    "name": "Non-Management Expenses",
                    "value": "0.0424%",
                    "is_summary": true
                },
                {
                    "name": "Audit and Legal Expenses",
                    "value": "0%"
                },
                {
                    "name": "Misc. Expenses",
                    "value": "0.0424%"
                },
                {
                    "name": "Registration Expenses",
                    "value": "0.0007%"
                },
                {
                    "name": "Other Expense",
                    "value": "0.0003%"
                }
            ]
        },
        {
            "name": "Prospectus Report",
            "as_of_date": 1737954000000,
            "fields": [
                {
                    "name": "Prospectus Gross Expense Ratio",
                    "value": "0.0945%"
                },
                {
                    "name": "Prospectus Gross Management Fee Ratio",
                    "value": "0.0945%"
                },
                {
                    "name": "Prospectus Gross Other Expense Ratio",
                    "value": "0.0482%"
                },
                {
                    "name": "Prospectus Net Expense Ratio",
                    "value": "0.0945%"
                },
                {
                    "name": "Prospectus Net Management Fee Ratio",
                    "value": "0.0463%"
                },
                {
                    "name": "Prospectus Net Other Expense Ratio",
                    "value": "0.0482%"
                }
            ]
        }
    ],
    "themes": [
        "Index Tracking"
    ],
    "expenses_allocation": [
        {
            "name": "Management Expenses",
            "value": "52.5%",
            "ratio": 0.5250381352562435
        },
        {
            "name": "Non-Management Expenses",
            "value": "47.5%",
            "ratio": 0.47496186474375646
        }
    ],
    "jap_fund_warning": false
}
```

These don't always look the same. Here's another example for another conId:

```
{
    "objective": "The investment objective of the Fund is to provide investment results that closely correspond, before fees and expenses, generally to the price and yield performance of the Solactive Global Uranium & Nuclear Components Total Return v2 Index (the Index). The Fund invests at least 80% of its total assets in securities of companies that are active in some aspect of the uranium industry such as mining, refining, exploration, manufacturing of equipment for the uranium industry, technologies related to the uranium industry or the production of nuclear components.",
    "symbol": "URNU",
    "fund_and_profile": [
        {
            "name": "Fund Management Company",
            "name_tag": "Fund_Management_Company",
            "value": "Global X Management Company (Europe) Ltd"
        },
        {
            "name": "Redemption Charge Max",
            "name_tag": "Redemption_Charge_Max",
            "value": "3%"
        },
        {
            "name": "Redemption Charge Actual",
            "name_tag": "Redemption_Charge_Actual",
            "value": "0%"
        },
        {
            "name": "Distribution Details",
            "name_tag": "Distribution_Details",
            "value": "Retained",
            "value_tag": "retained_tag"
        },
        {
            "name": "Fiscal Date",
            "name_tag": "Fiscal_Date",
            "value": "6/30"
        },
        {
            "name": "Launch Opening Price",
            "name_tag": "Launch_Opening_Price",
            "value": "2022-04-20"
        },
        {
            "name": "Geographical Focus",
            "name_tag": "Geographical_Focus",
            "value": "Global"
        },
        {
            "name": "Asset Type",
            "name_tag": "Asset_Type",
            "value": "Equity"
        },
        {
            "name": "Fund Manager Benchmark",
            "name_tag": "Fund_Manager_Benchmark",
            "value": "Solactive Global Uranium & Nuclear Components v2 TR USD"
        },
        {
            "name": "Management Approach",
            "name_tag": "Management_Approach",
            "value": "Passive"
        },
        {
            "name": "Domicile",
            "name_tag": "Domicile",
            "value": "Ireland"
        },
        {
            "name": "Fund Market Cap Focus",
            "name_tag": "Fund_Market_Cap_Focus",
            "value": "Broad Market"
        },
        {
            "name": "Total Net Assets (Month End)",
            "name_tag": "Total_Net_Assets_Month_End",
            "value": "$680.4M (2026/01/30)"
        },
        {
            "name": "Portfolio Manager",
            "name_tag": "Portfolio_Manager",
            "value": "Company Managed  "
        }
    ],
    "mstar": {
        "x_axis": [
            "Value",
            "Core",
            "Growth"
        ],
        "y_axis": [
            "Large",
            "Multi",
            "Mid",
            "Small"
        ],
        "x_axis_tag": [
            "value",
            "core",
            "growth"
        ],
        "y_axis_tag": [
            "large",
            "multi",
            "mid",
            "small"
        ],
        "selected": [],
        "hist": [
            [
                2,
                1
            ]
        ]
    },
    "reports": [
        {
            "name": "Annual Report",
            "as_of_date": 1751256000000,
            "fields": []
        },
        {
            "as_of_date": 0
        }
    ],
    "themes": [
        "Index Tracking"
    ],
    "expenses_allocation": [],
    "jap_fund_warning": false
}
```

The morningstar stylebox uses the first num in "hist" for x and the second for y, both starting at 0. In this case, it's growth-multi.


## Analyst Rating

### Lipper

Header: https://www.interactivebrokers.ie/tws.proxy/fundamentals/mf_lip_ratings/756733

Response:
```
{
    "universes": [
        {
            "as_of_date": 1769835600000,
            "overall": [
                {
                    "name": "Total Return",
                    "name_tag": "total_return",
                    "rating": {
                        "name": "114 funds",
                        "value": 5
                    }
                },
                {
                    "name": "Consistent Return",
                    "name_tag": "consistent_return",
                    "rating": {
                        "name": "114 funds",
                        "value": 5
                    }
                },
                {
                    "name": "Preservation",
                    "name_tag": "preservation",
                    "rating": {
                        "name": "11976 funds",
                        "value": 5
                    }
                },
                {
                    "name": "Tax Efficiency",
                    "name_tag": "tax_efficiency",
                    "rating": {
                        "name": "114 funds",
                        "value": 5
                    }
                },
                {
                    "name": "Expense",
                    "name_tag": "expense",
                    "rating": {
                        "name": "40 funds",
                        "value": 5
                    }
                }
            ],
            "3_year": [
                {
                    "name": "Total Return",
                    "name_tag": "total_return",
                    "rating": {
                        "name": "114 funds",
                        "value": 4
                    }
                },
                {
                    "name": "Consistent Return",
                    "name_tag": "consistent_return",
                    "rating": {
                        "name": "114 funds",
                        "value": 4
                    }
                },
                {
                    "name": "Preservation",
                    "name_tag": "preservation",
                    "rating": {
                        "name": "11976 funds",
                        "value": 5
                    }
                },
                {
                    "name": "Tax Efficiency",
                    "name_tag": "tax_efficiency",
                    "rating": {
                        "name": "114 funds",
                        "value": 5
                    }
                },
                {
                    "name": "Expense",
                    "name_tag": "expense",
                    "rating": {
                        "name": "40 funds",
                        "value": 5
                    }
                }
            ],
            "5_year": [
                {
                    "name": "Total Return",
                    "name_tag": "total_return",
                    "rating": {
                        "name": "113 funds",
                        "value": 5
                    }
                },
                {
                    "name": "Consistent Return",
                    "name_tag": "consistent_return",
                    "rating": {
                        "name": "113 funds",
                        "value": 5
                    }
                },
                {
                    "name": "Preservation",
                    "name_tag": "preservation",
                    "rating": {
                        "name": "11087 funds",
                        "value": 5
                    }
                },
                {
                    "name": "Tax Efficiency",
                    "name_tag": "tax_efficiency",
                    "rating": {
                        "name": "113 funds",
                        "value": 5
                    }
                },
                {
                    "name": "Expense",
                    "name_tag": "expense",
                    "rating": {
                        "name": "40 funds",
                        "value": 5
                    }
                }
            ],
            "10_year": [
                {
                    "name": "Total Return",
                    "name_tag": "total_return",
                    "rating": {
                        "name": "99 funds",
                        "value": 5
                    }
                },
                {
                    "name": "Consistent Return",
                    "name_tag": "consistent_return",
                    "rating": {
                        "name": "99 funds",
                        "value": 5
                    }
                },
                {
                    "name": "Preservation",
                    "name_tag": "preservation",
                    "rating": {
                        "name": "8551 funds",
                        "value": 5
                    }
                },
                {
                    "name": "Tax Efficiency",
                    "name_tag": "tax_efficiency",
                    "rating": {
                        "name": "99 funds",
                        "value": 5
                    }
                },
                {
                    "name": "Expense",
                    "name_tag": "expense",
                    "rating": {
                        "name": "39 funds",
                        "value": 5
                    }
                }
            ],
            "name": "United States"
        }
    ]
}
```

This can also vary to have various countries:
```
{
    "universes": [
        {
            "as_of_date": 1769749200000,
            "overall": [
                {
                    "name": "Total Return",
                    "name_tag": "total_return",
                    "rating": {
                        "name": "236 funds",
                        "value": 5
                    }
                },
                {
                    "name": "Consistent Return",
                    "name_tag": "consistent_return",
                    "rating": {
                        "name": "236 funds",
                        "value": 5
                    }
                },
                {
                    "name": "Preservation",
                    "name_tag": "preservation",
                    "rating": {
                        "name": "15816 funds",
                        "value": 1
                    }
                },
                {
                    "name": "Expense",
                    "name_tag": "expense",
                    "rating": {
                        "name": "235 funds",
                        "value": 5
                    }
                }
            ],
            "3_year": [
                {
                    "name": "Total Return",
                    "name_tag": "total_return",
                    "rating": {
                        "name": "236 funds",
                        "value": 5
                    }
                },
                {
                    "name": "Consistent Return",
                    "name_tag": "consistent_return",
                    "rating": {
                        "name": "236 funds",
                        "value": 5
                    }
                },
                {
                    "name": "Preservation",
                    "name_tag": "preservation",
                    "rating": {
                        "name": "15816 funds",
                        "value": 1
                    }
                },
                {
                    "name": "Expense",
                    "name_tag": "expense",
                    "rating": {
                        "name": "235 funds",
                        "value": 5
                    }
                }
            ],
            "name": "Sweden"
        },
        {
            "as_of_date": 1769749200000,
            "overall": [
                {
                    "name": "Total Return",
                    "name_tag": "total_return",
                    "rating": {
                        "name": "184 funds",
                        "value": 5
                    }
                },
                {
                    "name": "Consistent Return",
                    "name_tag": "consistent_return",
                    "rating": {
                        "name": "184 funds",
                        "value": 5
                    }
                },
                {
                    "name": "Preservation",
                    "name_tag": "preservation",
                    "rating": {
                        "name": "10798 funds",
                        "value": 1
                    }
                },
                {
                    "name": "Expense",
                    "name_tag": "expense",
                    "rating": {
                        "name": "179 funds",
                        "value": 5
                    }
                }
            ],
            "3_year": [
                {
                    "name": "Total Return",
                    "name_tag": "total_return",
                    "rating": {
                        "name": "184 funds",
                        "value": 5
                    }
                },
                {
                    "name": "Consistent Return",
                    "name_tag": "consistent_return",
                    "rating": {
                        "name": "184 funds",
                        "value": 5
                    }
                },
                {
                    "name": "Preservation",
                    "name_tag": "preservation",
                    "rating": {
                        "name": "10798 funds",
                        "value": 1
                    }
                },
                {
                    "name": "Expense",
                    "name_tag": "expense",
                    "rating": {
                        "name": "179 funds",
                        "value": 5
                    }
                }
            ],
            "name": "Denmark"
        },
        {
            "as_of_date": 1769749200000,
            "overall": [
                {
                    "name": "Total Return",
                    "name_tag": "total_return",
                    "rating": {
                        "name": "383 funds",
                        "value": 5
                    }
                },
                {
                    "name": "Consistent Return",
                    "name_tag": "consistent_return",
                    "rating": {
                        "name": "383 funds",
                        "value": 5
                    }
                },
                {
                    "name": "Preservation",
                    "name_tag": "preservation",
                    "rating": {
                        "name": "27639 funds",
                        "value": 1
                    }
                },
                {
                    "name": "Expense",
                    "name_tag": "expense",
                    "rating": {
                        "name": "374 funds",
                        "value": 5
                    }
                }
            ],
            "3_year": [
                {
                    "name": "Total Return",
                    "name_tag": "total_return",
                    "rating": {
                        "name": "383 funds",
                        "value": 5
                    }
                },
                {
                    "name": "Consistent Return",
                    "name_tag": "consistent_return",
                    "rating": {
                        "name": "383 funds",
                        "value": 5
                    }
                },
                {
                    "name": "Preservation",
                    "name_tag": "preservation",
                    "rating": {
                        "name": "27639 funds",
                        "value": 1
                    }
                },
                {
                    "name": "Expense",
                    "name_tag": "expense",
                    "rating": {
                        "name": "374 funds",
                        "value": 5
                    }
                }
            ],
            "name": "Luxembourg"
        },
        {
            "as_of_date": 1769749200000,
            "overall": [
                {
                    "name": "Total Return",
                    "name_tag": "total_return",
                    "rating": {
                        "name": "285 funds",
                        "value": 5
                    }
                },
                {
                    "name": "Consistent Return",
                    "name_tag": "consistent_return",
                    "rating": {
                        "name": "285 funds",
                        "value": 5
                    }
                },
                {
                    "name": "Preservation",
                    "name_tag": "preservation",
                    "rating": {
                        "name": "17982 funds",
                        "value": 1
                    }
                },
                {
                    "name": "Expense",
                    "name_tag": "expense",
                    "rating": {
                        "name": "282 funds",
                        "value": 5
                    }
                }
            ],
            "3_year": [
                {
                    "name": "Total Return",
                    "name_tag": "total_return",
                    "rating": {
                        "name": "285 funds",
                        "value": 5
                    }
                },
                {
                    "name": "Consistent Return",
                    "name_tag": "consistent_return",
                    "rating": {
                        "name": "285 funds",
                        "value": 5
                    }
                },
                {
                    "name": "Preservation",
                    "name_tag": "preservation",
                    "rating": {
                        "name": "17982 funds",
                        "value": 1
                    }
                },
                {
                    "name": "Expense",
                    "name_tag": "expense",
                    "rating": {
                        "name": "282 funds",
                        "value": 5
                    }
                }
            ],
            "name": "Austria"
        },
        {
            "as_of_date": 1769749200000,
            "overall": [
                {
                    "name": "Total Return",
                    "name_tag": "total_return",
                    "rating": {
                        "name": "234 funds",
                        "value": 5
                    }
                },
                {
                    "name": "Consistent Return",
                    "name_tag": "consistent_return",
                    "rating": {
                        "name": "234 funds",
                        "value": 5
                    }
                },
                {
                    "name": "Preservation",
                    "name_tag": "preservation",
                    "rating": {
                        "name": "15732 funds",
                        "value": 1
                    }
                },
                {
                    "name": "Expense",
                    "name_tag": "expense",
                    "rating": {
                        "name": "233 funds",
                        "value": 5
                    }
                }
            ],
            "3_year": [
                {
                    "name": "Total Return",
                    "name_tag": "total_return",
                    "rating": {
                        "name": "234 funds",
                        "value": 5
                    }
                },
                {
                    "name": "Consistent Return",
                    "name_tag": "consistent_return",
                    "rating": {
                        "name": "234 funds",
                        "value": 5
                    }
                },
                {
                    "name": "Preservation",
                    "name_tag": "preservation",
                    "rating": {
                        "name": "15732 funds",
                        "value": 1
                    }
                },
                {
                    "name": "Expense",
                    "name_tag": "expense",
                    "rating": {
                        "name": "233 funds",
                        "value": 5
                    }
                }
            ],
            "name": "Netherlands"
        },
        {
            "as_of_date": 1769749200000,
            "overall": [
                {
                    "name": "Total Return",
                    "name_tag": "total_return",
                    "rating": {
                        "name": "217 funds",
                        "value": 5
                    }
                },
                {
                    "name": "Consistent Return",
                    "name_tag": "consistent_return",
                    "rating": {
                        "name": "217 funds",
                        "value": 5
                    }
                },
                {
                    "name": "Preservation",
                    "name_tag": "preservation",
                    "rating": {
                        "name": "14551 funds",
                        "value": 1
                    }
                },
                {
                    "name": "Expense",
                    "name_tag": "expense",
                    "rating": {
                        "name": "216 funds",
                        "value": 5
                    }
                }
            ],
            "3_year": [
                {
                    "name": "Total Return",
                    "name_tag": "total_return",
                    "rating": {
                        "name": "217 funds",
                        "value": 5
                    }
                },
                {
                    "name": "Consistent Return",
                    "name_tag": "consistent_return",
                    "rating": {
                        "name": "217 funds",
                        "value": 5
                    }
                },
                {
                    "name": "Preservation",
                    "name_tag": "preservation",
                    "rating": {
                        "name": "14551 funds",
                        "value": 1
                    }
                },
                {
                    "name": "Expense",
                    "name_tag": "expense",
                    "rating": {
                        "name": "216 funds",
                        "value": 5
                    }
                }
            ],
            "name": "Finland"
        },
        {
            "as_of_date": 1769749200000,
            "overall": [
                {
                    "name": "Total Return",
                    "name_tag": "total_return",
                    "rating": {
                        "name": "185 funds",
                        "value": 5
                    }
                },
                {
                    "name": "Consistent Return",
                    "name_tag": "consistent_return",
                    "rating": {
                        "name": "185 funds",
                        "value": 5
                    }
                },
                {
                    "name": "Preservation",
                    "name_tag": "preservation",
                    "rating": {
                        "name": "12240 funds",
                        "value": 1
                    }
                },
                {
                    "name": "Expense",
                    "name_tag": "expense",
                    "rating": {
                        "name": "184 funds",
                        "value": 5
                    }
                }
            ],
            "3_year": [
                {
                    "name": "Total Return",
                    "name_tag": "total_return",
                    "rating": {
                        "name": "185 funds",
                        "value": 5
                    }
                },
                {
                    "name": "Consistent Return",
                    "name_tag": "consistent_return",
                    "rating": {
                        "name": "185 funds",
                        "value": 5
                    }
                },
                {
                    "name": "Preservation",
                    "name_tag": "preservation",
                    "rating": {
                        "name": "12240 funds",
                        "value": 1
                    }
                },
                {
                    "name": "Expense",
                    "name_tag": "expense",
                    "rating": {
                        "name": "184 funds",
                        "value": 5
                    }
                }
            ],
            "name": "Norway"
        },
        {
            "as_of_date": 1769749200000,
            "overall": [
                {
                    "name": "Total Return",
                    "name_tag": "total_return",
                    "rating": {
                        "name": "285 funds",
                        "value": 5
                    }
                },
                {
                    "name": "Consistent Return",
                    "name_tag": "consistent_return",
                    "rating": {
                        "name": "285 funds",
                        "value": 5
                    }
                },
                {
                    "name": "Preservation",
                    "name_tag": "preservation",
                    "rating": {
                        "name": "23185 funds",
                        "value": 1
                    }
                },
                {
                    "name": "Expense",
                    "name_tag": "expense",
                    "rating": {
                        "name": "283 funds",
                        "value": 5
                    }
                }
            ],
            "3_year": [
                {
                    "name": "Total Return",
                    "name_tag": "total_return",
                    "rating": {
                        "name": "285 funds",
                        "value": 5
                    }
                },
                {
                    "name": "Consistent Return",
                    "name_tag": "consistent_return",
                    "rating": {
                        "name": "285 funds",
                        "value": 5
                    }
                },
                {
                    "name": "Preservation",
                    "name_tag": "preservation",
                    "rating": {
                        "name": "23185 funds",
                        "value": 1
                    }
                },
                {
                    "name": "Expense",
                    "name_tag": "expense",
                    "rating": {
                        "name": "283 funds",
                        "value": 5
                    }
                }
            ],
            "name": "UK"
        },
        {
            "as_of_date": 1769749200000,
            "overall": [
                {
                    "name": "Total Return",
                    "name_tag": "total_return",
                    "rating": {
                        "name": "401 funds",
                        "value": 5
                    }
                },
                {
                    "name": "Consistent Return",
                    "name_tag": "consistent_return",
                    "rating": {
                        "name": "401 funds",
                        "value": 5
                    }
                },
                {
                    "name": "Preservation",
                    "name_tag": "preservation",
                    "rating": {
                        "name": "27653 funds",
                        "value": 1
                    }
                },
                {
                    "name": "Expense",
                    "name_tag": "expense",
                    "rating": {
                        "name": "390 funds",
                        "value": 5
                    }
                }
            ],
            "3_year": [
                {
                    "name": "Total Return",
                    "name_tag": "total_return",
                    "rating": {
                        "name": "401 funds",
                        "value": 5
                    }
                },
                {
                    "name": "Consistent Return",
                    "name_tag": "consistent_return",
                    "rating": {
                        "name": "401 funds",
                        "value": 5
                    }
                },
                {
                    "name": "Preservation",
                    "name_tag": "preservation",
                    "rating": {
                        "name": "27653 funds",
                        "value": 1
                    }
                },
                {
                    "name": "Expense",
                    "name_tag": "expense",
                    "rating": {
                        "name": "390 funds",
                        "value": 5
                    }
                }
            ],
            "name": "Switzerland"
        },
        {
            "as_of_date": 1769749200000,
            "overall": [
                {
                    "name": "Total Return",
                    "name_tag": "total_return",
                    "rating": {
                        "name": "424 funds",
                        "value": 5
                    }
                },
                {
                    "name": "Consistent Return",
                    "name_tag": "consistent_return",
                    "rating": {
                        "name": "424 funds",
                        "value": 5
                    }
                },
                {
                    "name": "Preservation",
                    "name_tag": "preservation",
                    "rating": {
                        "name": "28413 funds",
                        "value": 1
                    }
                },
                {
                    "name": "Expense",
                    "name_tag": "expense",
                    "rating": {
                        "name": "413 funds",
                        "value": 5
                    }
                }
            ],
            "3_year": [
                {
                    "name": "Total Return",
                    "name_tag": "total_return",
                    "rating": {
                        "name": "424 funds",
                        "value": 5
                    }
                },
                {
                    "name": "Consistent Return",
                    "name_tag": "consistent_return",
                    "rating": {
                        "name": "424 funds",
                        "value": 5
                    }
                },
                {
                    "name": "Preservation",
                    "name_tag": "preservation",
                    "rating": {
                        "name": "28413 funds",
                        "value": 1
                    }
                },
                {
                    "name": "Expense",
                    "name_tag": "expense",
                    "rating": {
                        "name": "413 funds",
                        "value": 5
                    }
                }
            ],
            "name": "Germany"
        },
        {
            "as_of_date": 1769749200000,
            "overall": [
                {
                    "name": "Total Return",
                    "name_tag": "total_return",
                    "rating": {
                        "name": "262 funds",
                        "value": 5
                    }
                },
                {
                    "name": "Consistent Return",
                    "name_tag": "consistent_return",
                    "rating": {
                        "name": "262 funds",
                        "value": 5
                    }
                },
                {
                    "name": "Preservation",
                    "name_tag": "preservation",
                    "rating": {
                        "name": "20280 funds",
                        "value": 1
                    }
                },
                {
                    "name": "Expense",
                    "name_tag": "expense",
                    "rating": {
                        "name": "250 funds",
                        "value": 5
                    }
                }
            ],
            "3_year": [
                {
                    "name": "Total Return",
                    "name_tag": "total_return",
                    "rating": {
                        "name": "262 funds",
                        "value": 5
                    }
                },
                {
                    "name": "Consistent Return",
                    "name_tag": "consistent_return",
                    "rating": {
                        "name": "262 funds",
                        "value": 5
                    }
                },
                {
                    "name": "Preservation",
                    "name_tag": "preservation",
                    "rating": {
                        "name": "20280 funds",
                        "value": 1
                    }
                },
                {
                    "name": "Expense",
                    "name_tag": "expense",
                    "rating": {
                        "name": "250 funds",
                        "value": 5
                    }
                }
            ],
            "name": "France"
        },
        {
            "as_of_date": 1769749200000,
            "overall": [
                {
                    "name": "Total Return",
                    "name_tag": "total_return",
                    "rating": {
                        "name": "206 funds",
                        "value": 5
                    }
                },
                {
                    "name": "Consistent Return",
                    "name_tag": "consistent_return",
                    "rating": {
                        "name": "206 funds",
                        "value": 5
                    }
                },
                {
                    "name": "Preservation",
                    "name_tag": "preservation",
                    "rating": {
                        "name": "16347 funds",
                        "value": 1
                    }
                },
                {
                    "name": "Expense",
                    "name_tag": "expense",
                    "rating": {
                        "name": "206 funds",
                        "value": 5
                    }
                }
            ],
            "3_year": [
                {
                    "name": "Total Return",
                    "name_tag": "total_return",
                    "rating": {
                        "name": "206 funds",
                        "value": 5
                    }
                },
                {
                    "name": "Consistent Return",
                    "name_tag": "consistent_return",
                    "rating": {
                        "name": "206 funds",
                        "value": 5
                    }
                },
                {
                    "name": "Preservation",
                    "name_tag": "preservation",
                    "rating": {
                        "name": "16347 funds",
                        "value": 1
                    }
                },
                {
                    "name": "Expense",
                    "name_tag": "expense",
                    "rating": {
                        "name": "206 funds",
                        "value": 5
                    }
                }
            ],
            "name": "Italy"
        },
        {
            "as_of_date": 1769749200000,
            "overall": [
                {
                    "name": "Total Return",
                    "name_tag": "total_return",
                    "rating": {
                        "name": "254 funds",
                        "value": 5
                    }
                },
                {
                    "name": "Consistent Return",
                    "name_tag": "consistent_return",
                    "rating": {
                        "name": "254 funds",
                        "value": 5
                    }
                },
                {
                    "name": "Preservation",
                    "name_tag": "preservation",
                    "rating": {
                        "name": "16664 funds",
                        "value": 1
                    }
                },
                {
                    "name": "Expense",
                    "name_tag": "expense",
                    "rating": {
                        "name": "253 funds",
                        "value": 5
                    }
                }
            ],
            "3_year": [
                {
                    "name": "Total Return",
                    "name_tag": "total_return",
                    "rating": {
                        "name": "254 funds",
                        "value": 5
                    }
                },
                {
                    "name": "Consistent Return",
                    "name_tag": "consistent_return",
                    "rating": {
                        "name": "254 funds",
                        "value": 5
                    }
                },
                {
                    "name": "Preservation",
                    "name_tag": "preservation",
                    "rating": {
                        "name": "16664 funds",
                        "value": 1
                    }
                },
                {
                    "name": "Expense",
                    "name_tag": "expense",
                    "rating": {
                        "name": "253 funds",
                        "value": 5
                    }
                }
            ],
            "name": "Singapore"
        },
        {
            "as_of_date": 1769749200000,
            "overall": [
                {
                    "name": "Total Return",
                    "name_tag": "total_return",
                    "rating": {
                        "name": "274 funds",
                        "value": 5
                    }
                },
                {
                    "name": "Consistent Return",
                    "name_tag": "consistent_return",
                    "rating": {
                        "name": "274 funds",
                        "value": 5
                    }
                },
                {
                    "name": "Preservation",
                    "name_tag": "preservation",
                    "rating": {
                        "name": "18258 funds",
                        "value": 1
                    }
                },
                {
                    "name": "Expense",
                    "name_tag": "expense",
                    "rating": {
                        "name": "270 funds",
                        "value": 5
                    }
                }
            ],
            "3_year": [
                {
                    "name": "Total Return",
                    "name_tag": "total_return",
                    "rating": {
                        "name": "274 funds",
                        "value": 5
                    }
                },
                {
                    "name": "Consistent Return",
                    "name_tag": "consistent_return",
                    "rating": {
                        "name": "274 funds",
                        "value": 5
                    }
                },
                {
                    "name": "Preservation",
                    "name_tag": "preservation",
                    "rating": {
                        "name": "18258 funds",
                        "value": 1
                    }
                },
                {
                    "name": "Expense",
                    "name_tag": "expense",
                    "rating": {
                        "name": "270 funds",
                        "value": 5
                    }
                }
            ],
            "name": "Spain"
        }
    ]
}
```

### Morningstar
Not all instruments have morningstar values.

Header: 
https://www.interactivebrokers.ie/tws.proxy/mstar/fund/detail?conid=756733

Response:
{
    "as_of_date": "20260131",
    "summary": [
        {
            "id": "medalist_rating",
            "title": "Medalist Rating",
            "value": "Silver",
            "q": false,
            "publish_date": "20260128"
        },
        {
            "id": "process",
            "title": "Process",
            "value": "High",
            "q": false,
            "publish_date": "20260128"
        },
        {
            "id": "people",
            "title": "People",
            "value": "Above_Average",
            "q": false,
            "publish_date": "20260128"
        },
        {
            "id": "parent",
            "title": "Parent",
            "value": "Above_Average",
            "q": false,
            "publish_date": "20250716"
        },
        {
            "id": "morningstar_rating",
            "title": "Morningstar Rating",
            "value": "4",
            "q": false,
            "publish_date": "20260131"
        },
        {
            "id": "sustainability_rating",
            "title": "Sustainability Rating",
            "value": "Average",
            "q": false,
            "publish_date": "20251231"
        },
        {
            "id": "category",
            "title": "Category",
            "value": "Large Blend",
            "q": false
        },
        {
            "id": "category_index",
            "title": "Category Index",
            "value": "Morningstar US Large-Mid TR USD",
            "q": false
        }
    ],
    "commentary": [
        {
            "id": "summary",
            "title": "Summary",
            "subtitle": "Title",
            "subsection_id": "summary_title",
            "q": false,
            "publish_date": "20260128",
            "text": "Best-in-class option for large-cap US stocks.",
            "author": {
                "name": "Brendan McCann"
            }
        },
        {
            "id": "summary",
            "title": "Summary",
            "subtitle": "Body",
            "subsection_id": "summary_body",
            "q": false,
            "publish_date": "20260128",
            "text": "State Street S&P 500 accurately represents the large-cap US stock market, allowing its low fee and efficient portfolio to carve out a long-term edge.   The fund tracks the S&P 500. A committee selects 500 of the largest US stocks, or roughly 80% of the US stock market, and weights them by market cap. The index committee has discretion over selecting companies that meet its liquidity and profitability standards. While a committee-based approach may lack clarity, it adds flexibility to reduce unnecessary changes during reconstitution, taming transaction costs compared with more rigid rules-based indexes.  Assigning position sizes based on a stock’s market cap is a simple and efficient method to weight the portfolio. Since US stocks are highly traded, they quickly reflect new information, and carving an edge is difficult. Market-cap weighting naturally adjusts to price changes without frequent rebalancing, generating lower trading costs. That, and lower fees, give large-blend index funds a long-term performance advantage over most actively managed peers.   The fund holds a broad, well-diversified portfolio. It typically includes around 500 stocks, and the top 10 represented around 40% of the portfolio at year-end 2025. Still, market-cap weighting can contribute to portfolio concentration when a few stocks dominate the market. This has been the case lately with a handful of mega-cap technology stocks growing to prominence and commanding a greater share of the portfolio.   When a few richly valued companies or sectors power most of the market gains, market-cap weighting may overexpose the strategy to the fluctuations of one stock or sector. But this is not a fault in design, as it simply reflects the market’s composition. Its low turnover, low fee, and broad diversification across the US market more than offset these risks.  The S&P 500 returned 14.8% annualized over the past 10 years through year-end 2025. It holds little cash, which should help it outperform cash-saddled active peers during market rallies. Likewise, low cash drag could hurt this fund when the stock market declines, but long-term positive returns give this efficient approach a clear edge. Performance across share classes will vary owing to differences in fees and currency exchange rates for non-US investors.",
            "author": {
                "name": "Brendan McCann"
            }
        },
        {
            "id": "process",
            "title": "Process",
            "subtitle": "Approach",
            "subsection_id": "process_approach",
            "q": false,
            "publish_date": "20260128",
            "text": "This strategy accurately captures the US stock market and benefits from the efficiencies of market-cap weighting, earning it a High Process Pillar rating.",
            "author": {
                "name": "Brendan McCann"
            }
        },
        {
            "id": "process",
            "title": "Process",
            "subtitle": "Portfolio",
            "subsection_id": "process_portfolio",
            "q": false,
            "publish_date": "20260128",
            "text": "The S&P 500 selects 500 of the largest US companies that pass its liquidity and profitability screens. Companies are eligible for inclusion only when the sum of their GAAP earnings over the past four quarters is positive, as well as in the most recent quarter. Screening for profitability imparts a slight quality tilt to the portfolio. There have been instances where the profitability screen prevented otherwise qualified companies from index inclusion. Most notably, Tesla was first added to the index in December 2020, despite passing the liquidity and market-cap thresholds in January 2013. Once the index committee selects stocks, it weights them by market cap.  Market-cap weighting is a sensible approach for the US stock market. Highly traded stocks usually reflect new information quickly, and market-cap weighting requires minimal trading costs, which can detract from returns. It follows the wisdom of crowds and takes the guesswork out of stock selection. The US stock market has historically produced solid long-term gains, and owning about 80% of the market has allowed investors to capitalize on those gains. Should strong market performance continue, the fund is well-positioned to reap those rewards.   Market-cap weighting tilts the index toward the largest and most established names. Companies with wide or narrow Morningstar Economic Moat Ratings dominate the portfolio, showcasing the strategy’s durability. Holding 500 stocks reduces the opportunity cost of missing out on strong performers, too. When a portfolio owns a greater chunk of the US stock universe, it has a better chance of capturing gains from companies that end up driving returns. Concentrated active funds are more likely to miss out if those stocks are excluded from their narrow portfolios.  Large allocations to the biggest names in the US stock market present some concentration risk, but the index simply represents the market. While higher concentration may be a concern for investors, there isn’t a clear relationship between index performance and market concentration. In addition, the largest companies, such as Apple and Microsoft, often have diversified business lines, so they don’t rely on a single product, service, or market to determine company success.",
            "author": {
                "name": "Brendan McCann"
            }
        },
        {
            "id": "people",
            "title": "People",
            "q": false,
            "publish_date": "20260128",
            "text": "State Street Investment Management's global equity beta solutions team garners an Above Average People Pillar rating, reflecting a robust and well-coordinated management structure that effectively oversees its stock ETFs and mutual funds. The team ensures that these investment vehicles accurately track their respective target indexes. It employs standardized processes across all State Street Investment Management’s global index-tracking funds irrespective of geographical location. The infrastructure includes coordinated research and trading functions across offices in Boston, London, and Hong Kong.   A group of research analysts complements the team's sizable roster of approximately 50 portfolio managers. These analysts play an important role by coordinating changes to a portfolio during index rebalances and corporate actions. Managers may occasionally deviate from index rules in these situations, within allowable tolerances, when it provides a benefit to a fund.   State Street Investment Management aligns the interests of managers and investors by linking manager compensation to index-tracking performance and operational efficiency. Independent risk oversight further underscores the team's commitment to accountability and tracking performance. Consistently tight tracking error and minimal capital gains distributions speak to the team's effectiveness and proficiency.",
            "author": {
                "name": "Brendan McCann"
            }
        },
        {
            "id": "parent",
            "title": "Parent",
            "q": false,
            "publish_date": "20250716",
            "text": "State Street Investment Management is stepping into unproven territory with recent launches, but most of its low-cost offerings remain competitive, supporting an Above Average Parent Pillar rating.  In June 2025, State Street rebranded State Street Global Advisors to State Street Investment Management, complete with a new logo. The update maintains consistency with other State Street product lines and provides greater clarity to investors. According to State Street, the rebrand does not change the firm’s investment philosophy, product strategy, or leadership.   Since taking over as the CEO in 2022, Yie-Hsin Hung has pushed the firm to move more quickly in product development. Hung hired Anna Paglia as the chief business officer in 2024 and Mark Alberici as the firm’s first head of product innovation in 2025. In late 2024 and early 2025, State Street launched unique and thematic exchange-traded funds through partnerships with firms like Apollo and Galaxy Asset Management. State Street’s private credit ETF is its first to include private assets in the ETF structure, and it remains to be seen how it will navigate liquidity constraints. The ETFs launched with Galaxy focus on the blockchain and digital assets.  Although some of these new strategies are riskier and more niche than the core ETFs that represent the bulk of State Street’s USD 1.7 trillion in assets under management, they are backed by the firm’s solid capital markets group and risk-management team. The firm is adept at handling rapid flows from short-term investors using its products as trading tools, which bodes well for its ability to manage novel liquidity challenges.  State Street continues to compete on price. Roughly 90% of its share classes carry below-average expenses, and–by the firm’s estimate–fee reductions in the fourth quarter of 2023 have saved investors more than USD 60 million in costs through March 2025.",
            "author": {
                "name": "Brendan McCann"
            }
        },
        {
            "id": "performance",
            "title": "Performance",
            "q": false,
            "publish_date": "20260128",
            "text": "The S&P 500 returned 14.8% annualized over the past 10 years through 2025. The State Street funds following this strategy vary in fees, influencing relative returns. However, fees across the board tend to be low, allowing them to capture nearly all of the S&P 500’s performance. Currency performance across regions can also drive relative returns. Additionally, most share classes engage in securities lending, which allows them to earn back a portion of their fees, slightly improving investor returns.   The strategy’s performance closely follows the ups and downs of the US stock market, since it is always fully invested. All else equal, this strategy should outperform Morningstar Category peers that hold cash during market rallies, holding back returns. But no cash buffer also means that the strategy may lag similar peers when the market falls.  The strategy tends to favor the largest US stocks and will perform best when those stocks soar. That’s been the case over the past decade or so. The market’s largest stocks, like Nvidia, have dominated the S&P 500’s returns. However, if mid- and small-cap stocks outperform, it will lag peers that favor smaller companies.   Investors should expect meaningful fluctuations in performance over shorter periods because of the S&P 500’s dependence on the market’s largest companies. In its more-than-50-year history, the S&P 500 has registered a negative annual return about 20% of the time. However, this is still less often than its average US large-blend peer, and only twice did it decline over a 10-year period.   For non-US investors, State Street offers traditional or currency-hedged funds that replicate this strategy. While hedged funds mitigate currency risk, the cost of hedging can erode returns. Unhedged funds are exposed to currency fluctuations, but the impact of foreign-exchange rates on total return tends to wash out in the long term.",
            "author": {
                "name": "Brendan McCann"
            }
        },
        {
            "id": "price",
            "title": "Price",
            "q": false,
            "publish_date": "20260128",
            "text": "It's critical to evaluate expenses, as they come directly out of returns. Based on our assessment of the fund's People, Process, and Parent Pillars in the context of these expenses, we think this share class will be able to deliver positive alpha relative to the lesser of its median category peer or the category benchmark index, explaining its Morningstar Medalist Rating of Silver.",
            "author": {
                "name": "Brendan McCann"
            }
        },
        {
            "id": "sustainability",
            "title": "Sustainability",
            "q": true,
            "publish_date": "20250531",
            "text": "Morningstar generates quantitatively driven content that covers the environmental, social, and governance characteristics for managed investments that have both a Morningstar Sustainability Rating and a Morningstar Carbon Risk Score, called the Sustainability Strategy Summary. This share class’ Sustainability Strategy Summary content was not generated because of insufficient data. To generate individualized content, the Sustainability Strategy Summary requires sufficient data to create its framework of “mental models” designed to mimic content written by analysts. The Sustainability Strategy Summary uses an algorithm designed to predict the ESG analysis that analysts would produce on the investment product if they covered it."
        }
    ],
    "q_full_report_id": "63723206e26a17cb57bfd41a"
}

## Performance

Header: https://www.interactivebrokers.ie/tws.proxy/fundamentals/mf_performance/756733?risk_period=6M&statistic_period=1Y

Possible payload options for both params ['6M', '1Y', '3Y', '5Y', '10Y']

Response:
```
{
    "title_vs": "S&P 500 Index Funds",
    "cumulative": [
        {
            "name": "1 Day",
            "name_tag": "1_Day",
            "name_tag_arg": "2026-02-12",
            "value_fmt": "0.06%",
            "value": 0.063522,
            "vs": 0.01608197249918434,
            "min": 0.0357015,
            "max": 0.1141553,
            "avg": 0.06269440733944955,
            "min_fmt": "0%",
            "max_fmt": "1.09%",
            "avg_fmt": "0.06%",
            "id": "PercentageGrowthCumulative_1D"
        },
        {
            "name": "1 Week",
            "name_tag": "1_Week",
            "name_tag_arg": "2026-02-06",
            "value_fmt": "-1.35%",
            "value": -1.3470677,
            "vs": 0.2370762017245023,
            "min": -1.3847793,
            "max": -1.3232514,
            "avg": -1.3544685412844035,
            "min_fmt": "-2.1%",
            "max_fmt": "0.44%",
            "avg_fmt": "-1.35%",
            "id": "PercentageGrowthCumulative_1W"
        },
        {
            "name": "1 Month",
            "name_tag": "1_Month",
            "name_tag_arg": "2025-12-31",
            "value_fmt": "1.44%",
            "value": 1.4415018,
            "vs": 0.6597634136334477,
            "min": 1.3173797,
            "max": 1.4537445,
            "avg": 1.4177615972477062,
            "min_fmt": "-3.44%",
            "max_fmt": "3.29%",
            "avg_fmt": "1.42%",
            "id": "PercentageGrowthCumulative_1M"
        },
        {
            "name": "YTD",
            "name_tag": "YTD",
            "name_tag_arg": "2025-12-31",
            "value_fmt": "1.44%",
            "value": 1.4415018,
            "vs": 0.6597634136334477,
            "min": 1.3173797,
            "max": 1.4537445,
            "avg": 1.4177615972477062,
            "min_fmt": "-3.44%",
            "max_fmt": "3.29%",
            "avg_fmt": "1.42%",
            "id": "PercentageGrowthCumulative_YT"
        },
        {
            "name": "1 Year",
            "name_tag": "1_Year",
            "name_tag_arg": "2025-01-31",
            "value_fmt": "16.23%",
            "value": 16.2267401,
            "vs": 0.7605013827007879,
            "min": 14.8662296,
            "max": 16.3240572,
            "avg": 15.91772040733945,
            "min_fmt": "3.04%",
            "max_fmt": "16.34%",
            "avg_fmt": "15.92%",
            "id": "PercentageGrowthCumulative_1Y"
        },
        {
            "name": "3 Years",
            "name_tag": "3_Years",
            "name_tag_arg": "2023-01-31",
            "value_fmt": "77.05%",
            "value": 77.0481969,
            "vs": 0.7213390953143127,
            "min": 70.952247,
            "max": 77.5412644,
            "avg": 75.7718468329532,
            "min_fmt": "53.21%",
            "max_fmt": "78.5%",
            "avg_fmt": "75.77%",
            "id": "PercentageGrowthCumulative_3Y"
        },
        {
            "name": "5 Years",
            "name_tag": "5_Years",
            "name_tag_arg": "2021-01-31",
            "value_fmt": "100.09%",
            "value": 100.0923927,
            "vs": 0.7954616774061646,
            "min": 88.6887481,
            "max": 100.7814153,
            "avg": 97.41274291009175,
            "min_fmt": "78.19%",
            "max_fmt": "118.21%",
            "avg_fmt": "97.41%",
            "id": "PercentageGrowthCumulative_5Y"
        },
        {
            "name": "10 Years",
            "name_tag": "10_Years",
            "name_tag_arg": "2016-01-31",
            "value_fmt": "321.18%",
            "value": 321.1785004,
            "vs": 0.8011091022182549,
            "min": 274.0548258,
            "max": 323.9014797,
            "avg": 310.2106606108929,
            "min_fmt": "232.65%",
            "max_fmt": "326.88%",
            "avg_fmt": "310.21%",
            "id": "PercentageGrowthCumulative_10Y"
        },
        {
            "name": "Since Inception (1993-01-22)",
            "name_tag": "Since_Inception_s",
            "name_tag_arg": "1993-01-22",
            "value_fmt": "2775.22%",
            "value": 2775.2161233,
            "vs": 0.7158012175942001,
            "min": 228.5700947,
            "max": 3445.9884853,
            "avg": 1085.7659807495697,
            "min_fmt": "7.79%",
            "max_fmt": "7557.59%",
            "avg_fmt": "1085.77%",
            "id": "PercentageGrowthCumulative_INC"
        }
    ],
    "annualized": [
        {
            "name": "1 Year",
            "name_tag": "1_Year",
            "name_tag_arg": "2025-01-31",
            "value_fmt": "16.23%",
            "value": 16.226740100000004,
            "vs": 0.7605013827008245,
            "min": 14.866229599999992,
            "max": 16.324057199999984,
            "avg": 15.917720407339448,
            "min_fmt": "3.04%",
            "max_fmt": "16.34%",
            "avg_fmt": "15.92%",
            "id": "AnnualizedPerformance_1Y"
        },
        {
            "name": "3 Years",
            "name_tag": "3_Years",
            "name_tag_arg": "2023-01-31",
            "value_fmt": "20.98%",
            "value": 20.975428587574196,
            "vs": 0.7227125106270685,
            "min": 19.570745735499173,
            "max": 21.087627348148374,
            "avg": 20.68299762313659,
            "min_fmt": "15.28%",
            "max_fmt": "21.3%",
            "avg_fmt": "20.68%",
            "id": "AnnualizedPerformance_3Y"
        },
        {
            "name": "5 Years",
            "name_tag": "5_Years",
            "name_tag_arg": "2021-01-31",
            "value_fmt": "14.88%",
            "value": 14.880446673345027,
            "vs": 0.797553924189116,
            "min": 13.540081233930955,
            "max": 14.959456593317721,
            "avg": 14.56918022141856,
            "min_fmt": "12.25%",
            "max_fmt": "16.89%",
            "avg_fmt": "14.57%",
            "id": "AnnualizedPerformance_5Y"
        },
        {
            "name": "10 Years",
            "name_tag": "10_Years",
            "name_tag_arg": "2016-01-31",
            "value_fmt": "15.46%",
            "value": 15.46400552509728,
            "vs": 0.8057218567834566,
            "min": 14.102070761858654,
            "max": 15.538438364933715,
            "avg": 15.15531322999009,
            "min_fmt": "12.77%",
            "max_fmt": "15.62%",
            "avg_fmt": "15.16%",
            "id": "AnnualizedPerformance_10Y"
        },
        {
            "name": "Since Inception (1993-01-22)",
            "name_tag": "Since_Inception_s",
            "name_tag_arg": "1993-01-22",
            "value_fmt": "10.71%",
            "value": 10.713896623071207,
            "id": "AnnualizedPerformance_INC"
        }
    ],
    "yield": [],
    "risk": [
        {
            "name": "Sharpe Ratio",
            "name_tag": "Sharpe_Ratio",
            "value_fmt": "0.08",
            "value": 0.0843193,
            "vs": 0.7430079619408627,
            "min": 0.0782027,
            "max": 0.0847915,
            "avg": 0.08295408899082568,
            "min_fmt": "-0.01",
            "max_fmt": "0.1",
            "avg_fmt": "0.08",
            "id": "SharpeRatio_6M"
        },
        ...
        {
            "name": "Value at Risk Quantile",
            "name_tag": "Value_at_Risk_Quantile",
            "value_fmt": "-1.16",
            "value": -1.1613227,
            "vs": 0.3086348740592412,
            "min": -1.1764361,
            "max": -1.1544859,
            "avg": -1.1643747412844037,
            "min_fmt": "-1.25",
            "max_fmt": "-0.9",
            "avg_fmt": "-1.16",
            "id": "ValueAtRiskQuantile_6M"
        },
        {
            "name": "Value at Risk Expected Tail Loss",
            "name_tag": "Value_at_Risk_Expected_Tail_Loss",
            "value_fmt": "-1.9",
            "value": -1.9017849,
            "vs": 0.321863970992593,
            "min": -1.9222823,
            "max": -1.8939956,
            "avg": -1.9054819385321102,
            "min_fmt": "-2.07",
            "max_fmt": "-1.34",
            "avg_fmt": "-1.91",
            "id": "ValueAtRiskQuantileEtl_6M"
        }
    ],
    "statistic": [
        {
            "name": "Standard Deviation",
            "name_tag": "Standard_Deviation",
            "value_fmt": "10.88",
            "value": 10.8787201,
            "vs": -0.6005595795495786,
            "min": 10.8691871,
            "max": 10.9349953,
            "avg": 10.893052987155965,
            "min_fmt": "10.84",
            "max_fmt": "12.11",
            "avg_fmt": "10.89",
            "id": "StandardDeviation_1Y"
        },
        ...
        {
            "name": "Semi Variance",
            "name_tag": "Semi_Variance",
            "value_fmt": "15.64",
            "value": 15.640732,
            "vs": -0.8027504248357539,
            "min": 15.6271753,
            "max": 15.8735019,
            "avg": 15.695903965137616,
            "min_fmt": "14.27",
            "max_fmt": "16.56",
            "avg_fmt": "15.7",
            "id": "SemiVariance_1Y"
        }
    ]
}
```


## Holdings

Header: https://www.interactivebrokers.ie/tws.proxy/fundamentals/mf_holdings/756733

Response:
```
{
    "as_of_date": 1769835600000,
    "allocation_self": [
        {
            "name": "Equity",
            "formatted_weight": "99.91%",
            "weight": 99.9088,
            "rank": 1,
            "vs": 107.19870416666667
        },
        {
            "name": "Cash",
            "formatted_weight": "0.05%",
            "weight": 0.0516,
            "rank": 2,
            "vs": 0.28529583333333336
        },
        {
            "name": "Other",
            "formatted_weight": "0.04%",
            "weight": 0.0396,
            "rank": 3,
            "vs": 0.45558125000000005
        }
    ],
    "top_10": [
        {
            "name": "NVIDIA CORPORATION",
            "ticker": "NVDA",
            "rank": 1,
            "assets_pct": "7.83%",
            "conids": [
                4815747,
                13104788,
                84223567,
                835397161,
                837454790
            ]
        },
        ...
        {
            "name": "TESLA, INC.",
            "ticker": "TSLA",
            "rank": 9,
            "assets_pct": "2.04%",
            "conids": [
                76792991,
                78046366,
                144303597,
                172604402,
                242507025
            ]
        },
        {
            "name": "Berkshire Hathaway Inc ORD",
            "ticker": "BRK.B",
            "rank": 10,
            "assets_pct": "1.49%",
            "conids": [
                72063691,
                74964638,
                275210194,
                501785781,
                835599867
            ]
        }
    ],
    "top_10_weight": "38.39%",
    "industry": [
        {
            "name": "Technology",
            "formatted_weight": "44.87%",
            "weight": 44.8681,
            "rank": 1,
            "vs": 48.34125624999998
        },
        {
            "name": "Consumer Cyclicals",
            "formatted_weight": "11.43%",
            "weight": 11.428,
            "rank": 2,
            "vs": 12.481191666666666
        },
        ...
        {
            "name": "Real Estate",
            "formatted_weight": "1.81%",
            "weight": 1.8087,
            "rank": 10,
            "vs": 1.9787520833333332
        }
    ],
    "currency": [
        {
            "name": "US Dollar",
            "formatted_weight": "99.96%",
            "weight": 99.9604,
            "rank": 1,
            "vs": 107.98499583333334,
            "code": "USD"
        },
        {
            "name": "<No Currency>",
            "formatted_weight": "0.04%",
            "weight": 0.0396,
            "rank": 2,
            "vs": 0.34473333333333334
        }
    ],
    "investor_country": [
        {
            "name": "United States",
            "formatted_weight": "97.34%",
            "weight": 97.3418,
            "rank": 1,
            "vs": 104.73608958333331,
            "country_code": "US"
        },
        ...
        {
            "name": "Unidentified",
            "formatted_weight": "0.04%",
            "weight": 0.0396,
            "rank": 8,
            "vs": 0.8406437499999999
        },
        {
            "name": "Canada",
            "formatted_weight": "0.03%",
            "weight": 0.0318,
            "rank": 9,
            "vs": 0.04137708333333335,
            "country_code": "CA"
        }
    ],
    "geographic": {
        "eu": "1.89%",
        "uk": "0.46%",
        "us": "97.34%",
        "na": "0.11%",
        "asia": "0.15%",
        "others": "0.04%"
    }
}
```

## Risk and Statistics

header: https://www.interactivebrokers.ie/tws.proxy/fundamentals/mf_risks_stats/89384980?period=1Y

Possible payload options for both params ['6M', '1Y', '3Y', '5Y', '10Y'] This looks to be redundant with the data from performance

response:
```
{
    "risk": [
        {
            "name": "Sharpe Ratio",
            "name_tag": "Sharpe_Ratio",
            "value_fmt": "0.89",
            "value": 0.8913706,
            "vs": 0.736649496820259,
            "min": 0.2447605,
            "max": 0.9836643,
            "avg": 0.6332047455824863,
            "min_fmt": "-0.31",
            "max_fmt": "1.42",
            "avg_fmt": "0.63",
            "id": "SharpeRatio_1Y"
        },
        ...
        {
            "name": "Value at Risk Quantile",
            "name_tag": "Value_at_Risk_Quantile",
            "value_fmt": "-3.75",
            "value": -3.7478476,
            "vs": -0.1986646515502721,
            "min": -5.9322034,
            "max": -1.1673152,
            "avg": -3.206308674120406,
            "min_fmt": "-20.45",
            "max_fmt": "1.03",
            "avg_fmt": "-3.21",
            "id": "ValueAtRiskQuantile_1Y"
        }
    ],
    "statistic": [
        {
            "name": "Standard Deviation",
            "name_tag": "Standard_Deviation",
            "value_fmt": "13.72",
            "value": 13.7211659,
            "vs": 0.3596553608961597,
            "min": 8.9437561,
            "max": 15.578087,
            "avg": 12.67820936129789,
            "min_fmt": "6.16",
            "max_fmt": "52.43",
            "avg_fmt": "12.68",
            "id": "StandardDeviation_1Y"
        },
        ...
        {
            "name": "Semi Variance",
            "name_tag": "Semi_Variance",
            "value_fmt": "17.91",
            "value": 17.9078048,
            "vs": 0.20533458818438852,
            "min": 5.7537344,
            "max": 29.4207504,
            "avg": 14.93296040836591,
            "min_fmt": "1.28",
            "max_fmt": "228.52",
            "avg_fmt": "14.93",
            "id": "SemiVariance_1Y"
        }
    ],
    "performance": [
        {
            "name": "Relative Average Return",
            "name_tag": "Relative_Average_Return",
            "value_fmt": "0",
            "value": 0.0020094,
            "vs": -0.03716010781002208,
            "min": -1.1393849,
            "max": 1.0374643,
            "avg": 0.04606068577013292,
            "min_fmt": "-3.07",
            "max_fmt": "2.39",
            "avg_fmt": "0.05",
            "id": "RelativeAverageReturn_1Y"
        },
        ...
        {
            "name": "Maximum Drawdown",
            "name_tag": "Maximum_Drawdown",
            "value_fmt": "-6.69",
            "value": -6.6906701,
            "vs": -0.42561294473923256,
            "min": -9.6432964,
            "max": -1.2254902,
            "avg": -4.502814432384052,
            "min_fmt": "-22.34",
            "max_fmt": "0",
            "avg_fmt": "-4.5",
            "id": "MaximumDrawdown_1Y"
        }
    ]
}
```

## Ownership

header: 
https://www.interactivebrokers.ie/tws.proxy/fundamentals/ownership/756733

response:
```
{
    "trade_log": [
        {
            "action": "PURCHASE",
            "shares": 74255,
            "value": 5.1382232E7,
            "holding": 322762,
            "party": "Tuttle Capital Management, LLC",
            "source": "Aggregate MFs",
            "insider": false,
            "displayDate": {
                "d": 31,
                "m": "JANUARY",
                "y": 2026,
                "t": "2026-01-31"
            },
            "display_shares": "74.25K",
            "display_value": "51.38M",
            "display_holding": "322.76K"
        },
        ...
        {
            "action": "NO CHANGE",
            "shares": 0,
            "value": 0.0,
            "holding": 949191,
            "party": "BlackRock Japan Co., Ltd.",
            "source": "13F",
            "insider": false,
            "displayDate": {
                "d": 31,
                "m": "DECEMBER",
                "y": 2025,
                "t": "2025-12-31"
            },
            "display_shares": "0",
            "display_value": "0",
            "display_holding": "949.19K"
        }
    ],
    "ownership_history": {
        "legends": [
            {
                "represent": "ASDATE",
                "display": "BOTTOM",
                "labels": [
                    {
                        "hilight": true,
                        "text": "2020",
                        "value": 1.5778368E12
                    },
                    {
                        "hilight": false,
                        "text": "Q1",
                        "value": 1.5856992E12
                    },
                    {
                        "hilight": false,
                        "text": "Q2",
                        "value": 1.5935616E12
                    },
                    {
                        "hilight": false,
                        "text": "Q3",
                        "value": 1.6015104E12
                    },
                    {
                        "hilight": true,
                        "text": "2021",
                        "value": 1.6094592E12
                    },
                    {
                        "hilight": false,
                        "text": "Q1",
                        "value": 1.6172352E12
                    },
                    {
                        "hilight": false,
                        "text": "Q2",
                        "value": 1.6250976E12
                    },
                    {
                        "hilight": false,
                        "text": "Q3",
                        "value": 1.6330464E12
                    },
                    {
                        "hilight": true,
                        "text": "2022",
                        "value": 1.6409952E12
                    },
                    {
                        "hilight": false,
                        "text": "Q1",
                        "value": 1.6487712E12
                    },
                    {
                        "hilight": false,
                        "text": "Q2",
                        "value": 1.6566336E12
                    },
                    {
                        "hilight": false,
                        "text": "Q3",
                        "value": 1.6645824E12
                    },
                    {
                        "hilight": true,
                        "text": "2023",
                        "value": 1.6725312E12
                    },
                    {
                        "hilight": false,
                        "text": "Q1",
                        "value": 1.6803072E12
                    },
                    {
                        "hilight": false,
                        "text": "Q2",
                        "value": 1.6881696E12
                    },
                    {
                        "hilight": false,
                        "text": "Q3",
                        "value": 1.6961184E12
                    },
                    {
                        "hilight": true,
                        "text": "2024",
                        "value": 1.7040672E12
                    },
                    {
                        "hilight": false,
                        "text": "Q1",
                        "value": 1.7119296E12
                    },
                    {
                        "hilight": false,
                        "text": "Q2",
                        "value": 1.719792E12
                    },
                    {
                        "hilight": false,
                        "text": "Q3",
                        "value": 1.7277408E12
                    },
                    {
                        "hilight": true,
                        "text": "2025",
                        "value": 1.7356896E12
                    },
                    {
                        "hilight": false,
                        "text": "Q1",
                        "value": 1.7434656E12
                    },
                    {
                        "hilight": false,
                        "text": "Q2",
                        "value": 1.751328E12
                    },
                    {
                        "hilight": false,
                        "text": "Q3",
                        "value": 1.7592768E12
                    },
                    {
                        "hilight": true,
                        "text": "2026",
                        "value": 1.7672256E12
                    },
                    {
                        "hilight": false,
                        "text": "Q1",
                        "value": 1.7750016E12
                    },
                    {
                        "hilight": false,
                        "text": "Q2",
                        "value": 1.782864E12
                    },
                    {
                        "hilight": false,
                        "text": "Q3",
                        "value": 1.7908128E12
                    },
                    {
                        "hilight": true,
                        "text": "2027",
                        "value": 1.7987616E12
                    },
                    {
                        "hilight": false,
                        "text": "Q1",
                        "value": 1.8065376E12
                    },
                    {
                        "hilight": false,
                        "text": "Q2",
                        "value": 1.8144E12
                    },
                    {
                        "hilight": false,
                        "text": "Q3",
                        "value": 1.8223488E12
                    }
                ],
                "min": 1614004200000,
                "max": 1771511400000,
                "title": "Calendar",
                "axis": "HORIZONTAL"
            },
            {
                "represent": "ASNUMBER",
                "display": "RIGHT",
                "labels": [
                    {
                        "hilight": true,
                        "text": "307.06",
                        "value": 307.0634667289679
                    },
                    {
                        "hilight": true,
                        "text": "398.66",
                        "value": 398.65857338317437
                    },
                    {
                        "hilight": true,
                        "text": "490.25",
                        "value": 490.25368003738083
                    },
                    {
                        "hilight": true,
                        "text": "581.85",
                        "value": 581.8487866915873
                    },
                    {
                        "hilight": true,
                        "text": "673.44",
                        "value": 673.4438933457938
                    }
                ],
                "min": 341.1816296988532,
                "max": 695.49,
                "title": "Price",
                "axis": "VERTICAL"
            },
            {
                "represent": "ASPERCENTS",
                "display": "LEFT",
                "labels": [],
                "min": 0.0,
                "max": 0.0,
                "title": "Ownership",
                "axis": "VERTICAL"
            }
        ],
        "series": [
            {
                "title": "Price",
                "name": "Price",
                "plotData": [
                    {
                        "x": 1614004200000,
                        "y": 361.2802021835987,
                        "open": 361.3082062299659,
                        "close": 361.2802021835987,
                        "high": 363.69788485330264,
                        "low": 361.00949640204885,
                        "debugY": 20210222
                    },
                    ... tons of data, will want to fetch via ib_async instead...
                    {
                        "x": 1771511400000,
                        "y": 684.48,
                        "open": 683.86,
                        "close": 684.48,
                        "high": 686.1800000000001,
                        "low": 681.54,
                        "debugY": 20260219
                    }
                ],
                "xlegend": "Calendar",
                "ylegend": "Price"
            },
            {
                "title": "Onership",
                "name": "Institutional Investors Pct",
                "plotData": [],
                "xlegend": "Calendar",
                "ylegend": "Ownership"
            },
            {
                "title": "Onership",
                "name": "Strategic Investors Pct",
                "plotData": [],
                "xlegend": "Calendar",
                "ylegend": "Ownership"
            }
        ]
    },
    "owners_types": [
        {
            "type": {
                "type": "INV_MANAGERS",
                "display_type": "Invest. Managers"
            },
            "float": 42.38514722559241,
            "display_float": "42.39%"
        },
        {
            "type": {
                "type": "FUND_PORTFOLIOS",
                "display_type": "Funds / Fund Portfolios"
            },
            "float": 2.090696157215018,
            "display_float": "2.09%"
        },
        {
            "type": {
                "type": "STRAT_ENTITIES",
                "display_type": "Strategic Entities"
            },
            "float": 0.11987757776865679,
            "display_float": "0.12%"
        },
        {
            "type": {
                "type": "BROCK_FIRMS",
                "display_type": "Brokerage Firms"
            },
            "float": 11.06183176108609,
            "display_float": "11.06%"
        }
    ],
    "institutional_owners": [
        {
            "name": "JPMorgan Private Bank (United States)",
            "type": {
                "type": "INV_MANAGERS",
                "display_type": "Ivst Mngr"
            },
            "display_value": "24.25B",
            "display_shares": "35.57M",
            "display_pct": "3.45%"
        },
        ...
        {
            "name": "Michigan Department of Treasury",
            "type": {
                "type": "INV_MANAGERS",
                "display_type": "Ivst Mngr"
            },
            "display_value": "599.98M",
            "display_shares": "879.84K",
            "display_pct": "0.09%"
        }
    ],
    "insider_owners": [
        {
            "name": "Sumitomo Mitsui Financial Group, Inc",
            "display_value": "513.83M",
            "display_shares": "753.51K",
            "display_pct": "0.07%"
        },
        ...
        {
            "name": "Berkshire Hathaway Inc.",
            "display_value": "0",
            "display_shares": "0",
            "display_pct": "0%"
        }
    ],
    "institutional_total": {
        "display_value": "389.29B",
        "display_shares": "572.31M",
        "display_pct": "55.54%"
    },
    "insider_total": {
        "display_value": "839.07M",
        "display_shares": "1.24M",
        "display_pct": "0.12%"
    },
    "institutional_summary": {
        "shares": 5.7230581E8,
        "pct": 55.537675143893516,
        "value": 3.89289460409E11,
        "sharesChange": -0.37954376175213034,
        "pctChange": -0.37954376175213034,
        "valueChange": -0.37954376175213034,
        "display_shares": "572.31M",
        "display_value": "389.29B",
        "display_pct": "55.54%",
        "display_pct_change": "-0.38%",
        "display_shares_change": "-0.38%",
        "display_value_change": "-0.38%"
    },
    "insider_summary": {
        "shares": 1235317.0,
        "pct": 0.11987757776865679,
        "value": 8.3906765E8,
        "sharesChange": -92.68422599219471,
        "pctChange": -92.68422599219471,
        "valueChange": -92.68422599219471,
        "display_shares": "1.24M",
        "display_value": "839.07M",
        "display_pct": "0.12%",
        "display_pct_change": "-92.68%",
        "display_shares_change": "-92.68%",
        "display_value_change": "-92.68%"
    },
    "others_summary": {
        "shares": 4.56940989E8,
        "pct": 44.34244727833782,
        "value": 3.17081043331E11,
        "display_shares": "456.94M",
        "display_value": "317.08B",
        "display_pct": "44.34%"
    }
}
```

## Ratios and Fundamentals

Header: https://www.interactivebrokers.ie/tws.proxy/fundamentals/mf_ratios_fundamentals/75961307

Response:
```
{
  "as_of_date": 1769835600000,
  "ratios": [
    {
      "name": "Price/Sales",
      "name_tag": "price_sales",
      "value_fmt": "3.63",
      "value": 3.6316317099,
      "vs": 0.0146752846905356,
      "min": 2.0848380499,
      "max": 5.5932789057,
      "avg": 3.60241521851044,
      "min_fmt": "0.91",
      "max_fmt": "10.71",
      "avg_fmt": "3.6",
      "percentile": 40.4705882352941,
      "id": "1004923"
    },
    ...
    {
      "name": "Dividend Per Share 3Yr",
      "name_tag": "Dividend_Per_Share_3Yr",
      "value_fmt": "24.81",
      "value": 24.8105201101,
      "vs": 0.792911845340361,
      "min": 11.4811869316,
      "max": 26.2840336645,
      "avg": 19.1686409139016,
      "min_fmt": "-4.8",
      "max_fmt": "33.06",
      "avg_fmt": "19.17",
      "percentile": 11.0588235294118,
      "id": "1005478"
    }
  ],
  "zscore": [
    {
      "name": "Average Final Composite Zscore",
      "name_tag": "Average_Final_Composite_Zscore",
      "value_fmt": "-0.04",
      "value": -0.0397692935,
      "id": "1006647"
    },
    ...
    {
      "name": "Weighted Final Composite ZScore",
      "name_tag": "Weighted_Final_Composite_ZScore",
      "value_fmt": "-0.01",
      "value": -0.0086158938,
      "id": "1006640"
    }
  ]
}
```

## Dividends

Can return two types from the same endpoint

### Short summary

Header: 
https://www.interactivebrokers.ie/tws.proxy/fundamentals/dividends/564156940

Response:
```
{
    "industry_average": {
        "dividend_yield": "1.22%",
        "annual_dividend": "25.65",
        "paying_companies": 0,
        "paying_companies_percent": 0
    },
    "no_dividend_text": "No dividend data available for URNU in the past 5 years",
    "no_div_data_marker": 1,
    "no_div_data_period": 5
}
```

### Long summary

Header:
https://www.interactivebrokers.ie/tws.proxy/fundamentals/dividends/237937002

Response:
```
{
    "history": {
        "legends": [
            {
                "represent": "ASDATE",
                "display": "BOTTOM",
                "labels": [],
                "min": 1613948340000,
                "max": 1771718400,
                "title": "Calendar",
                "axis": "HORIZONTAL"
            },
            {
                "represent": "ASNUMBER",
                "display": "RIGHT",
                "labels": [
                    {
                        "hilight": true,
                        "text": "419.96",
                        "value": 419.9628573704165
                    },
                    {
                        "hilight": true,
                        "text": "566.28",
                        "value": 566.2761164881313
                    },
                    {
                        "hilight": true,
                        "text": "712.59",
                        "value": 712.589375605846
                    },
                    {
                        "hilight": true,
                        "text": "858.9",
                        "value": 858.9026347235608
                    },
                    {
                        "hilight": true,
                        "text": "1.01K",
                        "value": 1005.2158938412756
                    },
                    {
                        "hilight": true,
                        "text": "1.15K",
                        "value": 1151.5291529589904
                    }
                ],
                "min": 466.62539707824055,
                "max": 1046.8446845081733,
                "title": "Price",
                "axis": "VERTICAL"
            },
            {
                "represent": "ASNUMBER",
                "display": "LEFT",
                "labels": [
                    {
                        "hilight": true,
                        "text": "1.15",
                        "value": 1.1500092
                    },
                    {
                        "hilight": true,
                        "text": "1.36",
                        "value": 1.35854832
                    },
                    {
                        "hilight": true,
                        "text": "1.57",
                        "value": 1.5670874399999999
                    },
                    {
                        "hilight": true,
                        "text": "1.78",
                        "value": 1.7756265599999999
                    },
                    {
                        "hilight": true,
                        "text": "1.98",
                        "value": 1.9841656799999998
                    },
                    {
                        "hilight": true,
                        "text": "2.19",
                        "value": 2.1927048
                    }
                ],
                "min": 1.277788,
                "max": 1.993368,
                "title": "Dividends",
                "axis": "VERTICAL"
            }
        ],
        "series": [
            {
                "title": "Dividends",
                "name": "dividends",
                "plotData": [
                    {
                        "x": 1616112000000,
                        "y": 1.277788,
                        "type": "ACTUAL",
                        "amount": 1.277788,
                        "ex_dividend_date": {
                            "d": 19,
                            "m": "MAR",
                            "y": 2021
                        },
                        "formatted_amount": "1.28 USD",
                        "description": "Regular Dividend",
                        "declaration_date": {
                            "d": 19,
                            "m": "MAR",
                            "y": 2021
                        },
                        "record_date": {
                            "d": 22,
                            "m": "MAR",
                            "y": 2021
                        },
                        "payment_date": {
                            "d": 12,
                            "m": "MAY",
                            "y": 2021
                        }
                    },
                    ...
                    {
                        "x": 1766102400000,
                        "y": 1.993368,
                        "type": "ACTUAL",
                        "amount": 1.993368,
                        "ex_dividend_date": {
                            "d": 19,
                            "m": "DEC",
                            "y": 2025
                        },
                        "formatted_amount": "1.99 USD",
                        "description": "Regular Dividend",
                        "declaration_date": {
                            "d": 19,
                            "m": "DEC",
                            "y": 2025
                        },
                        "record_date": {
                            "d": 22,
                            "m": "DEC",
                            "y": 2025
                        },
                        "payment_date": {
                            "d": 13,
                            "m": "FEB",
                            "y": 2026
                        }
                    }
                ],
                "xlegend": "Calendar",
                "ylegend": "Dividends"
            },
            {
                "title": "Price",
                "name": "price",
                "plotData": [
                    {
                        "x": 1613948340000,
                        "y": 473.22429485511105,
                        "open": 476.15500997807527,
                        "close": 473.22429485511105,
                        "high": 476.15500997807527,
                        "low": 473.02316734667227,
                        "debugY": 20210221
                    },
                    {
                        "x": 1614034740000,
                        "y": 470.41808723737074,
                        "open": 478.87502009219895,
                        "close": 470.41808723737074,
                        "high": 478.87502009219895,
                        "low": 465.4760970300192,
                        "debugY": 20210222
                    },
                    ...
                    {
                        "x": 1771455540000,
                        "y": 972.62,
                        "open": 967.5,
                        "close": 972.62,
                        "high": 977.25,
                        "low": 967.5,
                        "debugY": 20260218
                    },
                    {
                        "x": 1771541940000,
                        "y": 975.52,
                        "open": 970.3100000000001,
                        "close": 975.52,
                        "high": 976.64,
                        "low": 969.69,
                        "debugY": 20260219
                    }
                ],
                "xlegend": "Calendar",
                "ylegend": "Price"
            }
        ]
    },
    "industry_comparison": {
        "content": [
            {
                "title": "Dividend Yield TTM",
                "title_tag": "Dividend_Yield_TTM",
                "value": 0.008496586146673148,
                "formatted_value": "0.85%",
                "search_id": "div_yield"
            },
            {
                "title": "Dividend TTM",
                "title_tag": "Dividend_TTM",
                "value": 5.585599,
                "formatted_value": "5.59",
                "search_id": "div_per_share"
            }
        ],
        "title": "Industry Comparison",
        "title_tag": "Industry_Comparison"
    },
    "no_div_data_marker": 2,
    "last_payed_dividend_date": {
        "d": 19,
        "m": "DEC",
        "y": 2025,
        "t": "2025-12-19"
    },
    "last_payed_dividend_amount": 1.993368,
    "last_payed_dividend_currency": "USD",
    "formatted_last_payed_dividend_amount": "1.99"
}
```

## Impact

Header:
https://www.interactivebrokers.ie/tws.proxy/impact/esg/564156940?accounts=U19746488

Response:
```
{"title":"ESG","content":[{"name":"TRESGS","value":5},{"name":"TRESGCS","value":5},{"name":"TRESGCCS","value":8},{"name":"TRESGENS","value":5,"children":[{"name":"TRESGENRRS","value":5},{"name":"TRESGENERS","value":6},{"name":"TRESGENPIS","value":1}]},{"name":"TRESGSOS","value":5,"children":[{"name":"TRESGSOWOS","value":6},{"name":"TRESGSOHRS","value":5},{"name":"TRESGSOCOS","value":6},{"name":"TRESGSOPRS","value":4}]},{"name":"TRESGCGS","value":5,"children":[{"name":"TRESGCGBDS","value":6},{"name":"TRESGCGSRS","value":5},{"name":"TRESGCGVSS","value":5}]}],"asOfDate":"20251231","coverage":0.838711013577,"source":"REFINITIV_LIPPER","symbol":"URNU","no_settings":true}
```

## Sentiment

Header:
https://www.interactivebrokers.ie/tws.proxy/sma/request?type=search&conid=237937002&from=2021-02-21%2012:40&to=2026-02-21%2012:40&bar_size=1D&tz=-60

Response:
```
{
    "sentiment": [
        {
            "datetime": 1740114000000,
            "svolatility": 2.1045,
            "sdispersion": 0.398,
            "svscore": 0.3972,
            "sbuzz": 1.3417,
            "svolume": 1113,
            "sdelta": 0.0039,
            "sscore": -0.465,
            "smean": 2.4122
        },
        {
            "sdispersion": 0.526,
            "svscore": -1.3189,
            "sbuzz": 0.3846,
            "sdelta": -0.0389,
            "datetime": 1740286800000,
            "svolatility": 2.055,
            "high": 944.5876336506726,
            "low": 937.8760192893193,
            "price": 939.7254419133367,
            "svolume": 390,
            "close": 939.7254419133367,
            "open": 939.6061243246904,
            "sscore": -1.3996,
            "smean": 2.4023
        },
        {
            "price_change": 2.674702612154192,
            "sdispersion": 0.385,
            "svscore": 1.3177,
            "sbuzz": 2.1962,
            "sdelta": 0.0431,
            "datetime": 1740459600000,
            "svolatility": 1.9671,
            "high": 937.2794313460879,
            "low": 932.3575808144288,
            "price": 937.0507393011825,
            "price_change_p": 0.28543839730054377,
            "svolume": 1476,
            "close": 937.0507393011825,
            "open": 936.444208225564,
            "sscore": -0.9849,
            "smean": 2.1879
        },
        ...
        {
            "price_change": -2.8999999999999773,
            "sdispersion": 0.4017595552467,
            "svscore": 0.43059242529534425,
            "sbuzz": 1.5178863099374544,
            "sdelta": -0.0038222376650451787,
            "datetime": 1771477200000,
            "svolatility": 1.7422417651146602,
            "high": 976.64,
            "low": 969.69,
            "price": 975.52,
            "price_change_p": -0.29727734951615314,
            "svolume": 1035.1646977067426,
            "close": 975.52,
            "open": 970.3100000000001,
            "sscore": 0.3973481584433641,
            "smean": 1.1370299513551079
        },
        {
            "datetime": 1771563600000,
            "svolatility": 1.73575,
            "sdispersion": 0.373,
            "svscore": 0.49395,
            "sbuzz": 1.6033,
            "svolume": 1069,
            "sdelta": -0.2448,
            "sscore": -0.23795,
            "smean": 1.0806
        },
        {
            "datetime": 1771650000000,
            "svolatility": 1.7783499999999999,
            "sdispersion": 0.3807512562814065,
            "svscore": 0.750520603015076,
            "sbuzz": 1.7732567839195943,
            "svolume": 1212.281407035175,
            "sdelta": 0.005522361809045202,
            "sscore": 0.5566027638190957,
            "smean": 1.1194452261306536
        }
    ]
}
```

### Tick snapshot

Header: 
https://www.interactivebrokers.ie/tws.proxy/sma/request?type=tick&conid=237937002

Response:
```
{
    "schange": 0.20295184155663587,
    "datetime": 1771674120000,
    "svolatility": 1.7863,
    "sdispersion": 0.389,
    "svscore": 0.6983,
    "sbuzz": 1.7465,
    "svolume": 1193,
    "sdelta": -0.0232,
    "svchange": 0.2677075747046558,
    "sscore": 0.6003,
    "smean": 1.1163
}
```

### High-low

Header:
https://www.interactivebrokers.ie/tws.proxy/sma/request?type=high_low&conid=237937002

Response:
```
{
    "sscore_high_date": 1770699600000,
    "svscore_low": -1.6834,
    "svscore_high_date": 1770354000000,
    "svscore_high": 1.45385,
    "svscore_low_date": 1770526800000,
    "sscore_low": -1.4894500000000002,
    "sscore_high": 2.1028000000000002,
    "sscore_low_date": 1770354000000
}
```