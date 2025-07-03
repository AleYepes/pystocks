import os
import pandas as pd
import numpy as np
import re
import ast
from tqdm.auto import tqdm
from datetime import datetime, timedelta

from ib_async import *
import pandas_datareader.data as web
import wbgapi as wb
import country_converter as coco

from scipy.stats import skew
from scipy.optimize import minimize_scalar

from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

# pd.set_option('display.max_colwidth', None)
# pd.set_option('display.max_columns', None)


def evaluate_literal(val):
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return val
    
def load(path):
    df = pd.read_csv(path)
    for col in df.columns:
        df[col] = df[col].apply(evaluate_literal)
    return df

def ensure_series_types(df, price_col):
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    for col in ['volume', price_col]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def validate_raw_prices(df, price_col):
    invalid_price_mask = df[price_col] <= 0
    inconsistent_mask = pd.Series(False, index=df.index)
    if 'low' in df.columns and 'high' in df.columns:
        inconsistent_mask = (df['low'] > df['high'])
    local_error_mask = invalid_price_mask | inconsistent_mask
    df = df[~local_error_mask].copy()
    return df

def handle_stale_periods(df, price_col, max_stale_days=5):
    stale_groups = (df[price_col].diff() != 0).cumsum()
    if stale_groups.empty:
        return df
    period_lengths = df.groupby(stale_groups)[price_col].transform('size')
    long_stale_mask = period_lengths > max_stale_days
    is_intermediate_stale_row = (stale_groups.duplicated(keep='first') & stale_groups.duplicated(keep='last'))
    rows_to_drop_mask = long_stale_mask & is_intermediate_stale_row
    df = df[~rows_to_drop_mask].copy()
    return df

def detect_and_nullify_global_outliers(meta_df, price_col, z_threshold=120.0, window=5):
    all_pct_changes = pd.concat(
        [row['df']['pct_change'] for _, row in meta_df.iterrows()],
        ignore_index=True
    ).dropna()
    all_pct_changes = all_pct_changes[~np.isinf(all_pct_changes) & (all_pct_changes != 0)]

    global_median_return = all_pct_changes.median()
    global_mad = (all_pct_changes - global_median_return).abs().median()

    for idx, row in meta_df.iterrows():
        df = row['df']
        df = df.reset_index(drop=True)
        if df['pct_change'].isnull().all():
            continue
        cols_to_null = [price_col, 'volume', 'high', 'low', 'pct_change']
        cols_to_null = [c for c in cols_to_null if c in df.columns]

        absolute_modified_z = (df['pct_change'] - global_median_return).abs() / global_mad
        outlier_mask = absolute_modified_z > z_threshold

        if outlier_mask.any():

            candidate_indices = df.index[outlier_mask]
            for df_idx in candidate_indices:
                price_to_check_idx = df_idx - 1
                price_to_check = df.loc[price_to_check_idx, price_col]
                local_window_start = max(0, price_to_check_idx - window)
                local_window = df.loc[local_window_start : price_to_check_idx - 1, price_col].dropna()
                local_mean = local_window.mean()
                local_std = local_window.std()
                if local_std != 0: 
                    price_z_score = abs(price_to_check - local_mean) / local_std
                    if price_z_score > z_threshold / 10:
                        df.loc[price_to_check_idx, cols_to_null] = np.nan

                price_to_check = df.loc[df_idx, price_col]
                local_window_end = min(df_idx + window, df.index[outlier_mask].max())
                local_window = df.loc[df_idx + 1: local_window_end, price_col].dropna()
                local_mean = local_window.mean()
                local_std = local_window.std()
                if local_std != 0:
                    price_z_score = abs(price_to_check - local_mean) / local_std
                    if price_z_score > z_threshold / 10:
                        df.loc[df_idx, cols_to_null] = np.nan

            df['pct_change'] = df[price_col].pct_change(fill_method=None)
            meta_df.at[idx, 'df'] = df

def calculate_slope(value1, value2, time1, time2):
    return (value1 - value2) / (time1 - time2)

def get_return_stats(df, training_cutoff, momentum_cutoffs, risk_free_df):
    training_df = df[df.index < training_cutoff]
    training_rf = risk_free_df[risk_free_df.index < training_cutoff]

    excess_returns = training_df['pct_change'] - training_rf['daily_nominal_rate']
    sharpe = excess_returns.mean() / excess_returns.std()

    momentum_3mo = training_df[training_df.index >= momentum_cutoffs['3mo']]['pct_change'].mean()
    momentum_6mo = training_df[training_df.index >= momentum_cutoffs['6mo']]['pct_change'].mean()
    momentum_1y  = training_df[training_df.index >= momentum_cutoffs['1y']]['pct_change'].mean()

    return pd.Series(
        [momentum_3mo, momentum_6mo, momentum_1y, sharpe],
        index=['momentum_3mo', 'momentum_6mo', 'momentum_1y', 'stats_sharpe']
    )

def create_continent_map(standard_names):
    continents = cc.convert(names=standard_names, to='continent', not_found=None)
    return {name: (cont if cont is not None else 'Other')
            for name, cont in zip(standard_names, continents)}

def create_metric_maps(standard_names, indicators, start_year, end_year, window_size=3):
    data = wb.data.DataFrame(list(indicators), standard_names, time=range(2000, end_year.year + 1), labels=False)
    data.dropna(axis=1, inplace=True) 
    
    if len(data.columns) <= window_size * 2:
         raise (f"Warning: Not enough data points ({len(data.columns)} years) for a window size of {window_size} + 1 required for derivatives. Skipping derivative calculation.")

    yoy_change = data.diff(axis=1)
    first_div = yoy_change.rolling(window=window_size, axis=1).mean()

    yoy_change_first_div = first_div.diff(axis=1)
    second_div = yoy_change_first_div.rolling(window=window_size, axis=1).mean()

    latest_year_col = data.columns[-1]
    latest_first_div_col = first_div.columns[-1]
    latest_second_div_col = second_div.columns[-1]

    derivatives = pd.DataFrame(data[latest_year_col])
    derivatives.rename(columns={latest_year_col: 'raw_value'}, inplace=True)
    derivatives['1st_div'] = first_div[latest_first_div_col] / derivatives['raw_value']
    derivatives['2nd_div'] = second_div[latest_second_div_col] / derivatives['raw_value']
    
    metric_df_reshaped = derivatives.unstack(level='series')
    if isinstance(metric_df_reshaped.columns, pd.MultiIndex):
         metric_df_final = metric_df_reshaped.swaplevel(0, 1, axis=1)
         metric_df_final.sort_index(axis=1, level=0, inplace=True)
    else:
         metric_df_final = metric_df_reshaped

    return metric_df_final

def construct_long_short_factor_returns(full_meta_df, returns_df, long_symbols, short_symbols, factor_column=None):
    long_df = full_meta_df[full_meta_df['conId'].isin(long_symbols)].set_index('conId')
    long_weights = long_df['profile_cap_usd'].reindex(returns_df.columns).fillna(0)
    if long_weights.mean() == 0:
        print(f'Long {factor_column}')
        print(long_df.index)
        print()
    if factor_column:
        factor_weights = (full_meta_df[factor_column].max() - long_df[factor_column]) / (full_meta_df[factor_column].max() - full_meta_df[factor_column].min())
        factor_weights = factor_weights.reindex(returns_df.columns).fillna(0)
        if factor_weights.sum() != 0:
            long_weights *= factor_weights

    long_weights /= long_weights.sum()
    long_returns = returns_df.dot(long_weights)
    
    short_df = full_meta_df[full_meta_df['conId'].isin(short_symbols)].set_index('conId')
    short_weights = short_df['profile_cap_usd'].reindex(returns_df.columns).fillna(0)
    if short_weights.mean() == 0:
        print(f'Short {factor_column}')
        print(short_df.index)
        print()
    if factor_column:
        factor_weights = (short_df[factor_column] - full_meta_df[factor_column].min()) / (full_meta_df[factor_column].max() - full_meta_df[factor_column].min())
        factor_weights = factor_weights.reindex(returns_df.columns).fillna(0)
        if factor_weights.sum() != 0:
            short_weights *= factor_weights

    short_weights /= short_weights.sum()
    short_returns = returns_df.dot(short_weights)
    
    factor_returns = long_returns - short_returns
    return factor_returns

def construct_factors(filtered_df, pct_changes, portfolio_dfs, risk_free_df, scaling_factor=0.5):
    factors = {}
    # Market risk premium
    factors['factor_market_premium'] = (portfolio_dfs['equity']['pct_change'] - risk_free_df['daily_nominal_rate'])

    # SMB_ETF
    small_symbols = filtered_df[filtered_df['marketcap_small'] == 1]['conId'].tolist()
    large_symbols = filtered_df[filtered_df['marketcap_large'] == 1]['conId'].tolist()

    intersection = set(small_symbols) & set(large_symbols)
    small_symbols = [s for s in small_symbols if s not in intersection]
    large_symbols = [s for s in large_symbols if s not in intersection]
    smb_etf = construct_long_short_factor_returns(filtered_df, pct_changes, small_symbols, large_symbols)
    factors['factor_smb'] = smb_etf

    # HML_ETF
    value_cols = [col for col in filtered_df.columns if col.startswith('style_') and col.endswith('value')]
    growth_cols = [col for col in filtered_df.columns if col.startswith('style_') and col.endswith('growth')]
    value_symbols = filtered_df[filtered_df[value_cols].ne(0).any(axis=1)]['conId'].tolist()
    growth_symbols = filtered_df[filtered_df[growth_cols].ne(0).any(axis=1)]['conId'].tolist()

    intersection = set(value_symbols) & set(growth_symbols)
    value_symbols = [s for s in value_symbols if s not in intersection]
    growth_symbols = [s for s in growth_symbols if s not in intersection]
    hml_etf = construct_long_short_factor_returns(filtered_df, pct_changes, value_symbols, growth_symbols)
    factors['factor_hml'] = hml_etf

    # Metadata
    excluded = ['style_', 'marketcap_', 'countries_', 'fundamental']
    numerical_cols = [col for col in filtered_df.columns if filtered_df[col].dtype in [np.int64, np.float64] and col not in ['conId']]
    for col in numerical_cols:
        if not any(col.startswith(prefix) for prefix in excluded) and col in filtered_df.columns:
            try:
                std = filtered_df[col].std()
                mean = filtered_df[col].mean()

                upper_boundary = min(filtered_df[col].max(), mean + (scaling_factor * std))
                lower_boundary = max(filtered_df[col].min(), mean - (scaling_factor * std))

                low_factor_symbols = filtered_df[filtered_df[col] <= lower_boundary]['conId'].tolist()
                high_factor_symbols = filtered_df[filtered_df[col] >= upper_boundary]['conId'].tolist()
                if col.endswith('variety'):
                    var_etf = construct_long_short_factor_returns(filtered_df, pct_changes, low_factor_symbols, high_factor_symbols, factor_column=col)
                else:
                    var_etf = construct_long_short_factor_returns(filtered_df, pct_changes, high_factor_symbols, low_factor_symbols, factor_column=col)
                var_etf = construct_long_short_factor_returns(filtered_df, pct_changes, high_factor_symbols, low_factor_symbols, factor_column=col)
                factors[col] = var_etf

            except Exception as e:
                print(col)
                print(e)
                raise

    return pd.DataFrame(factors)

def prescreen_factors(factors_df, correlation_threshold=0.99, drop_map=None):
    if factors_df is None or factors_df.empty or factors_df.shape[1] == 0:
        raise ValueError("factors_df must be a non-empty DataFrame with at least one column.")
    temp_factors_df = factors_df.copy()

    corr_matrix = temp_factors_df.corr().abs()
    corr_pairs = corr_matrix.where(np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)).stack()
    corr_pairs = corr_pairs.sort_values(ascending=False)

    if not drop_map:
        drop_map = {}
    col_order = list(temp_factors_df.columns)
    for (col1, col2), corr_val in corr_pairs.items():
        if corr_val < correlation_threshold:
            break

        already_dropped = {c for drops in drop_map.values() for c in drops}
        if col1 in already_dropped or col2 in already_dropped:
            continue

        if col_order.index(col1) < col_order.index(col2):
            keeper, to_drop = col1, col2
        else:
            keeper, to_drop = col2, col1

        drop_map.setdefault(keeper, []).append(to_drop)

    cols_to_drop = set(col for drops in drop_map.values() for col in drops)
    temp_factors_df = temp_factors_df.drop(columns=cols_to_drop)
    return temp_factors_df, drop_map

def merge_drop_map(drop_map):
    cols_to_drop = set(col for drops in drop_map.values() for col in drops)
    final_drop_map = {}
    for keeper, direct_drops in drop_map.items():
        if keeper not in cols_to_drop:
            cols_to_check = list(direct_drops) 
            all_related_drops = set(direct_drops)
            while cols_to_check:
                col = cols_to_check.pop(0)
                if col in drop_map:
                    new_drops = [d for d in drop_map[col] if d not in all_related_drops]
                    cols_to_check.extend(new_drops)
                    all_related_drops.update(new_drops)
            
            final_drop_map[keeper] = sorted(list(all_related_drops))
    
    return final_drop_map

def run_regressions(distilled_factors):
    results = []
    for symbol in pct_changes.columns:
        etf_excess = pct_changes[symbol] - risk_free_df['daily_nominal_rate']
        data = pd.concat([etf_excess.rename('etf_excess'), distilled_factors], axis=1).dropna()

        Y = data['etf_excess']
        X = sm.add_constant(data.iloc[:, 1:])
        model = sm.OLS(Y, X).fit()
        result = {
            'conId': symbol,
            'nobs': model.nobs,
            'r_squared': model.rsquared,
            'r_squared_adj': model.rsquared_adj,
            'f_statistic': model.fvalue,
            'f_pvalue': model.f_pvalue,
            'aic': model.aic,
            'bic': model.bic,
            'condition_number': model.condition_number,
            'alpha': model.params['const'],
            'alpha_pval': model.pvalues['const'],
            'alpha_tval': model.tvalues['const'],
            'alpha_bse': model.bse['const'],
        }
        for factor in distilled_factors.columns:
            result[f'beta_{factor}'] = model.params[factor]
            result[f'pval_beta_{factor}'] = model.pvalues[factor]
            result[f'tval_beta_{factor}'] = model.tvalues[factor]
            result[f'bse_beta_{factor}'] = model.bse[factor]
        results.append(result)

    results_df = pd.DataFrame(results)
    return results_df

def calculate_vif(df):
    vif_data = pd.DataFrame()
    vif_data["feature"] = df.columns
    vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    return vif_data.sort_values(by='VIF', ascending=False)

def run_elastic_net(factors_df,
                    pct_changes,
                    risk_free_df,
                    training_cutoff,
                    alphas=np.logspace(-4, 1, 50),
                    l1_ratio=[.1, .5, .9],
                    cv=5,
                    tol=5e-4,
                    random_state=42):

    data = data = (
        factors_df.copy()
        .join(pct_changes, how='inner')
        .join(risk_free_df[['daily_nominal_rate']], how='inner')
        .fillna(0)
    )

    train = data[data.index < training_cutoff]
    test = data[data.index >= training_cutoff]

    X_train = train[factors_df.columns].values
    X_test = test[factors_df.columns].values
    
    metrics = []
    for etf in tqdm(pct_changes.columns, total=len(pct_changes.columns), desc="Elastic Net Regression"):
        Y_train = train[etf].values - train['daily_nominal_rate'].values
        Y_test = test[etf].values - test['daily_nominal_rate'].values

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('enet', ElasticNetCV(alphas=alphas,
                                l1_ratio=l1_ratio,
                                cv=cv,
                                random_state=random_state,
                                max_iter=499999,
                                tol=tol,
                                fit_intercept=True,
                                n_jobs=-1)),
        ])

        try:
            pipeline.fit(X_train, Y_train)
        except ValueError as e:
            print(f"Skipping {etf} due to error: {e}")
            continue

        # Unscale coefficients and intercept
        enet = pipeline.named_steps['enet']
        scaler = pipeline.named_steps['scaler']
        betas_train = enet.coef_ / scaler.scale_
        intercept = enet.intercept_ - np.dot(betas_train, scaler.mean_)

        # out-of-sample stats
        er_test = pipeline.predict(X_test)

        # in-sample stats
        er_train = pipeline.predict(X_train)

        row = {
            'conId': etf,
            'jensens_alpha': intercept,
            'enet_alpha': enet.alpha_,
            'l1_ratio': enet.l1_ratio_,
            'n_iter': enet.n_iter_,
            'dual_gap': enet.dual_gap_,
            'n_nonzero': np.sum(np.abs(betas_train) > 1e-6),
            'cv_mse_best': np.min(enet.mse_path_.mean(axis=2)),
            'cv_mse_average': np.mean(enet.mse_path_.mean(axis=2)),
            'cv_mse_worst': np.max(enet.mse_path_.mean(axis=2)),
            'mse_test' : mean_squared_error(Y_test, er_test),
            'mse_train' : mean_squared_error(Y_train, er_train),
            'r2_test' : r2_score(Y_test, er_test),
            'r2_train' : r2_score(Y_train, er_train),
        }

        # Map back coefficients to factor names
        for coef, fname in zip(betas_train, factors_df.columns):
            row[f'{fname}_beta'] = coef

        metrics.append(row)
    
    results_df = pd.DataFrame(metrics).set_index('conId')
    return results_df

def optimize_scalar(series):
    def obj(s):
        if s <= 0:
            return np.inf
        return skew(np.log1p(s * series))**2

    result = minimize_scalar(obj, bounds=(1e-5, 1e20), method='bounded')
    print(result.x)
    return result.x


kind = 'trades'
root = 'data/daily-trades/'
data_path = root + 'series/'
verified_path = root + 'verified_files.csv'
price_col = 'average'
fund_df = load('data/fundamentals.csv')

try:
    verified_df = pd.read_csv(verified_path)
except FileNotFoundError:
    util.startLoop()
    ib = IB()
    ib.connect('127.0.0.1', 7497, clientId=2)

    file_list = os.listdir(data_path)
    verified_files = []

    for file_name in tqdm(file_list, total=len(file_list), desc="Verifying files"):
        if not file_name.endswith('.csv'):
            continue
        try:
            symbol, exchange, currency = file_name.replace('.csv', '').split('-')
            symbol_data = fund_df[(fund_df['symbol'] == symbol) & (fund_df['currency'] == currency)]
            if symbol_data.empty:
                continue

            contract_details = ib.reqContractDetails(Stock(symbol, exchange, currency))
            if not contract_details:
                continue
            isin = contract_details[0].secIdList[0].value

            if symbol_data['isin'].iloc[0] != isin:
                continue

            instrument_name = symbol_data['longName'].iloc[0].replace('-', '').replace('+', '')
            leveraged = any(
                re.fullmatch(r'\d+X', word) and int(word[:-1]) > 1 or word.lower().startswith(('lv', 'lev'))
                for word in instrument_name.split()
            )
            if leveraged:
                continue

            verified_files.append({'symbol': symbol, 'currency': currency})
        except ValueError as e:
            print(f"Invalid filename format {file_name}: {e}")
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    verified_df = pd.DataFrame(verified_files)
    verified_df.to_csv(verified_path, index=False)

    ib.disconnect()


# Merge historical series with fundamentals


# Load historical series
latest = (datetime.now() - timedelta(days=365 * 6))
meta = []
file_list = os.listdir(data_path)
for file in tqdm(file_list, total=len(file_list)):
    if not file.endswith('.csv'):
        continue
    parts = os.path.splitext(file)[0].split('-')
    symbol, exchange, currency = parts[0], parts[1], parts[2]
    if not ((verified_df['symbol'] == symbol) & (verified_df['currency'] == currency)).any():
        continue
    try:
        df = load(data_path + file)
        df = ensure_series_types(df, price_col)
        df = validate_raw_prices(df, price_col)
        df = handle_stale_periods(df, price_col)
        df['pct_change'] = df[price_col].pct_change()
        if df['date'].max() > latest:
            latest = df['date'].max()
        meta.append({
            'symbol': symbol,
            'currency': currency,
            'exchange_api': exchange,
            'df': df[['date', price_col, 'volume', 'pct_change']],
        })
    except Exception as e:
        print(f"ERROR {file}: {e}")
meta = pd.DataFrame(meta)
copied = meta.copy()
copied['df'] = copied['df'].apply(lambda x: x.copy())


# Detect pct_change outliers


detect_and_nullify_global_outliers(meta, price_col=price_col, z_threshold=50, window=5)



# RESET
meta = copied.copy()
meta['df'] = copied['df'].apply(lambda x: x.copy())

# Calculate series gap stats
while True:
    year_range = int(input('Year range (>= 2):'))
    if year_range >= 2:
        break

oldest = latest - pd.Timedelta(days=365 * year_range)
business_days = pd.date_range(start=oldest, end=latest, freq='B')

for idx, row in tqdm(meta.iterrows(), total=len(meta)):
    df = row['df']
    merged = pd.DataFrame({'date': business_days}).merge(df, on='date', how='left')
    present = merged[price_col].notna()
    present_idx = np.flatnonzero(present)
    gaps = []
    length = len(merged)
    if present_idx.size > 0:
        if present_idx[0] > 0:
            gaps.append(present_idx[0])
        if present_idx.size > 1:
            internal_gaps = np.diff(present_idx) - 1
            gaps.extend(gap for gap in internal_gaps if gap > 0)
        if present_idx[-1] < length - 1:
            gaps.append(length - 1 - present_idx[-1])
    else:
        gaps = [length]
    gaps = np.array(gaps, dtype=int)
    gaps = gaps[gaps > 0]
    max_gap = float(gaps.max()) if gaps.size > 0 else 0.0
    std_gap = float(gaps.std()) if gaps.size > 0 else 0.0
    missing = length - present.sum()
    pct_missing = missing / length
    meta.at[idx, 'df'] = merged
    meta.at[idx, 'max_gap'] = max_gap
    meta.at[idx, 'missing'] = missing
    meta.at[idx, 'pct_missing'] = pct_missing

# Remove series with large day gaps
meta['max_gap_log'] = np.log1p(meta['max_gap'])
meta['max_gap_log'] = meta['max_gap_log'] / meta['max_gap_log'].max()

## 3y window mean stats
max_gap_log = 0.46
max_pct_missing = 0.3

condition = ((meta['max_gap_log'] < max_gap_log) & \
             (meta['pct_missing'] < max_pct_missing))
filtered = meta[condition].copy()
del meta

# Interpolate/extrapolate price column and merge with fund
for idx, row in tqdm(filtered.iterrows(), total=len(filtered)):
    df = row['df']
    df[price_col] = df[price_col].interpolate(method='akima', limit_direction='both')
    if df[price_col].isna().any():
        df[price_col] = df[price_col].ffill()
        df[price_col] = df[price_col].bfill()
    df['pct_change'] = df[price_col].pct_change()
    filtered.at[idx, 'df'] = df.set_index('date')

filtered = pd.merge(filtered, fund_df, on=['symbol', 'currency'], how='inner').drop(['max_gap', 'missing', 'pct_missing', 'max_gap_log'], axis=1)

# Risk-free series calculation
tickers = {
    'US': 'DTB3',
    'Canada': 'IR3TIB01CAM156N',
    'Germany': 'IR3TIB01DEM156N',
    'UK': 'IR3TIB01GBM156N',
    'France': 'IR3TIB01FRA156N',
}
## Fetch each series and convert from percentage to decimal
bonds = {}
failed = []
for country, ticker in tickers.items():
    try:
        series = web.DataReader(ticker, 'fred', oldest, latest)
        bonds[country] = series / 100.0
    except Exception:
        try:
            series = web.DataReader(ticker, 'oecd', oldest, latest)
            bonds[country] = series / 100.0
        except Exception as oecd_err:
            failed.append(country)

# Combine into a single DataFrame
df_bonds = pd.concat(bonds, axis=1)
df_bonds.columns = [c for c in tickers if c not in failed]
df_bonds = df_bonds.interpolate(method='akima').bfill().ffill()

risk_free_df = df_bonds.mean(axis=1).rename('nominal_rate')
risk_free_df = risk_free_df.reindex(business_days, copy=False)

risk_free_df = pd.DataFrame(risk_free_df)
risk_free_df['daily_nominal_rate'] = risk_free_df['nominal_rate'] / 252


# Add pct_change cols to dfs and create pct_changes
numerical_cols = [col for col in filtered.columns if filtered[col].dtype in [np.int64, np.float64] and col not in ['conId']]

pct_changes = pd.concat(
        [row['df']['pct_change'].rename(row['conId']) 
        for _, row in filtered.iterrows()], axis=1
    )

# Remove uninformative cols for market portfolios 
uninformative_cols = [col for col in numerical_cols if filtered[col].nunique(dropna=True) <= 1]
filtered = filtered.drop(columns=uninformative_cols)
filtered = filtered.dropna(axis=1, how='all')

# Add rate of change fundamentals


rate_fundamentals = [('EPSGrowth-1yr', 'EPS_growth_3yr', 'EPS_growth_5yr'),
                     ('ReturnonAssets1Yr', 'ReturnonAssets3Yr'),
                     ('ReturnonCapital', 'ReturnonCapital3Yr'),
                     ('ReturnonEquity1Yr', 'ReturnonEquity3Yr'),
                     ('ReturnonInvestment1Yr', 'ReturnonInvestment3Yr')]

for cols in rate_fundamentals:
    base_name = cols[0].replace('-1yr', '').replace('1Yr', '')
    slope_col = f'fundamentals_{base_name}_slope'
    
    if len(cols) == 3:
        col_1yr, col_3yr, col_5yr = cols
        filtered[slope_col] = calculate_slope(filtered[f'fundamentals_{col_1yr}'], filtered[f'fundamentals_{col_5yr}'], 1, 5)

        slope_1yr_3yr = calculate_slope(filtered[f'fundamentals_{col_1yr}'], filtered[f'fundamentals_{col_3yr}'], 1, 3)
        slope_3yr_5yr = calculate_slope(filtered[f'fundamentals_{col_3yr}'], filtered[f'fundamentals_{col_5yr}'], 3, 5)
        filtered[f'fundamentals_{base_name}_second_deriv'] = calculate_slope(slope_1yr_3yr, slope_3yr_5yr, 1, 3)

    elif len(cols) == 2:
        col_1yr, col_3yr = cols
        filtered[slope_col] = calculate_slope(filtered[f'fundamentals_{col_1yr}'], filtered[f'fundamentals_{col_3yr}'], 1, 3)

## Add new cols to numericals
numerical_cols = [col for col in filtered.columns if filtered[col].dtype in [np.int64, np.float64] and col not in ['conId']]


# Return stats and split training and tests sets


training_cutoff = latest - pd.Timedelta(days=365)
momentum_cutoffs = {
    '1y':  training_cutoff - pd.Timedelta(days=365),
    '6mo': training_cutoff - pd.Timedelta(days=365 // 2),
    '3mo': training_cutoff - pd.Timedelta(days=365 // 4),
}

## Apply to each row
filtered[['momentum_3mo', 'momentum_6mo', 'momentum_1y', 'stats_sharpe']] = filtered['df'].apply(lambda df: get_return_stats(df, training_cutoff, momentum_cutoffs, risk_free_df))

# Create all asset type indices/portfolios
holding_cols = [col for col in filtered.columns if col.startswith('holding_') and col != 'holding_types_variety'] + ['total']
portfolio_dfs = {}

for holding_col in holding_cols:
    name = holding_col.split('_')[-1]
    if holding_col == 'total':
        weight = filtered['profile_cap_usd']
    else:
        weight = (filtered['profile_cap_usd'] * filtered[holding_col])
 
    total_market_cap = (weight).sum()
    filtered['weight'] = weight / total_market_cap
    
    weights = filtered.set_index('conId')['weight']
    portfolio_return = pct_changes.dot(weights)
    initial_price = 1
    portfolio_price = initial_price * (1 + portfolio_return.fillna(0)).cumprod()

    portfolio_df = pd.DataFrame({
        'date': portfolio_price.index,
        price_col: portfolio_price.values,
        'pct_change': portfolio_return.values
    }).set_index('date')

    portfolio_dfs[name] = portfolio_df

filtered.drop('weight', axis=1, inplace=True)

# Avoid dummy trap
empty_subcategories = {
'holding_types': ['other'],
'countries': ['Unidentified'], 
'currencies': ['<NoCurrency>'],
'industries': ['NonClassifiedEquity', 'NotClassified-NonEquity'],
'top10': ['OtherAssets', 'AccountsPayable','AccountsReceivable','AccountsReceivable&Pay','AdministrationFees','CustodyFees','ManagementFees','OtherAssetsandLiabilities','OtherAssetslessLiabilities', 'OtherFees','OtherLiabilities','Tax','Tax--ManagementFees'],
'debtors': ['OTHER'],
'maturity': ['%MaturityOther'],
'debt_type': ['%QualityNotAvailable', '%QualityNotRated'],
'manual': ['asset_other']
}

dummy_trap_cols = []
for k, lst in empty_subcategories.items():
    for i in lst:
        if k == 'manual':
            dummy_trap_cols.append(i)
        else:
            dummy_trap_cols.append(f'{k}_{i}')
    
filtered = filtered.drop(columns=dummy_trap_cols, axis=1, errors='ignore')
numerical_cols = [col for col in filtered.columns if filtered[col].dtype in [np.int64, np.float64] and col not in ['conId']]

# Select asset types to work on
asset_conditions = {
    'equity': (filtered['asset_equity'] == 1),
    'cash': (filtered['asset_cash'] == 1),
    'bond': (filtered['asset_bond'] == 1),
    'other': (filtered['asset_equity'] == 0) & (filtered['asset_cash'] == 0) & (filtered['asset_bond'] == 0),
}

exclude_assets = ['bond']
asset_classes = list(asset_conditions.keys())

include_assets = [asset for asset in asset_classes if asset not in exclude_assets]
combined_condition = pd.Series(False, index=filtered.index)
for asset in include_assets:
    combined_condition |= asset_conditions[asset]

filtered_df = filtered[combined_condition]
numerical_cols = [col for col in filtered_df.columns if filtered_df[col].dtype in [np.int64, np.float64] and col not in ['conId']]

single_value_columns = [col for col in filtered_df.columns if col in numerical_cols and filtered_df[col].nunique() == 1]
asset_cols = [col for col in filtered_df if col.startswith('asset')]
filtered_df = filtered_df.drop(columns=single_value_columns + asset_cols)
numerical_cols = [col for col in filtered_df.columns if filtered_df[col].dtype in [np.int64, np.float64] and col not in ['conId']]

pct_changes = pct_changes[filtered_df['conId']]


# Rename to standard country strings
cc = coco.CountryConverter()
country_cols = [col for col in filtered_df.columns if col.startswith('countries') and not col.endswith('variety')]

standard_names = set()
rename_map = {}
for col in country_cols:
    if col == 'countries_Unidentified':
        continue

    raw_name = col.replace('countries_', '')
    raw_name = ''.join(raw_name.split(' '))
    raw_name = ''.join([' ' + char if char.isupper() and i > 0 else char for i, char in enumerate(raw_name)]).strip()

    standard_name = cc.convert(names=raw_name, to='ISO3', not_found=None)
    standard_names.add(standard_name)
    if standard_name:
        rename_map[col] = f'countries_{standard_name}'
    else:
        print(f"Could not standardize: '{raw_name}' (from column '{col}')")

filtered_df.rename(columns=rename_map, inplace=True)


# Create new country columns
indicator_name_map = {
    'NY.GDP.PCAP.CD': 'gdp_pcap',
    'SP.POP.TOTL': 'population',
}

continent_map = create_continent_map(standard_names)
metric_df = create_metric_maps(standard_names, indicator_name_map.keys(), oldest, latest)

metric_suffixes = {
    'raw_value': '_value',
    '1st_div': '_growth',
    '2nd_div': '_acceleration'
}
continents = list(continent_map.values())

for cont in continents:
    filtered_df[f'continent_{cont}'] = 0.0
for ind_code, ind_name in indicator_name_map.items():
    if ind_code in metric_df.columns.get_level_values(0):
        for metric_col, suffix in metric_suffixes.items():
            new_col_name = f'{ind_name}{suffix}'
            filtered_df[new_col_name] = 0.0

for std_name in standard_names:
    country_weight_col = f'countries_{std_name}'
    if country_weight_col not in filtered_df.columns:
        continue
    if std_name in continent_map:
        continent = continent_map[std_name]
        filtered_df[f'continent_{continent}'] += filtered_df[country_weight_col]        
    # if std_name in msci_map:
    #     market_group = msci_map[std_name]
    #     filtered_df[f'msci_{market_group}'] += filtered_df[country_weight_col]

    if std_name in metric_df.index:
        for ind_code, ind_name in indicator_name_map.items():
            if ind_code in metric_df.columns.get_level_values(0):
                for metric_col, suffix in metric_suffixes.items():
                    value = metric_df.loc[std_name, (ind_code, metric_col)]
                    target_col = f'{ind_name}{suffix}'
                    filtered_df[target_col] += filtered_df[country_weight_col] * value

# Drop single unique value columns
numerical_cols = [col for col in filtered_df.columns if filtered_df[col].dtype in [np.int64, np.float64] and col not in ['conId']]
single_value_columns = [col for col in numerical_cols if filtered_df[col].nunique() == 1]
filtered_df = filtered_df.drop(columns=single_value_columns, errors='ignore')
numerical_cols = [col for col in filtered_df.columns if filtered_df[col].dtype in [np.int64, np.float64] and col not in ['conId']]


# Fundamentals reduced to factor columns
fundamental_columns = [col for col in filtered_df.columns if col.startswith('fundamentals')]

value_columns_inverted = [
    'fundamentals_Price/Book',
    'fundamentals_Price/Cash',
    'fundamentals_Price/Earnings',
    'fundamentals_Price/Sales',
]
investment_columns = [
    'fundamentals_EPSGrowth-1yr',
    'fundamentals_EPS_growth_3yr',
    'fundamentals_EPS_growth_5yr',
    'fundamentals_EPSGrowth_slope',
    'fundamentals_EPSGrowth_second_deriv',
    'fundamentals_ReturnonAssets_slope',
    'fundamentals_ReturnonCapital_slope',
    'fundamentals_ReturnonEquity_slope',
    'fundamentals_ReturnonInvestment_slope',
]
leverage_columns_inverted = [
    'fundamentals_LTDebt/Shareholders',
    'fundamentals_TotalDebt/TotalCapital',
    'fundamentals_TotalDebt/TotalEquity',
    'fundamentals_TotalAssets/TotalEquity',
]
profitability_columns = [
    'fundamentals_ReturnonAssets1Yr',
    'fundamentals_ReturnonAssets3Yr', #
    'fundamentals_ReturnonCapital',
    'fundamentals_ReturnonCapital3Yr', #
    'fundamentals_ReturnonEquity1Yr',
    'fundamentals_ReturnonEquity3Yr', #
    'fundamentals_ReturnonInvestment1Yr',
    'fundamentals_ReturnonInvestment3Yr', #
    'fundamentals_SalestoTotalAssets',
    'fundamentals_EBITtoInterest',
]
momentum_columns = [
    'momentum_3mo',
    'momentum_6mo',
    'momentum_1y',
    'fundamentals_RelativeStrength'
]

columns_to_scale = value_columns_inverted + investment_columns + leverage_columns_inverted + profitability_columns + momentum_columns

if any(x in filtered_df.columns for x in columns_to_scale):
    scaler = MinMaxScaler()
    filtered_df[columns_to_scale] = scaler.fit_transform(filtered_df[columns_to_scale])

    # Value Score
    filtered_df['factor_investment'] = filtered_df[investment_columns].sum(axis=1)
    filtered_df['factor_value'] = (1 - filtered_df[value_columns_inverted]).sum(axis=1)
    filtered_df['factor_leverage'] = (1 - filtered_df[leverage_columns_inverted]).sum(axis=1)
    filtered_df['factor_profitability'] = filtered_df[profitability_columns].sum(axis=1)
    filtered_df['factor_momentum'] = filtered_df[momentum_columns].sum(axis=1)

    filtered_df = filtered_df.drop(columns=columns_to_scale, errors='ignore')


# Reorganize columns
categories = ['factor', 'holding_types', 'stats', 'momentum', 'profile', 'top10', 'population', 'msci', 'gdp', 'continent', 'countries', 'fundamentals', 'industries', 'currencies', 'debtors', 'maturity', 'debt_type', 'lipper', 'dividends', 'marketcap', 'style', 'domicile', 'asset']

numerical_cols = [col for col in filtered_df.columns if filtered_df[col].dtype in [np.int64, np.float64] and col not in ['conId']]
non_numerical = [col for col in filtered_df.columns if col not in numerical_cols]

for category in reversed(categories):
    cat_cols = [col for col in numerical_cols if col.startswith(category)]
    remaining = [col for col in numerical_cols if col not in cat_cols]
    numerical_cols = cat_cols + remaining

new_column_order = non_numerical + numerical_cols
filtered_df = filtered_df[new_column_order]


factors_df, diffs = construct_factors(filtered_df, pct_changes, portfolio_dfs, risk_free_df, scaling_factor=0.6, diffs=True)

# Custom drop
low_absolute_beta = ['profile_cap_usd', 'holding_types_equity', 'industries_BasicMaterials']
frequently_lassoed = ['gdp_pcap_growth', 'gdp_pcap_acceleration', 'continent_Africa', 'population_value']
custom_drop = low_absolute_beta + frequently_lassoed
custom_drop = [c.split('_beta')[0] for c in custom_drop]
factors_df = factors_df.drop(columns=custom_drop, errors='ignore')

# Screen factors
distilled_factors, drop_map = prescreen_factors(factors_df, correlation_threshold=0.95)

# corr_matrix = distilled_factors.corr()
# vif_df = calculate_vif(distilled_factors.dropna(axis=0))
# highest_vif = vif_df['VIF'].iloc[0]
# if distilled_factors.shape[1] > 2:
#     to_drop = vif_df['feature'].iloc[0]
#     distilled_factors.drop(columns=[to_drop], inplace=True)

# np.fill_diagonal(corr_matrix.values, 0)
# keeper = corr_matrix[to_drop].sort_values(ascending=False).index[0]
# drop_map.setdefault(keeper, []).append(to_drop)

drop_map = merge_drop_map(drop_map)


# ElasticNet regression
results_df = run_elastic_net(
    factors_df=distilled_factors,
    pct_changes=pct_changes,
    risk_free_df=risk_free_df,
    training_cutoff=training_cutoff,
    alphas=np.logspace(-11, -4, 30),
    l1_ratio=[0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1],
    cv=5,
    tol=5e-4,
)

# Factor beta post-screening score 


beta_cols = [col for col in results_df if col.endswith('beta')]
screening_df = pd.DataFrame(index=results_df.index)

screening_df['r2_test'] = results_df['r2_test'].max() - results_df['r2_test']
screening_df['r2_test'] = np.log1p(screening_df['r2_test'] * optimize_scalar(screening_df['r2_test']))
screening_df['r2_test'] = screening_df['r2_test'].max() - screening_df['r2_test']

screening_df['cv_mse_std'] = results_df[['cv_mse_best','cv_mse_average','cv_mse_worst']].std(axis=1)
screening_df['cv_mse_std'] = np.log1p(screening_df['cv_mse_std'] * optimize_scalar(screening_df['cv_mse_std']))

screening_df['screening_score'] = screening_df['r2_test'] / (1 + screening_df['cv_mse_std'])

screening_df = results_df[beta_cols].abs().multiply(screening_df['screening_score'], axis=0).mean()

# Lasso count post-screening
good_models_df = results_df[results_df['r2_test'] > 0.05]

if not good_models_df.empty:
    good_model_betas = good_models_df[beta_cols]
    non_zero = (good_model_betas.abs() > 1e-6).sum()
    good_factor_importance = pd.DataFrame({
        'non_zero_percentage': non_zero / len(results_df),
        'mean_abs_beta': good_model_betas.abs().mean()
    }).sort_values('mean_abs_beta', ascending=True)

    good_factor_importance['mean'] = good_factor_importance.mean(axis=1)
    display(good_factor_importance.sort_values(by='non_zero_percentage'))