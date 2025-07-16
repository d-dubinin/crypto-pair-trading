import pandas as pd
import numpy as np
import time
import ccxt
import requests
from itertools import combinations

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

def compute_turnover(port):
    to = (port.fillna(0)-port.shift().fillna(0)).abs().sum(1)   
    return to

def spread_formation(in_sample_df: pd.DataFrame,pairs,window=90):

    data_dict = {}  # Dictionary to store calculated spread, centered spread, and beta for each pair
    for pair in pairs:  # Loop over each pair of symbols

        symbol_i = pair[0]  # First symbol in the pair
        symbol_j = pair[1]  # Second symbol in the pair

        px_i = in_sample_df[symbol_i]  # Price series of symbol_i
        px_j = in_sample_df[symbol_j]  # Price series of symbol_j

        log_px_i = np.log(px_i)  # Log-price of symbol_i
        log_px_j = np.log(px_j)  # Log-price of symbol_j

        # Rolling covariance and variance to compute rolling beta
        rolling_cov = log_px_i.rolling(window=window).cov(log_px_j)
        rolling_var = log_px_i.rolling(window=window).var()

        beta = rolling_cov / rolling_var  # Rolling beta (slope of linear relationship)
        alpha = log_px_j.rolling(window=window).mean() - beta * log_px_i.rolling(window=window).mean()  # Rolling alpha (intercept)

        # Calculate spread: difference between log_px_i and the linear combination of alpha and beta * log_px_j
        spread = log_px_i - (alpha + beta * log_px_j)

        # Rolling mean of spread
        spread_mean = spread.rolling(window=window).mean()
        centered_spread = (spread - spread_mean)  # Centered spread

        # Store the calculated series in the dictionary with tuple keys
        data_dict[(tuple(pair), 'spread')] = spread
        data_dict[(tuple(pair), 'centered_spread')] = centered_spread
        data_dict[(tuple(pair), 'beta')] = beta

    data = pd.DataFrame(data_dict)  # Create a DataFrame from the dictionary

    return data  # Return the resulting DataFrame

def drawdown_series(rets_series):
    running_max = rets_series.cummax()
    drawdown = rets_series - running_max
    drawdown_pct = drawdown / running_max

    return drawdown_pct

def sharpe_ratio(rets_series,rate=0):

    return (rets_series.mean() - rate)/rets_series.std()

def strategy_metrics(benchmark_rets_series, strat_rets_series):
    benchmark_cum_returns = (1+benchmark_rets_series).cumprod()
    strat_cum_returns = (1+strat_rets_series).cumprod()

    metrics_dict = {
    'BTC buy & hold': {'Cumulative return': (benchmark_cum_returns.iloc[-1]-1)*100,
                       'Annualized return': (((benchmark_cum_returns.iloc[-1])**(365/len(benchmark_cum_returns)))-1)*100,
                       'Max drawdown (%)': drawdown_series(benchmark_cum_returns).min()*100,
                       'Annualized Sharpe Ratio': sharpe_ratio(benchmark_rets_series)*np.sqrt(365),
                       'Annualized std': benchmark_rets_series.std()*np.sqrt(365),
                       },
    'Strategy': {'Cumulative return': (strat_cum_returns.iloc[-1]-1)*100,
                 'Annualized return': (((strat_cum_returns.iloc[-1])**(365/len(strat_cum_returns)))-1)*100,
                 'Max drawdown (%)': drawdown_series(strat_cum_returns).min()*100,
                 'Annualized Sharpe Ratio': sharpe_ratio(strat_rets_series)*np.sqrt(365),
                 'Annualized std': strat_rets_series.std()*np.sqrt(365),
                 }
    }

    return pd.DataFrame(metrics_dict)

def close_df(symbols,since,exchange = ccxt.binance({'enableRateLimit': True})): 
    df = pd.DataFrame()
    # First check dates:
    series_dict = {}
    for ticker in symbols:
        symbol = f'{ticker}/USDT:USDT'
        timeframe = '1d'
        limit = 500

        all_candles_perpet = []

        since_temp = since

        while True:
            candles = exchange.fetch_ohlcv(symbol, timeframe, since_temp, limit)
            if not candles:
                break
            all_candles_perpet += candles
            since_temp = candles[-1][0] + 1  # move to the next candle after last timestamp
            time.sleep(exchange.rateLimit / 500)  # avoid rate limits (to be sure we divide by 500 and not 1000) -> rate limit is is ms
        time_and_close = np.array(all_candles_perpet)[:,[0,4]]
        df = pd.DataFrame(time_and_close, columns = ['timestamp',ticker])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        series_dict[ticker] = df
    return pd.concat(series_dict.values(), axis=1)

def market_cap_coingecko():
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        'vs_currency': 'usd',
        'order': 'market_cap_desc',
        'per_page': 100,
        'page': 1,
    }

    response = requests.get(url, params=params)
    data = response.json()
    return data

def is_ticker_available(symbol,exchange = ccxt.binance({'enableRateLimit': True})):
    try:
        # Try to fetch the ticker
        exchange.fetch_ticker(symbol)
        return True
    except ccxt.BaseError:
        return False

def get_available_tickers(exchange = ccxt.binance({'enableRateLimit': True})):
    data = market_cap_coingecko()

    # Remove stablecoins
    tickers = [coin['symbol'].upper() for coin in data if coin['symbol'].upper() not in ['USDT', 'USDC', 'USDS', 'USDE', 'DAI', 'SUSDS', 'USD1', 'FDUSD', 'PYUSD','USDT0','SUSDE','BSC-USD','STETH','WBTC','WSTETH','BCH','DOGE','SHIB', 'PEPE','ICP']]

    # Check if perpetual ticker available in the exchange

    mask = [is_ticker_available(ticker+'/USDT:USDT',exchange) for ticker in tickers]

    available_tickers = [ticker for ticker, m in zip(tickers, mask) if m]

    return available_tickers

def adf_for_pair(symbol_i:str, symbol_j:str, in_sample_df: pd.DataFrame):
    # OLS regression
    Y = np.log(in_sample_df[symbol_i])
    X = np.log(in_sample_df[symbol_j])
    model = sm.OLS(Y,sm.add_constant(X))
    results = model.fit()

    alpha = results.params.iloc[0]
    beta = results.params.iloc[1]

    spread = Y - (alpha + beta * X)

    # ADF test
    adf_test = adfuller(spread)
    test_statistic = adf_test[0]
    p_val = adf_test[1]
    r_squared = results.rsquared

    return p_val, test_statistic, r_squared

def cointegration (in_sample_df, threshold=0.05):
    new_pairs = []
    df = pd.DataFrame(columns=["coin_1","coin_2","p_value", "test_stat"])
    tickers = []
    pairs = list(combinations(in_sample_df.columns.tolist(), 2))

    for pair in pairs:
        p_val_1, test_statistic_1, r_squared_1 = adf_for_pair(pair[0], pair[1], in_sample_df)
        p_val_2, test_statistic_2, r_squared_2 = adf_for_pair(pair[1], pair[0], in_sample_df)

        if r_squared_1 >= r_squared_2:
            p_val = p_val_1
            test_statistic = test_statistic_1
        else:
            p_val = p_val_2
            test_statistic = test_statistic_2
            pair = (pair[1],pair[0])

        if p_val < 0.05:
            new_pairs.append(pair)
            new_row = pd.DataFrame({'coin_1': [pair[0]], 'coin_2': [pair[1]], 'p_value': [p_val], 'test_stat': [test_statistic]})
            df = pd.concat([df, new_row], ignore_index=True)

    return new_pairs, df

def generate_weights(signals_df,inst, threshold=0.5):
    # Initialize a DataFrame with the same index and columns as crypto_px, filled with NaN

    nb_trades = 0
    pairs = signals_df.columns.get_level_values(0).unique().tolist()


    pos = pd.DataFrame(index=signals_df.index, columns=inst,dtype=float)

    tickers = [ticker for pair in pairs for ticker in pair]
   
    #z_scores = signals_df[(tuple(pair), 'z_score')]

    for pair in pairs:
        asset_i = pair[0]
        asset_j = pair[1]
        betas = signals_df[(tuple(pair), 'beta')]
        # If negative beta, replace by recent positive
        #betas[betas<=0] = np.nan
        #betas.ffill(inplace=True)
        z_scores = signals_df[(tuple(pair), 'centered_spread')]
        fast_sma = z_scores.rolling(8).mean()
        slow_sma = z_scores.rolling(16).mean()
        #std_i = signals_df[(tuple(pair), 'std_i')]
        #std_j = signals_df[(tuple(pair), 'std_j')]
        diff = fast_sma.diff()

        short_pos = 0
        long_pos = 0

        long_trigger = False
        short_trigger = False

        
        for ind in pos.index:


            if long_pos == 1:
                pos.loc[ind, asset_j] = -betas.loc[ind]#*(1/std_j.loc[ind])
                pos.loc[ind,asset_i] = 1#*(1/std_i.loc[ind])
                nb_trades = nb_trades+1
            elif short_pos == 1:
                pos.loc[ind, asset_j] = betas.loc[ind]#*(1/std_j.loc[ind])
                pos.loc[ind,asset_i] = -1#*(1/std_i.loc[ind])
                nb_trades = nb_trades+1

            if (z_scores.loc[ind] > 0) and (z_scores.shift().loc[ind]<= 0):
                short_trigger = True
                long_trigger = False

            elif (z_scores.loc[ind]) < 0 and (z_scores.shift().loc[ind]>= 0):
                long_trigger = True
                short_trigger = False

            if (short_trigger == True) and (fast_sma.loc[ind]<slow_sma.loc[ind]) and (short_pos == 0) and (np.abs(z_scores.loc[ind]) > threshold):
            #if (short_trigger == True) and (diff.loc[ind]<0) and (short_pos == 0) and (np.abs(z_scores.loc[ind]) > threshold):
                pos.loc[ind,asset_i] = -1 #*(1/std_i.loc[ind])
                pos.loc[ind, asset_j] = betas.loc[ind] #*(1/std_j.loc[ind])
                short_pos = 1
                short_trigger = False

            elif (long_trigger == True) and (fast_sma.loc[ind]>slow_sma.loc[ind]) and (long_pos == 0) and (np.abs(z_scores.loc[ind]) > threshold):
            #elif (long_trigger == True) and (diff.loc[ind]>0) and (long_pos == 0) and (np.abs(z_scores.loc[ind]) > threshold):
                pos.loc[ind, asset_i] = 1 #*(1/std_i.loc[ind])
                pos.loc[ind, asset_j] = -betas.loc[ind]#*(1/std_j.loc[ind])
                long_pos = 1
                long_trigger = False

            if (short_pos == 1) and (np.abs(z_scores.loc[ind]) <= threshold):
                pos.loc[ind, asset_i] = 0 
                pos.loc[ind, asset_j] = 0
                short_pos = 0
                nb_trades = nb_trades+1
            


            elif (long_pos == 1) and (np.abs(z_scores.loc[ind]) <= threshold):
                pos.loc[ind, asset_j] = 0  
                pos.loc[ind, asset_i] = 0
                long_pos = 0
                nb_trades = nb_trades+1
                
        #pos[[asset_i,asset_j]] = pos[[asset_i,asset_j]].divide(pos[[asset_i,asset_j]].abs().sum(axis=1), axis=0)/2
            

    # Forward-fill missing values
    pos = pos.ffill()
    # Normalization to get fully invested portfolio and divide by number of pairs to not be concerated if only one pair is active
    pos_final = (pos.divide(pos.abs().sum(axis=1), axis=0).fillna(0))
    print(nb_trades)
    return pos_final


def filter(df,tickers):
    keep_indices = []
    for ticker in unique_transitions:
        keep_indices.append(df[(df['coin_1'] == ticker) | (df['coin_2'] == ticker)].sort_values(by='test_stat').index[0])

    sorted_df = df.sort_values(by=['coin_1','test_stat']).drop_duplicates(subset=['coin_1'])
    sorted_df = sorted_df.sort_values(by=['coin_2','test_stat']).drop_duplicates(subset=['coin_2'])
    sorted_df = sorted_df[sorted_df.index.isin(set(keep_indices))]
    sorted_df.reset_index(drop=True,inplace=True)
    
    return sorted_df