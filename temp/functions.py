import pandas as pd
import numpy as np
import time
import ccxt
import requests
from itertools import combinations

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

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

def is_ticker_available(symbol,exchange = ccxt.binance({'enableRateLimit': True})):
    try:
        # Try to fetch the ticker
        exchange.fetch_ticker(symbol)
        return True
    except ccxt.BaseError:
        return False

def spread_formation(in_sample_df: pd.DataFrame,pairs,window=90):

    data_dict = {}  # Dictionary to store calculated spread, centered spread, and beta for each pair
    for pair in pairs:  # Loop over each pair of symbols

        symbol_i = pair[0]
        symbol_j = pair[1]

        px_i = in_sample_df[symbol_i]
        px_j = in_sample_df[symbol_j]

        log_px_i = np.log(px_i)
        log_px_j = np.log(px_j) 

        # Rolling covariance and variance to compute rolling beta
        rolling_cov = log_px_i.rolling(window=window,min_periods=window).cov(log_px_j)
        rolling_var = log_px_i.rolling(window=window,min_periods=window).var()

        beta = rolling_cov / rolling_var
        alpha = log_px_j.rolling(window=window,min_periods=window).mean() - beta * log_px_i.rolling(window=window,min_periods=window).mean()  # Rolling alpha (intercept)

        spread = log_px_j - (alpha + beta * log_px_i)

        # Classic z-score
        zs = (spread-spread.rolling(window,min_periods=window).mean())/spread.rolling(window,min_periods=window).std()

        # Robust z-score
        median_roll = spread.rolling(window=window, min_periods=window).median()
        mad_roll = spread.rolling(window=window, min_periods=window).apply(lambda x: np.median(np.abs(x - np.median(x))), raw=True)
        rzs = (spread - median_roll) / (mad_roll)

        data_dict[(tuple(pair), 'spread')] = spread
        data_dict[(tuple(pair), 'zs')] = zs
        data_dict[(tuple(pair), 'rzs')] = rzs
        data_dict[(tuple(pair), 'log_i')] = log_px_i
        data_dict[(tuple(pair), 'log_j')] = log_px_j
        data_dict[(tuple(pair), 'alpha')] = alpha
        data_dict[(tuple(pair), 'beta')] = beta

    data = pd.DataFrame(data_dict) 

    return data 


def generate_pair_trading_signals(signal_df, pairs, bounds, 
                                  sl_percent=0.03, 
                                  starting_capital=1_000_000, 
                                  smoothing_span=6, 
                                  spread_window=90):
    """
    Generate trading signals for pairs trading strategy with stop-loss and mean-reversion exit.

    Parameters
    ----------
    signal_df : pd.DataFrame
        MultiIndex dataframe with fields ["zs", "log_i", "log_j", "alpha", "beta", "spread"] per pair.
    pairs : list
        List of trading pairs to process.
    bounds : dict
        Dict mapping pair -> (upper_bound, lower_bound) for z-score entry conditions.
    sl_percent : float, optional
        Stop-loss percentage of entry notional (default = 0.03, i.e. 3%).
    starting_capital : float, optional
        Capital allocation per pair (default = 1,000,000).
    smoothing_span : int, optional
        Span for exponential moving average smoothing of z-score.
    spread_window : int, optional
        Rolling window length for spread mean/std calculations.

    Returns
    -------
    pd.DataFrame
        DataFrame of signals for all given pairs.
    """

    all_signals = {}

    for pair in pairs:
        z      = signal_df[(pair, "zs")]
        log_i  = signal_df[(pair, "log_i")]
        log_j  = signal_df[(pair, "log_j")]
        alpha  = signal_df[(pair, "alpha")]
        beta   = signal_df[(pair, "beta")]
        spread = signal_df[(pair, "spread")]

        signal_df[(pair, "modified_spread")] = spread.copy()
        signal_df[(pair, "modified_zs")] = z.copy()

        i_px = np.exp(log_i)
        j_px = np.exp(log_j)

        # Smoothed z-scores, turning points
        z_smooth = z.ewm(span=smoothing_span, adjust=False).mean()
        d_ma = z_smooth.diff()
        turning_points = (d_ma.shift(1) * d_ma < 0)

        signals = pd.Series(0, index=signal_df.index, dtype="int8")
        state = None

        # Freeze-at-entry fields
        entry_alpha = entry_beta = None
        entry_i_price = entry_j_price = None
        entry_shares_i = entry_shares_j = 0.0
        entry_notional = 0.0
        entry_spread_mean = entry_spread_std = entry_spread = None

        for i in range(len(signals)):
            idx = signals.index[i]
            cur_z = z.iloc[i]

            if pd.isna(cur_z) or pd.isna(beta.iloc[i]):
                continue

            # Entry arming
            if state is None:
                if cur_z > bounds[pair][0]:
                    state = "pos_waiting"
                elif cur_z < bounds[pair][1]:
                    state = "neg_waiting"

            # Short entry
            elif state == "pos_waiting":
                if bool(turning_points.iloc[i]) and cur_z > 1.0:
                    entry_spread_mean = spread.rolling(window=spread_window).mean().iloc[i]
                    entry_spread_std  = spread.rolling(window=spread_window).std().iloc[i]
                    entry_alpha = alpha.iloc[i]
                    entry_beta  = beta.iloc[i]
                    entry_i_price = i_px.iloc[i]
                    entry_j_price = j_px.iloc[i]

                    total_beta = abs(entry_beta) + 1.0
                    notional_i = starting_capital * (abs(entry_beta) / total_beta)
                    notional_j = starting_capital * (1.0 / total_beta)

                    entry_shares_i = notional_i / entry_i_price
                    entry_shares_j = -notional_j / entry_j_price
                    entry_notional = notional_i + notional_j

                    entry_spread = log_j.iloc[i] - (entry_alpha + entry_beta * log_i.iloc[i])

                    signals.iloc[i] = -1
                    state = "short"

            # Long entry
            elif state == "neg_waiting":
                if bool(turning_points.iloc[i]) and cur_z < -1.0:
                    entry_spread_mean = spread.rolling(window=spread_window).mean().iloc[i]
                    entry_spread_std  = spread.rolling(window=spread_window).std().iloc[i]
                    entry_alpha = alpha.iloc[i]
                    entry_beta  = beta.iloc[i]
                    entry_i_price = i_px.iloc[i]
                    entry_j_price = j_px.iloc[i]

                    total_beta = abs(entry_beta) + 1.0
                    notional_i = starting_capital * (abs(entry_beta) / total_beta)
                    notional_j = starting_capital * (1.0 / total_beta)

                    entry_shares_i = -notional_i / entry_i_price
                    entry_shares_j = notional_j / entry_j_price
                    entry_notional = notional_i + notional_j

                    entry_spread = log_j.iloc[i] - (entry_alpha + entry_beta * log_i.iloc[i])

                    signals.iloc[i] = +1
                    state = "long"

            # Exit logic
            if state in ("long", "short"):
                cur_i = i_px.iloc[i]
                cur_j = j_px.iloc[i]
                dollar_pnl = ((cur_j - entry_j_price) * entry_shares_j +
                              (cur_i - entry_i_price) * entry_shares_i)

                max_loss_dollars = sl_percent * entry_notional
                stop_hit = (dollar_pnl <= -max_loss_dollars)

                cur_spread = log_j.iloc[i] - (entry_alpha + entry_beta * log_i.iloc[i])
                signal_df.loc[idx, (pair, "modified_spread")] = cur_spread

                z_dyn = (cur_spread - entry_spread_mean) / entry_spread_std
                signal_df.loc[idx, (pair, "modified_zs")] = z_dyn

                mean_revert_exit = ((state == "long"  and z_dyn >= -0.3) or
                                    (state == "short" and z_dyn <=  0.3))

                if stop_hit or mean_revert_exit:
                    signals.iloc[i] = 6 if stop_hit else 5

                    # reset state
                    state = None
                    entry_alpha = entry_beta = None
                    entry_i_price = entry_j_price = None
                    entry_shares_i = entry_shares_j = 0.0
                    entry_notional = 0.0
                    entry_spread = None

        all_signals[pair] = signals

    return pd.DataFrame(all_signals)

import matplotlib.pyplot as plt

def plot_pair_trading_signals(signal_df, pairs, bounds, all_signals_df):
    """
    Plots spread, z-score, and modified spread with trade signals for each pair in a subplot grid.

    Parameters
    ----------
    signal_df : pd.DataFrame
        DataFrame indexed by datetime with multi-level columns for each pair and metric.
    pairs : list
        List of pairs to plot.
    bounds : dict
        Dictionary with pair keys and (lower_bound, upper_bound) z-score entry thresholds.
    all_signals_df : pd.DataFrame
        DataFrame of signals for each pair indexed by datetime.
    """
    n_rows = len(pairs)
    n_cols = 3

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5.5 * n_rows), squeeze=False)

    for row_idx, pair in enumerate(pairs):
        # Plot spread and rolling mean
        ax_spread = axes[row_idx, 0]
        spread = signal_df[(pair, 'spread')]
        ax_spread.plot(spread, label='Spread')
        ax_spread.plot(spread.rolling(90, min_periods=90).mean(), label='Rolling SMA (90)', color='red')

        # Mark signals on spread plot
        for sig_val, color, marker, label in [(1, 'green', '^', 'short spread'),
                                              (-1, 'red', 'v', 'long spread'),
                                              (6, 'blue', '*', 'exit stop loss'),
                                              (5, 'black', '*', 'exit trigger')]:
            mask = all_signals_df[pair] == sig_val
            ax_spread.scatter(spread.index[mask], spread[mask], color=color, marker=marker, label=label, s=50)

        ax_spread.set_title(f'{pair} - Spread')
        ax_spread.legend()

        # Plot z-score with bounds
        ax_z = axes[row_idx, 1]
        z_score = signal_df[(pair, 'zs')]
        ax_z.plot(z_score, label='Z-score')
        lower_bound, upper_bound = bounds[pair]
        ax_z.axhline(y=lower_bound, color='grey', linestyle='--', linewidth=0.8)
        ax_z.axhline(y=upper_bound, color='grey', linestyle='--', linewidth=0.8)

        for sig_val, color, marker, label in [(1, 'green', '^', 'short spread'),
                                              (-1, 'red', 'v', 'long spread'),
                                              (6, 'blue', '*', 'exit stop loss'),
                                              (5, 'black', '*', 'exit trigger')]:
            mask = all_signals_df[pair] == sig_val
            ax_z.scatter(z_score.index[mask], z_score[mask], color=color, marker=marker, label=label, s=50)

        ax_z.set_title(f'{pair} - Z-score')
        ax_z.legend()

        # Plot modified spread with signals
        ax_mod_spread = axes[row_idx, 2]
        mod_spread = signal_df[(pair, 'modified_spread')]
        ax_mod_spread.plot(mod_spread, label='Modified Spread')

        for sig_val, color, marker, label in [(1, 'green', '^', 'short spread'),
                                              (-1, 'red', 'v', 'long spread'),
                                              (6, 'blue', '*', 'exit stop loss'),
                                              (5, 'black', '*', 'exit trigger')]:
            mask = all_signals_df[pair] == sig_val
            ax_mod_spread.scatter(mod_spread.index[mask], mod_spread[mask], color=color, marker=marker, label=label, s=50)

        ax_mod_spread.set_title(f'{pair} - Modified Spread')
        ax_mod_spread.legend()

    plt.tight_layout()
    plt.show()

import numpy as np
import pandas as pd

def backtest_pairs_strategy(
    pairs,
    signal_df,
    bounds,
    sl_percent=0.05,        # Stop-loss %
    starting_capital=1_000_000,
    fee_rate=0.00045        # Fee per side (each entry and exit)
):
    """
    Backtest a pairs trading strategy based on z-score signals and mean reversion with stop-loss.

    Parameters
    ----------
    pairs : list of (str, str)
        List of asset pairs (tuples of symbol names).
    signal_df : pd.DataFrame
        MultiIndex DataFrame holding signals, prices, alphas, betas, spreads, zs, etc.
    bounds : dict
        Dict mapping each pair -> (upper_bound, lower_bound) z-score thresholds
    sl_percent : float, default 0.05
        Fractional stop-loss relative to entry notional
    starting_capital : float, default 1_000_000
        Notional capital to deploy per pair
    fee_rate : float, default 0.00045
        Fee rate per side (entry and exit)

    Returns
    -------
    result : dict
        - "positions": dict of pair -> DataFrame (positions over time)
        - "pnl": dict of pair -> Series (unrealized PnL curve)
        - "realized_pnl": dict of pair -> float (final realized PnL)
        - "fees_paid": dict of pair -> float
    """

    all_positions = {}
    all_pnl = {}
    all_realized_pnl = {}
    all_fees_paid = {}

    for pair in pairs:
        i_sym, j_sym = pair
        z      = signal_df[(pair, "zs")]
        log_i  = signal_df[(pair, "log_i")]
        log_j  = signal_df[(pair, "log_j")]
        alpha  = signal_df[(pair, "alpha")]
        beta   = signal_df[(pair, "beta")]
        spread = signal_df[(pair, "spread")]

        i_px = np.exp(log_i)
        j_px = np.exp(log_j)

        # Smooth z-score, detect turning points
        z_smooth = z.ewm(span=6, adjust=False).mean()
        d_ma = z_smooth.diff()
        turning_points = (d_ma.shift(1) * d_ma < 0)

        # Signal tracking
        positions_df = pd.DataFrame(0.0, index=signal_df.index, columns=[i_sym, j_sym])
        pnl_series = pd.Series(0.0, index=signal_df.index)
        realized_pnl_curve = pd.Series(0.0, index=signal_df.index)

        state = None
        current_position_i = current_position_j = 0.0
        entry_alpha = entry_beta = entry_i_price = entry_j_price = None
        entry_shares_i = entry_shares_j = entry_notional = 0.0
        entry_spread_mean = entry_spread_std = None

        realized_pnl = 0.0
        fees_paid = 0.0

        for idx, timestamp in enumerate(signal_df.index):
            cur_z = z.iloc[idx]
            b = beta.iloc[idx]

            positions_df.loc[timestamp, i_sym] = current_position_i
            positions_df.loc[timestamp, j_sym] = current_position_j

            # Skip if missing data
            if pd.isna(cur_z) or pd.isna(b):
                pnl_series.loc[timestamp] = pnl_series.iloc[idx-1] if idx > 0 else 0.0
                continue

            # --- Entry logic ---
            if state is None:
                if cur_z > bounds[pair][0]:
                    state = "pos_waiting"
                elif cur_z < bounds[pair][1]:
                    state = "neg_waiting"

            elif state == "pos_waiting":
                if bool(turning_points.iloc[idx]) and cur_z > 1.0:
                    # Enter short
                    entry_spread_mean = spread.rolling(window=90).mean().iloc[idx]
                    entry_spread_std = spread.rolling(window=90).std().iloc[idx]
                    entry_alpha = alpha.iloc[idx]
                    entry_beta = b
                    entry_i_price = i_px.iloc[idx]
                    entry_j_price = j_px.iloc[idx]

                    total_beta = abs(entry_beta) + 1.0
                    notional_i = starting_capital * (abs(entry_beta) / total_beta)
                    notional_j = starting_capital * (1.0 / total_beta)

                    entry_shares_i = notional_i / entry_i_price
                    entry_shares_j = -notional_j / entry_j_price
                    entry_notional = notional_i + notional_j

                    # Update positions
                    current_position_i = entry_shares_i
                    current_position_j = entry_shares_j
                    positions_df.loc[timestamp, [i_sym, j_sym]] = [current_position_i, current_position_j]

                    # Fees
                    entry_fees = fee_rate * entry_notional * 2
                    realized_pnl -= entry_fees
                    fees_paid += entry_fees

                    state = "short"

            elif state == "neg_waiting":
                if bool(turning_points.iloc[idx]) and cur_z < -1.0:
                    # Enter long
                    entry_spread_mean = spread.rolling(window=90).mean().iloc[idx]
                    entry_spread_std = spread.rolling(window=90).std().iloc[idx]
                    entry_alpha = alpha.iloc[idx]
                    entry_beta = b
                    entry_i_price = i_px.iloc[idx]
                    entry_j_price = j_px.iloc[idx]

                    total_beta = abs(entry_beta) + 1.0
                    notional_i = starting_capital * (abs(entry_beta) / total_beta)
                    notional_j = starting_capital * (1.0 / total_beta)

                    entry_shares_i = -notional_i / entry_i_price
                    entry_shares_j = notional_j / entry_j_price
                    entry_notional = notional_i + notional_j

                    # Update positions
                    current_position_i = entry_shares_i
                    current_position_j = entry_shares_j
                    positions_df.loc[timestamp, [i_sym, j_sym]] = [current_position_i, current_position_j]

                    # Fees
                    entry_fees = fee_rate * entry_notional * 2
                    realized_pnl -= entry_fees
                    fees_paid += entry_fees

                    state = "long"

            # --- Manage open positions ---
            if state in ("long", "short"):
                cur_i = i_px.iloc[idx]
                cur_j = j_px.iloc[idx]

                dollar_pnl = (cur_j - entry_j_price) * entry_shares_j + \
                             (cur_i - entry_i_price) * entry_shares_i
                pnl_series.loc[timestamp] = dollar_pnl

                # Stop-loss
                max_loss_dollars = sl_percent * entry_notional
                stop_hit = (dollar_pnl <= -max_loss_dollars)

                # Dynamic z-score
                cur_spread = log_j.iloc[idx] - (entry_alpha + entry_beta * log_i.iloc[idx])
                signal_df.loc[timestamp, (pair, "modified_spread")] = cur_spread
                z_dyn = (cur_spread - entry_spread_mean) / entry_spread_std
                signal_df.loc[timestamp, (pair, "modified_zs")] = z_dyn

                # Exit condition
                mean_revert_exit = ((state == "long" and z_dyn >= -0.3) or
                                    (state == "short" and z_dyn <= 0.3))

                if stop_hit or mean_revert_exit:
                    realized_pnl += dollar_pnl
                    exit_fees = fee_rate * entry_notional * 2
                    realized_pnl -= exit_fees
                    pnl_series.loc[timestamp] = dollar_pnl - exit_fees
                    fees_paid += exit_fees

                    # Reset position
                    current_position_i = current_position_j = 0.0
                    positions_df.loc[timestamp, [i_sym, j_sym]] = [0.0, 0.0]
                    state = None

            else:
                pnl_series.loc[timestamp] = pnl_series.iloc[idx-1] if idx > 0 else 0.0
                realized_pnl_curve.loc[timestamp] = realized_pnl_curve.iloc[idx-1] if idx > 0 else 0.0

        # Save results for this pair
        all_positions[pair] = positions_df
        all_pnl[pair] = pnl_series
        all_realized_pnl[pair] = realized_pnl
        all_fees_paid[pair] = fees_paid

    return {
        "positions": all_positions,
        "pnl": all_pnl,
        "realized_pnl": all_realized_pnl,
        "fees_paid": all_fees_paid
    }

import matplotlib.pyplot as plt
import pandas as pd

def plot_cumulative_realized_pnl(all_positions, all_pnl, signal_df, pairs, starting_capital=1_000_000):
    """
    Plot cumulative realized PnL and portfolio value over time for each pair.

    Parameters
    ----------
    all_positions : dict
        Dictionary with pair keys and DataFrame values for positions.
    all_pnl : dict
        Dictionary with pair keys and Series values for daily pnl.
    signal_df : pd.DataFrame
        Signal DataFrame indexed by timestamp.
    pairs : list of tuples
        List of (i_sym, j_sym) pairs.
    starting_capital : float
        Initial capital amount per pair.

    Returns
    -------
    None
    """
    for pair in pairs:
        cumulative_realized_pnl = pd.Series(0.0, index=signal_df.index)
        realized_sum = 0.0

        pos_df = all_positions[pair]
        pnl_series = all_pnl[pair]
        i_sym, j_sym = pair

        for idx, timestamp in enumerate(signal_df.index):
            if idx > 0:
                prev_pos_i = pos_df[i_sym].iloc[idx-1]
                prev_pos_j = pos_df[j_sym].iloc[idx-1]
                cur_pos_i = pos_df[i_sym].iloc[idx]
                cur_pos_j = pos_df[j_sym].iloc[idx]

                exited = (prev_pos_i != 0 or prev_pos_j != 0) and (cur_pos_i == 0 and cur_pos_j == 0)
                if exited:
                    realized_sum += pnl_series.iloc[idx]

            cumulative_realized_pnl.iloc[idx] = realized_sum

        portfolio_value = starting_capital + cumulative_realized_pnl

        plt.figure(figsize=(12, 6))
        plt.plot(portfolio_value.index, portfolio_value.values, label='Portfolio Value')
        plt.title(f'Cumulative Portfolio Value Over Time for {i_sym}-{j_sym}')
        plt.xlabel('Time')
        plt.ylabel('Portfolio Value ($)')
        plt.legend()
        plt.grid(True)
        plt.show()

