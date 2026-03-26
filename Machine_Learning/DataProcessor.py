"""
Data Processing and Feature Engineering Module

This module transforms raw stock price data into machine learning features
for prediction models. It calculates technical indicators, handles missing data,
and creates target variables for training.

Key Features:
- Technical indicator calculation (RSI, MACD, ATR, Moving Averages)
- Context features (relative strength vs peers/partners)
- Volatility regime detection
- Robust data validation
- Target generation for ML models

Author: TradeAlchemy Team
"""

import numpy as np
import pandas as pd
from typing import Tuple, List


# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class DataValidationError(Exception):
    """
    Raised when input data fails validation checks.

    Use Cases:
        - Empty DataFrame
        - Missing required columns
        - Insufficient data points for indicator calculation
    """
    pass


# ============================================================================
# FEATURE CALCULATOR CLASS
# ============================================================================

class FeatureCalculator:
    """
    Calculates technical indicators and contextual features for stock analysis.

    This class transforms raw OHLCV (Open, High, Low, Close, Volume) data into
    a comprehensive set of features suitable for machine learning models.

    Attributes:
        required_columns (list): Column names that must exist in input DataFrame

    Feature Categories:
        1. Price-based: Returns, Moving Averages
        2. Volatility: ATR, Volatility Regime
        3. Momentum: RSI, MACD
        4. Context: Relative Strength vs Peers/Partners
        5. Target: Binary classification (significant price move or not)

    Example:
        >>> calc = FeatureCalculator()
        >>> df_with_features = calc.calculate_features(raw_df, regime="volatile")
        >>> print(df_with_features[['RSI', 'MACD', 'Target']].tail())
    """

    def __init__(self):
        """
        Initialize the feature calculator with required column definitions.

        Required Columns:
            - Open: Opening price
            - High: Highest price of the period
            - Low: Lowest price of the period
            - Close: Closing price
            - Volume: Trading volume

        Note: AdjClose (Adjusted Close) is also used if available, otherwise
        Close price is used as fallback.
        """
        # Define required columns for stock data processing
        # standard format of Yahoo Finance
        self.required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    def validate_input(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        # -> : in python that the function returns
        """
        Validate input DataFrame before processing features.

        This method performs three critical validation checks:
        1. Data existence (non-empty DataFrame)
        2. Column presence (all required OHLCV columns present)
        3. Sufficient data (at least 60 rows for indicator calculation)

        Args:
            df (pd.DataFrame): Raw stock data to validate

        Returns:
            Tuple[bool, List[str]]: (is_valid, error_messages)
                - is_valid: True if all validations pass, False otherwise
                - error_messages: List of specific validation failures

        Validation Rules:
            - Minimum 60 rows required (for 50-day MA + buffer)
            - Must have Close or AdjClose column
            - Must have Open, High, Low, Volume columns

        Example:
            >>> calc = FeatureCalculator()
            >>> is_valid, errors = calc.validate_input(df)
            >>> if not is_valid:
            >>>     print(f"Validation failed: {errors}")
        """
        errors = []

        # Check 1: DataFrame is not empty
        if df.empty:
            errors.append("DataFrame is empty")
            return False, errors

        # Check 2: Required columns present
        missing = set(self.required_columns) - set(df.columns)
        if missing and 'AdjClose' not in df.columns and 'Close' not in df.columns:
            errors.append(f"Missing critical columns: {missing}")

        # Check 3: Sufficient data for indicator calculation
        if len(df) < 60:
            errors.append(f"Need at least 60 rows, got {len(df)}")

        return len(errors) == 0, errors

    def add_context_features(self, target_df, market_map, context):
        """
        Calculate Relative Strength vs Peers and Partners.

        This method adds ecosystem-aware features by comparing the target stock's
        performance against its competitors (peers) and supply chain (partners).

        Args:
            target_df (pd.DataFrame): DataFrame with features for target stock
            market_map (dict): Dictionary mapping tickers to their DataFrames
                Format: {"AAPL": df_aapl, "MSFT": df_msft, ...}
            context (dict): Ecosystem information from Gemini
                Format: {
                    "peers": ["MSFT", "GOOGL", ...],
                    "partners": ["TSM", "QCOM", ...]
                }

        Returns:
            pd.DataFrame: Original DataFrame with two new columns:
                - Rel_Str_Peers: Relative strength vs competitors (ratio)
                - Rel_Str_Partners: Relative strength vs supply chain (ratio)

        Calculation Method:
            1. For each group (peers/partners):
               - Normalize each stock to start at 1.0
               - Average the normalized prices to create group index
            2. Calculate target's relative strength:
               - Rel_Str = (Target Normalized) / (Group Index)
               - Values > 1.0: Outperforming the group
               - Values < 1.0: Underperforming the group

        Edge Cases:
            - Missing data: Forward fill (carry last known value)
            - No peers/partners data: Default to 1.0 (neutral)
            - Date misalignment: Reindex to target's dates

        Use Case:
            If AAPL is at 1.2 relative to peers, it has outperformed its
            competitors by 20% over the time period.
        """
        df = target_df.copy()

        # Helper function to calculate group index
        def get_group_index(tickers):
            """
            Create an averaged, normalized index from multiple tickers.

            Args:
                tickers (list): List of ticker symbols

            Returns:
                pd.Series or None: Average normalized price series,
                                   or None if no valid data
            """
            prices = pd.DataFrame(index=df.index)

            # Collect all available ticker data
            for t in tickers:
                if t in market_map and not market_map[t].empty:
                    # Align dates and forward fill missing values
                    if 'AdjClose' in market_map[t].columns:
                        clean_series = market_map[t]['AdjClose'].reindex(df.index).ffill()
                        prices[t] = clean_series

            if prices.empty:
                return None

            # Normalize each stock to start at 1.0, then average them
            # This creates a "group performance index"
            return (prices / prices.iloc[0]).mean(axis=1)

        # 1. Peer Analysis (Relative Strength vs Competitors)
        peer_idx = get_group_index(context.get('peers', []))
        if peer_idx is not None:
            # Normalize target stock to start at 1.0
            target_norm = df['AdjClose'] / df['AdjClose'].iloc[0]
            # Calculate relative strength ratio
            df['Rel_Str_Peers'] = target_norm / peer_idx
        else:
            # No peer data available - set to neutral (1.0)
            df['Rel_Str_Peers'] = 1.0

        # 2. Partner Analysis (Relative Strength vs Supply Chain)
        partner_idx = get_group_index(context.get('partners', []))
        if partner_idx is not None:
            target_norm = df['AdjClose'] / df['AdjClose'].iloc[0]
            df['Rel_Str_Partners'] = target_norm / partner_idx
        else:
            # No partner data available - set to neutral (1.0)
            df['Rel_Str_Partners'] = 1.0

        return df

    def calculate_features(self, df, threshold=0.01, regime="volatile"):
        """
        Calculate comprehensive technical indicators and generate targets.

        This is the main feature engineering method. It transforms raw OHLCV data
        into a rich feature set for machine learning models.

        Args:
            df (pd.DataFrame): Raw stock data with OHLCV columns
            threshold (float): Minimum price change % to be considered "significant"
                (default: 0.01 = 1%)
            regime (str): Market regime classification ("stable" or "volatile")
                Currently not used in feature calculation but reserved for future use

        Returns:
            pd.DataFrame or None: Enhanced DataFrame with features:
                - Original OHLCV columns
                - Ret: Daily returns
                - ATR: Average True Range (volatility)
                - Vol_Regime: Current volatility vs historical average
                - RSI: Relative Strength Index (momentum)
                - MACD: MACD line and histogram
                - MA_50: 50-day moving average
                - Dist_MA50: Distance from 50-day MA (%)
                - Target: Binary (1 if |next return| > threshold, 0 otherwise)
                - Target_Direction: Binary (1 if next return > 0, 0 otherwise)
                Returns None if validation fails

        Feature Descriptions:

            1. Ret (Returns):
               - Daily percentage price change
               - Formula: (Close_t - Close_t-1) / Close_t-1
               - Range: Typically -10% to +10% for most stocks

            2. ATR (Average True Range):
               - Measures market volatility
               - Formula: 14-day average of True Range
               - True Range = max(H-L, |H-C_prev|, |L-C_prev|)
               - Higher ATR = More volatile

            3. Vol_Regime (Volatility Regime):
               - Current volatility relative to long-term average
               - Formula: ATR_14 / ATR_100_average
               - Values > 1.0: More volatile than usual
               - Values < 1.0: Less volatile than usual

            4. RSI (Relative Strength Index):
               - Momentum oscillator (0-100)
               - Formula: 100 - (100 / (1 + RS))
               - RS = Average Gain / Average Loss over 14 days
               - > 70: Potentially overbought
               - < 30: Potentially oversold

            5. MACD (Moving Average Convergence Divergence):
               - Trend-following momentum indicator
               - MACD Line: EMA_12 - EMA_26
               - Signal Line: EMA_9 of MACD Line
               - MACD_Hist: MACD Line - Signal Line
               - Crossovers indicate potential trend changes

            6. MA_50 (50-Day Moving Average):
               - Average closing price over last 50 days
               - Acts as support/resistance level
               - Price above MA_50: Uptrend
               - Price below MA_50: Downtrend

            7. Dist_MA50 (Distance from MA):
               - How far current price is from 50-day average
               - Formula: (Close - MA_50) / MA_50
               - Positive: Above average (bullish)
               - Negative: Below average (bearish)

            8. Target (Binary Classification):
               - Predicts if a SIGNIFICANT price move will occur
               - 1: Next day's |return| > threshold (e.g., 1%)
               - 0: Next day's |return| <= threshold
               - This is what the ML model tries to predict

            9. Target_Direction (Directional Indicator):
               - Direction of next day's price move
               - 1: Price went up
               - 0: Price went down
               - Stored for display but not used in training

        Data Cleaning:
            - Replaces infinite values with 0
            - Drops rows with NaN (typically first 50 rows due to MA calculation)

        Example:
            >>> calc = FeatureCalculator()
            >>> df_enhanced = calc.calculate_features(raw_df, threshold=0.015)
            >>> print(f"Generated {len(df_enhanced)} training samples")
            >>> print(f"Target distribution: {df_enhanced['Target'].value_counts()}")
        """
        # Validate input data first
        is_valid, errors = self.validate_input(df)
        if not is_valid:
            return None

        df = df.copy()

        # Fallback: Use Close if AdjClose not available
        if 'AdjClose' not in df.columns:
            df['AdjClose'] = df['Close']

        # ===================================================================
        # 1. RETURNS - Daily percentage change in price
        # ===================================================================
        df['Ret'] = df['AdjClose'].pct_change()

        # ===================================================================
        # 2. VOLATILITY (ATR - Average True Range)
        # ===================================================================
        # True Range is the greatest of:
        # - Today's High - Today's Low
        # - |Today's High - Yesterday's Close|
        # - |Today's Low - Yesterday's Close|
        h_l = df['High'] - df['Low']
        h_c = np.abs(df['High'] - df['AdjClose'].shift())
        # .shift() moves the columns down by exactly one rows
        l_c = np.abs(df['Low'] - df['AdjClose'].shift())
        tr = pd.concat([h_l, h_c, l_c], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(14).mean()

        # Volatility Regime: Current volatility vs historical average
        # > 1.0: More volatile than usual
        # < 1.0: Less volatile than usual
        df['Vol_Regime'] = df['ATR'] / df['ATR'].rolling(100).mean()

        # ===================================================================
        # 3. RSI (Relative Strength Index) - Momentum Indicator
        # ===================================================================
        delta = df['AdjClose'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # ===================================================================
        # 4. MACD (Moving Average Convergence Divergence)
        # ===================================================================
        ema12 = df['AdjClose'].ewm(span=12).mean()
        # Exponential weighted math
        ema26 = df['AdjClose'].ewm(span=26).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Hist'] = df['MACD'] - df['MACD'].ewm(span=9).mean()
        # This mathematical difference tells you whether the short-term trend is pulling away from the long-term trend (Divergence) or coming back toward it (Convergence).
        # When the macd_line crosses above the signal_line, it is generally considered a bullish (buy) signal. When it crosses below, it is a bearish (sell) signal.

        # ===================================================================
        # 5. MOVING AVERAGES
        # ===================================================================
        df['MA_50'] = df['AdjClose'].rolling(50).mean()
        # Distance from MA (percentage)
        df['Dist_MA50'] = (df['AdjClose'] - df['MA_50']) / df['MA_50']

        # ===================================================================
        # TARGET GENERATION (Volatility Focus)
        # ===================================================================
        # Next day's return (shifted by -1)
        df['Next_Ret'] = df['AdjClose'].pct_change().shift(-1)
        # shift up

        # Binary Target: Is next day's move > threshold?
        # This predicts VOLATILITY (significant moves), not direction
        df['Target'] = (df['Next_Ret'].abs() > threshold).astype(int)

        # Store direction for UI display (not used in training)
        df['Target_Direction'] = (df['Next_Ret'] > 0).astype(int)

        # ===================================================================
        # DATA CLEANING
        # ===================================================================
        # Replace infinite values with 0
        df = df.replace([np.inf, -np.inf], 0)
        # Drop rows with NaN (first ~50 rows due to MA calculation)
        df = df.dropna()

        return df

    def calculate_features_for_api(self, df, regime="volatile"):
        """
        Flask-specific wrapper for calculate_features().

        This method exists for API consistency and can be extended
        with API-specific logic in the future (e.g., caching, logging).

        Args:
            df (pd.DataFrame): Raw stock data
            regime (str): Market regime classification

        Returns:
            pd.DataFrame or None: Enhanced DataFrame with features

        Example:
            >>> calc = FeatureCalculator()
            >>> df_for_api = calc.calculate_features_for_api(raw_df, regime="stable")
        """
        df_with_features = self.calculate_features(df, regime=regime)
        if df_with_features is None:
            return None
        return df_with_features
