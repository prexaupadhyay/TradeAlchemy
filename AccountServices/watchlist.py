"""
Watchlist Manager Module

This module manages user watchlists - tracked stocks with real-time price data.
It provides CRUD operations and price fetching with in-memory caching.

Key Features:
- Add/remove stocks from watchlist
- Fetch real-time prices with caching (5-minute expiry)
- Sparkline data for mini charts
- Percentage change tracking
- Bulk price updates (efficient API usage)

Author: TradeAlchemy Team
"""

import sqlite3
import time
from typing import List, Dict, Optional
from datetime import datetime
import yfinance as yf
import pandas as pd

# ============================================================================
# GLOBAL CACHE
# ============================================================================

# Global in-memory cache for price data
# Structure: { 'TICKER': { 'data': {...}, 'timestamp': time.time() } }
PRICE_CACHE = {}
CACHE_DURATION = 300  # 5 minutes in seconds


# ============================================================================
# WATCHLIST MANAGER CLASS
# ============================================================================

class WatchlistManager:
    """
    Manages user watchlists with real-time price data and caching.

    This class provides a complete interface for watchlist operations:
    - Add stocks to watchlist
    - Remove stocks from watchlist
    - Fetch basic watchlist (tickers only)
    - Fetch enhanced watchlist (with prices, changes, sparklines)

    The manager implements intelligent caching to minimize API calls:
    - Prices cached for 5 minutes
    - Bulk downloads for efficiency
    - Graceful error handling for missing data

    Attributes:
        db: DatabaseManager instance
        user_id (int): Current user's database ID

    Example:
        >>> wm = WatchlistManager(db, user_id=123)
        >>> wm.add_stock("AAPL")
        >>> stocks = wm.get_watchlist_with_prices()
        >>> for stock in stocks:
        >>>     print(f"{stock['ticker']}: ${stock['current_price']}")
    """

    def __init__(self, db_manager, user_id: int):
        """
        Initialize watchlist manager for specific user.

        Args:
            db_manager: DatabaseManager instance
            user_id (int): Database ID of the user

        Side Effects:
            - Stores references to db and user_id
            - Does NOT create database connection (created per-operation)
        """
        self.db = db_manager
        self.user_id = user_id

    def get_watchlist(self) -> List[Dict]:
        """
        Fetch basic watchlist data from database (tickers only, no prices).

        This method retrieves the user's watchlist without fetching prices.
        Used when only ticker symbols are needed (faster than full data).

        Returns:
            List[Dict]: List of watchlist entries:
                [
                    {
                        'ticker': 'AAPL',
                        'buy_price': 150.00,
                        'added_at': '2024-02-19 14:30:00'
                    },
                    ...
                ]
                Empty list if no stocks in watchlist

        Database Query:
            - Filters by user_id (only this user's stocks)
            - Orders by ticker alphabetically

        Example:
            >>> stocks = wm.get_watchlist()
            >>> print(f"You're tracking {len(stocks)} stocks")
        """
        conn = self.db.get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """SELECT ticker, buy_price, added_at
               FROM watchlist
               WHERE user_id = ?
               ORDER BY ticker""",
            (self.user_id,)
        )

        # Convert database rows to dictionaries
        results = [
            {
                'ticker': row[0],
                'buy_price': row[1],
                'added_at': row[2]
            }
            for row in cursor.fetchall()
        ]

        conn.close()
        return results

    def add_stock(self, ticker: str, buy_price: Optional[float] = None,
                  added_at: Optional[str] = None) -> Dict:
        """
        Add a stock to the watchlist.

        This method creates a new watchlist entry with the ticker symbol
        and optional purchase price for tracking gains/losses.

        Args:
            ticker (str): Stock ticker symbol (e.g., "AAPL", "GOOGL")
            buy_price (float, optional): Purchase price for P/L tracking
            added_at (str, optional): Custom timestamp (default: current time)

        Returns:
            Dict: Operation result:
                Success: {'success': True, 'message': 'AAPL added to watchlist'}
                Duplicate: {'success': False, 'message': 'AAPL is already in your watchlist'}
                Error: {'success': False, 'message': 'Failed to add stock: {error}'}

        Database Constraints:
            - UNIQUE(user_id, ticker): Prevents duplicate entries
            - ticker is stored in uppercase
            - Whitespace is stripped

        Example:
            >>> wm.add_stock("AAPL", buy_price=150.00)
            {'success': True, 'message': 'AAPL added to watchlist'}
            >>>
            >>> # Try adding again
            >>> wm.add_stock("AAPL")
            {'success': False, 'message': 'AAPL is already in your watchlist'}
        """
        try:
            # Default to current timestamp if not provided
            if not added_at:
                added_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Clean and normalize ticker symbol
            clean_ticker = ticker.upper().strip()

            conn = self.db.get_connection()
            cursor = conn.cursor()

            # Insert into database
            cursor.execute(
                """INSERT INTO watchlist (user_id, ticker, buy_price, added_at)
                   VALUES (?, ?, ?, ?)""",
                (self.user_id, clean_ticker, buy_price, added_at)
            )

            conn.commit()
            conn.close()

            return {
                'success': True,
                'message': f'{clean_ticker} added to watchlist'
            }

        except sqlite3.IntegrityError:
            # UNIQUE constraint violated - ticker already in watchlist
            return {
                'success': False,
                'message': f'{ticker.upper()} is already in your watchlist'
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Failed to add stock: {str(e)}'
            }

    def remove_stock(self, ticker: str) -> Dict:
        """
        Remove a stock from the watchlist.

        This method deletes a watchlist entry for the specified ticker.

        Args:
            ticker (str): Stock ticker to remove

        Returns:
            Dict: Operation result:
                Success: {'success': True, 'message': 'AAPL removed.'}
                Not found: {'success': False, 'message': 'Stock not found in watchlist.'}
                Error: {'success': False, 'message': 'Error: {error}'}

        Implementation Notes:
            - Ticker is normalized (uppercase, stripped) before deletion
            - Uses cursor.rowcount to check if row was actually deleted

        Example:
            >>> wm.remove_stock("AAPL")
            {'success': True, 'message': 'AAPL removed.'}
            >>>
            >>> # Try removing non-existent stock
            >>> wm.remove_stock("INVALID")
            {'success': False, 'message': 'Stock not found in watchlist.'}
        """
        try:
            conn = self.db.get_connection()
            cursor = conn.cursor()

            # Clean ticker to ensure trailing spaces don't prevent deletion
            clean_ticker = ticker.strip().upper()

            cursor.execute(
                "DELETE FROM watchlist WHERE user_id = ? AND ticker = ?",
                (self.user_id, clean_ticker)
            )

            # Check if any row was actually deleted
            count = cursor.rowcount
            conn.commit()
            conn.close()

            if count > 0:
                return {'success': True, 'message': f'{clean_ticker} removed.'}
            return {'success': False, 'message': 'Stock not found in watchlist.'}

        except Exception as e:
            return {'success': False, 'message': f'Error: {str(e)}'}

    def _get_cached_data(self, ticker: str):
        """
        Retrieve price data from cache if still valid.

        This internal method checks if we have recent price data cached
        to avoid unnecessary API calls.

        Args:
            ticker (str): Stock ticker symbol

        Returns:
            dict or None: Cached data if valid, None if expired or not cached

        Cache Validity:
            - Data is considered valid if < 5 minutes old
            - After 5 minutes, returns None to trigger fresh fetch

        Example (internal):
            >>> cached = self._get_cached_data("AAPL")
            >>> if cached:
            >>>     print("Using cached price")
            >>> else:
            >>>     print("Need to fetch fresh data")
        """
        if ticker in PRICE_CACHE:
            entry = PRICE_CACHE[ticker]
            # Check if cache entry is still valid (< 5 minutes old)
            if time.time() - entry['timestamp'] < CACHE_DURATION:
                return entry['data']
        return None

    def _update_cache(self, ticker: str, data: Dict):
        """
        Update cache with new price data.

        This internal method stores freshly fetched data in the cache
        with a timestamp for expiration checking.

        Args:
            ticker (str): Stock ticker symbol
            data (Dict): Price data to cache

        Side Effects:
            - Updates global PRICE_CACHE dictionary
            - Overwrites any existing cached data for this ticker

        Cache Structure:
            PRICE_CACHE = {
                'AAPL': {
                    'data': {
                        'ticker': 'AAPL',
                        'current_price': '178.50',
                        'change_percent': 2.5,
                        'sparkline_data': [175.2, 176.3, ...]
                    },
                    'timestamp': 1708355400.123
                }
            }
        """
        PRICE_CACHE[ticker] = {
            'data': data,
            'timestamp': time.time()
        }

    def get_watchlist_with_prices(self) -> List[Dict]:
        """
        Fetch watchlist data with real-time prices, changes, and sparklines.

        This is the main method for displaying the watchlist with current market data.
        It efficiently fetches prices using caching and bulk downloads.

        Returns:
            List[Dict]: Enhanced watchlist with price data:
                [
                    {
                        'ticker': 'AAPL',
                        'added_at': '2024-02-19',
                        'current_price': '178.50',
                        'change_percent': 2.34,
                        'sparkline_data': [175.2, 176.3, 177.1, 178.5]
                    },
                    ...
                ]

        Process Flow:
            1. Fetch basic watchlist from database
            2. Check cache for each ticker
            3. Bulk download missing tickers from yfinance
            4. Calculate price changes and sparklines
            5. Update cache for future requests
            6. Return sorted results

        Price Data Structure:
            - current_price: Latest close price (formatted string)
            - change_percent: % change from previous day (float)
            - sparkline_data: Last 20 days of closes (for mini chart)

        Optimization:
            - Cached tickers skip the API call
            - Non-cached tickers downloaded in single bulk call
            - Efficient handling of international stocks

        Error Handling:
            - Missing data shows "N/A" or "Error"
            - Partial failures don't crash entire watchlist
            - Gracefully handles yfinance API errors

        Edge Cases:
            - Single ticker: Works correctly (no MultiIndex)
            - International stocks: Handles .NS, .SZ, .KS, etc.
            - Suspended trading: Shows last known price

        Example:
            >>> stocks = wm.get_watchlist_with_prices()
            >>> for stock in stocks:
            >>>     if stock['current_price'] != "N/A":
            >>>         print(f"{stock['ticker']}: ${stock['current_price']} "
            >>>               f"({stock['change_percent']:+.2f}%)")
        """
        # Step 1: Get basic watchlist from database
        stocks = self.get_watchlist()
        if not stocks:
            return []

        detailed_stocks = []
        tickers_to_fetch = []

        # Step 2: Check cache for each ticker
        for stock in stocks:
            cached = self._get_cached_data(stock['ticker'])
            if cached:
                # Use cached data
                stock_data = cached.copy()
                stock_data['added_at'] = stock['added_at']
                detailed_stocks.append(stock_data)
            else:
                # Need to fetch fresh data
                tickers_to_fetch.append(stock['ticker'])

        # Step 3: Fetch missing tickers in bulk
        if tickers_to_fetch:
            print(f"📊 Fetching fresh data for: {', '.join(tickers_to_fetch)}")

            try:
                # Bulk download: More efficient than individual calls
                bulk_data = yf.download(
                    tickers_to_fetch,
                    period="1mo",  # Last month for sparklines
                    interval="1d",  # Daily candles
                    group_by="ticker",  # Organize by ticker
                    progress=False,  # Suppress progress bar
                    threads=False  # Sequential (more reliable)
                )

                # Step 4: Process each ticker's data
                for ticker in tickers_to_fetch:
                    # Default data structure (in case of errors)
                    data = {
                        'ticker': ticker,
                        'current_price': "N/A",
                        'change_percent': 0.0,
                        'sparkline_data': []
                    }

                    try:
                        hist = pd.DataFrame()

                        # Handle MultiIndex vs Flat DataFrame structure
                        # (Depends on whether we fetched single or multiple tickers)
                        if isinstance(bulk_data.columns, pd.MultiIndex):
                            # Multiple tickers: DataFrame has MultiIndex columns
                            try:
                                hist = bulk_data[ticker]
                            except KeyError:
                                pass
                        else:
                            # Single ticker: Flat DataFrame
                            if len(tickers_to_fetch) == 1:
                                hist = bulk_data

                        # Extract price data if available
                        if not hist.empty and 'Close' in hist.columns:
                            # Drop rows with missing Close prices
                            hist = hist.dropna(subset=['Close'])

                            if not hist.empty:
                                closes = hist['Close'].tolist()
                                current_price = closes[-1]

                                # Calculate percentage change from previous day
                                if len(closes) > 1:
                                    prev_close = closes[-2]
                                    change = ((current_price - prev_close) / prev_close) * 100
                                else:
                                    change = 0.0

                                # Update data structure
                                data['current_price'] = f"{current_price:,.2f}"
                                data['change_percent'] = round(change, 2)
                                data['sparkline_data'] = closes[-20:]  # Last 20 days for chart

                    except Exception as e:
                        print(f"⚠️ Error processing {ticker}: {e}")

                    # Step 5: Cache valid data
                    if data['current_price'] != "N/A" and data['current_price'] != "Error":
                        self._update_cache(ticker, data)

                    # Prepare final record with added_at timestamp
                    final_stock_record = data.copy()
                    original = next((s for s in stocks if s['ticker'] == ticker), None)
                    if original:
                        final_stock_record['added_at'] = original['added_at']

                    detailed_stocks.append(final_stock_record)

            except Exception as e:
                # Critical failure in bulk download
                print(f"❌ Bulk download critical error: {e}")

                # Add error entries for all failed tickers
                for ticker in tickers_to_fetch:
                    detailed_stocks.append({
                        'ticker': ticker,
                        'added_at': datetime.now().strftime("%Y-%m-%d"),
                        'current_price': "Error",
                        'change_percent': 0.0,
                        'sparkline_data': []
                    })

        # Step 6: Sort alphabetically and return
        detailed_stocks.sort(key=lambda x: x['ticker'])
        return detailed_stocks