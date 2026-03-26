"""
Yahoo Finance Web Scraper Module

This module provides a robust interface to scrape financial data from Yahoo Finance
without using official APIs. It handles session management, proxy support, and multiple
data endpoint formats (v7, v8, v10).

Key Features:
- Session management with retry logic
- Optional proxy support (Tor SOCKS5)
- Historical price data (v8)
- Real-time quote data (v7)
- Fundamental financial data (v10)
- International stock support

Author: TradeAlchemy Team
"""

import requests
import time
import pandas as pd


# ============================================================================
# CUSTOM EXCEPTION CLASSES
# ============================================================================

class ScraperException(Exception):
    """Base exception class for all scraper-related errors"""
    pass


class SessionSetupError(ScraperException):
    """Raised when unable to establish a session with Yahoo Finance after all retries"""
    pass


class DataFetchError(ScraperException):
    """Raised when data fetching fails for a specific endpoint"""
    pass


class InvalidTickerError(ScraperException):
    """Raised when the provided ticker symbol is invalid or not found"""
    pass


# ============================================================================
# MAIN SCRAPER CLASS
# ============================================================================

class YahooScraper:
    """
    Main class for scraping financial data from Yahoo Finance.

    This scraper uses Yahoo Finance's internal API endpoints to retrieve:
    - Historical OHLCV data (Open, High, Low, Close, Volume)
    - Real-time market quotes
    - Fundamental financial metrics
    - Company profile information

    Attributes:
        proxy (str): Status of proxy connection ("Not Checked", "Proxy Active", "Proxy Inactive")

    Example:
        >>> scraper = YahooScraper()
        >>> data = scraper.scrape("AAPL", v7=True, v8=True, v10=True)
        >>> print(data['v7']['Current Price'])
    """

    def __init__(self):
        """Initialize the scraper with default proxy status"""
        self.proxy = "Not Checked"

    def _setup_session(self, use_proxy=False, max_retries=3):
        """
        Establish an authenticated session with Yahoo Finance.

        This method creates a requests session, optionally routes through Tor proxy,
        and obtains a 'crumb' token required for authenticated API calls.

        Args:
            use_proxy (bool): If True, route traffic through Tor SOCKS5 proxy on localhost:9050
            max_retries (int): Number of attempts before giving up (default: 3)

        Returns:
            tuple: (session, crumb)
                - session: Authenticated requests.Session object
                - crumb: Yahoo Finance authentication token (string)

        Raises:
            SessionSetupError: If unable to establish session after all retries

        Implementation Notes:
            - Uses exponential backoff (2^attempt seconds) between retries
            - First visits fc.yahoo.com to establish cookies
            - Then retrieves crumb from /v1/test/getcrumb endpoint
            - Crumb is required for subsequent API calls to prevent scraping
        """
        for attempt in range(max_retries):
            try:
                # Create a new session with persistent cookies
                session = requests.Session()

                # Configure Tor SOCKS5 proxy if requested
                if use_proxy:
                    session.proxies = {
                        "http": "socks5h://127.0.0.1:9050",
                        "https": "socks5h://127.0.0.1:9050"
                    }

                # Set realistic browser headers to avoid bot detection
                session.headers.update({
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                                  "Chrome/120.0.0.0 Safari/537.36"
                })

                # Step 1: Visit Yahoo Finance to establish session cookies
                session.get("https://fc.yahoo.com", timeout=30)

                # Step 2: Retrieve authentication crumb
                response = session.get(
                    "https://query1.finance.yahoo.com/v1/test/getcrumb",
                    timeout=30
                )

                if response.status_code == 200:
                    crumb = response.text
                    return session, crumb
                else:
                    print(f"Failed to get crumb: HTTP {response.status_code}")

            except Exception as e:
                print(f"Setup error on attempt {attempt + 1}: {e}")

            # Wait before retry (exponential backoff: 1s, 2s, 4s)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

        # All retries exhausted
        raise SessionSetupError("Failed to setup session after all retries")

    def data_v10(self, ticker, session, crumb, full_access=False):
        """
        Fetch fundamental financial data from Yahoo Finance v10 API.

        This endpoint provides company fundamentals, financial statements, and profile data.

        Args:
            ticker (str): Stock ticker symbol (e.g., "AAPL", "TCS.NS")
            session: Authenticated requests.Session object
            crumb (str): Yahoo Finance authentication token
            full_access (bool): If True, fetch comprehensive financial statements;
                               if False, fetch only key metrics (faster)

        Returns:
            dict or None: Financial data dictionary containing:
                - Industry, Sector, Website (from assetProfile)
                - Description (longBusinessSummary with fallback to summaryProfile)
                - Key Financial Metrics: margins, growth rates, returns
                - Balance Sheet Items: cash, debt, ratios
                Returns None if request fails or ticker is invalid

        Data Structure (full_access=False):
            {
                "Industry": str,
                "Sector": str,
                "Website": str,
                "Description": str,
                "Target Mean Price": float,
                "Recommendation": str,
                "Number of Analyst Opinions": int,
                "Profit Margins": float,
                "Gross Margins": float,
                ... (30+ financial metrics)
            }

        Implementation Notes:
            - Uses fallback logic: assetProfile -> summaryProfile for description
            - Safely extracts nested .get('raw') values to avoid KeyErrors
            - Full access mode retrieves complete financial statements (slower)
        """
        try:
            if full_access:
                # Comprehensive data: income statements, balance sheets, cash flows
                modules = [
                    "financialData", "incomeStatementHistory", "quarterlyIncomeStatementHistory",
                    "balanceSheetHistory", "quarterlyBalanceSheetHistory",
                    "cashflowStatementHistory", "quarterlyCashflowStatementHistory",
                    "assetProfile", "summaryProfile"
                ]
                modules_string = ",".join(modules)
                url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}?modules={modules_string}&crumb={crumb}"
            else:
                # Quick mode: just key metrics and profile
                url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}?modules=assetProfile,financialData,summaryProfile&crumb={crumb}"

            response = session.get(url, timeout=15)
            if response.status_code != 200:
                return None

            data = response.json()
            result = data.get('quoteSummary', {}).get('result')

            if not result:
                return None

            data_content = result[0]

            # If full access mode, inject description fallback and return raw data
            if full_access:
                if 'assetProfile' in data_content:
                    desc = data_content['assetProfile'].get('longBusinessSummary')
                    if not desc and 'summaryProfile' in data_content:
                        data_content['assetProfile']['longBusinessSummary'] = \
                            data_content['summaryProfile'].get('longBusinessSummary')
                return data_content

            # Extract key sections
            shortpath_v10 = data_content.get('financialData', {})
            raw_profile = data_content.get('assetProfile', {})
            summary_profile = data_content.get('summaryProfile', {})

            # --- ROBUST DESCRIPTION EXTRACTION ---
            # Try assetProfile first, then summaryProfile, then default to 'N/A'
            description = raw_profile.get('longBusinessSummary')
            if not description:
                description = summary_profile.get('longBusinessSummary')
            if not description:
                description = 'N/A'
            # -------------------------------------

            # Build structured financial data dictionary
            financial_data = {
                # Company Profile
                "Industry": raw_profile.get('industry', 'N/A'),
                "Sector": raw_profile.get('sector', 'N/A'),
                "Website": raw_profile.get('website', 'N/A'),
                "Description": description,

                # Analyst Opinions
                "Target Mean Price": shortpath_v10.get('targetMeanPrice', {}).get('raw'),
                "Recommendation": shortpath_v10.get('recommendationKey'),
                "Number of Analyst Opinions": shortpath_v10.get('numberOfAnalystOpinions', {}).get('raw'),

                # Profitability Margins
                "Profit Margins": shortpath_v10.get('profitMargins', {}).get('raw'),
                "Gross Margins": shortpath_v10.get('grossMargins', {}).get('raw'),
                "Operating Margins": shortpath_v10.get('operatingMargins', {}).get('raw'),
                "EBITDA Margins": shortpath_v10.get('ebitdaMargins', {}).get('raw'),

                # Growth Metrics
                "Revenue Growth": shortpath_v10.get('revenueGrowth', {}).get('raw'),
                "Earnings Growth": shortpath_v10.get('earningsGrowth', {}).get('raw'),

                # Returns
                "Return on Equity": shortpath_v10.get('returnOnEquity', {}).get('raw'),
                "Return on Assets": shortpath_v10.get('returnOnAssets', {}).get('raw'),

                # Balance Sheet
                "Total Cash": shortpath_v10.get('totalCash', {}).get('raw'),
                "Total Debt": shortpath_v10.get('totalDebt', {}).get('raw'),
                "Debt to Equity": shortpath_v10.get('debtToEquity', {}).get('raw'),
                "Current Ratio": shortpath_v10.get('currentRatio', {}).get('raw'),

                # Cash Flow
                "Free Cash Flow": shortpath_v10.get('freeCashflow', {}).get('raw'),

                # Per Share Metrics
                "Revenue Per Share": shortpath_v10.get('revenuePerShare', {}).get('raw'),
                "Total Cash Per Share": shortpath_v10.get('totalCashPerShare', {}).get('raw')
            }
            return financial_data

        except Exception as e:
            print(f"Error v10: {e}")
            return None

    def data_v8(self, ticker, session, time_range="1d", interval="1d"):
        """
        Fetch historical OHLCV price data from Yahoo Finance v8 API.

        This endpoint provides time-series data for charting and technical analysis.
        IMPROVED VERSION: More lenient with international stocks that may have
        incomplete data or unusual formatting.

        Args:
            ticker (str): Stock ticker symbol
            session: Authenticated requests.Session object
            time_range (str): Time period for historical data
                Options: "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"
            interval (str): Candle interval
                Options: "1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo"

        Returns:
            dict or None: Historical data dictionary with lists of equal length:
                {
                    "TimeStamp": [unix_timestamps],
                    "Close": [close_prices],
                    "Open": [open_prices],
                    "High": [high_prices],
                    "Low": [low_prices],
                    "Volume": [volumes],
                    "AdjClose": [adjusted_close_prices]
                }
                Returns None if no valid data points found

        Implementation Notes:
            - Filters out None/invalid data points automatically
            - Falls back to Close price if OHLV data is missing
            - Handles international stocks with incomplete data gracefully
            - Uses adjusted close for dividend/split adjustments
            - Validates at least one valid data point exists before returning

        Edge Cases Handled:
            - Missing OHLV data -> uses Close as fallback
            - Missing AdjClose -> uses Close
            - Missing Volume -> uses 0
            - Invalid prices (None, <= 0) -> skipped
        """
        try:
            # Build URL based on time range format
            if time_range == "max":
                # For 'max', use explicit period1/period2 (from epoch to now)
                period1 = 0
                period2 = int(time.time())
                url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?period1={period1}&period2={period2}&interval={interval}&events=history&includeAdjustedClose=true"
            else:
                # For relative ranges, use the 'range' parameter
                url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?range={time_range}&interval={interval}&events=history&includeAdjustedClose=true"

            # Make API request
            response = session.get(url, timeout=15)
            if response.status_code != 200:
                return None

            data = response.json()

            # Check for API-level errors
            if data.get("chart", {}).get("error"):
                return None

            if not data.get("chart", {}).get("result"):
                return None

            # Navigate to the data payload
            shortpath = data["chart"]["result"][0]

            # MORE LENIENT APPROACH: Check if we have any valid data instead of failing immediately
            raw_timestamps = shortpath.get("timestamp")

            # If no timestamps at all, then we truly can't proceed
            if not raw_timestamps or len(raw_timestamps) == 0:
                return None

            # Extract price/volume data safely (might be None or empty)
            quote_data = shortpath.get("indicators", {}).get("quote", [{}])[0]

            # Helper function to safely get list or empty list
            def get_col(name):
                """Safely extract a column from quote_data, return empty list if missing"""
                return quote_data.get(name, [])

            # Get all data columns (may contain None values or be shorter than timestamps)
            raw_close = get_col("close")
            raw_open = get_col("open")
            raw_high = get_col("high")
            raw_low = get_col("low")
            raw_volume = get_col("volume")

            # Adjusted close is in a separate structure
            adj_close_data = shortpath.get("indicators", {}).get("adjclose", [{}])[0].get("adjclose", [])

            # CRITICAL FIX: Filter out None/invalid data but don't fail if some data is missing
            # We build a list of clean data points, using Close as fallback for OHLV
            clean_data = []
            for i in range(len(raw_timestamps)):
                # We need at least a timestamp and close price to be valid
                if i < len(raw_close) and raw_close[i] is not None and raw_close[i] > 0:
                    clean_data.append({
                        "timestamp": raw_timestamps[i],
                        "close": raw_close[i],
                        # Use Close as fallback if Open/High/Low are missing or None
                        "open": raw_open[i] if i < len(raw_open) and raw_open[i] is not None else raw_close[i],
                        "high": raw_high[i] if i < len(raw_high) and raw_high[i] is not None else raw_close[i],
                        "low": raw_low[i] if i < len(raw_low) and raw_low[i] is not None else raw_close[i],
                        "volume": raw_volume[i] if i < len(raw_volume) and raw_volume[i] is not None else 0,
                        "adjclose": adj_close_data[i] if i < len(adj_close_data) and adj_close_data[i] is not None else
                        raw_close[i]
                    })

            # If we have no valid data after filtering, return None
            if len(clean_data) == 0:
                return None

            # Build the return dictionary with clean data (all lists have equal length)
            historical_data = {
                "TimeStamp": [d["timestamp"] for d in clean_data],
                "Close": [d["close"] for d in clean_data],
                "Open": [d["open"] for d in clean_data],
                "High": [d["high"] for d in clean_data],
                "Low": [d["low"] for d in clean_data],
                "Volume": [d["volume"] for d in clean_data],
                "AdjClose": [d["adjclose"] for d in clean_data]
            }

            return historical_data

        except Exception as e:
            return None

    def data_v7(self, ticker, session, crumb):
        """
        Fetch real-time market snapshot from Yahoo Finance v7 API.

        This endpoint provides current market data, intraday statistics, and key ratios.

        Args:
            ticker (str): Stock ticker symbol
            session: Authenticated requests.Session object
            crumb (str): Yahoo Finance authentication token

        Returns:
            dict or None: Current market data dictionary containing:
                - Symbol and Name
                - Current Price, Open, Previous Close
                - Day High/Low Range
                - Volume (current and averages)
                - Moving Averages (50-day, 200-day)
                - 52-Week High/Low
                - Valuation Ratios (P/E, Price-to-Book)
                - Market Cap, EPS
                Returns None if request fails

        Use Cases:
            - Real-time price display
            - Intraday trading statistics
            - Quick fundamental ratios
            - Market cap and valuation metrics
        """
        try:
            url = f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={ticker}&crumb={crumb}"

            response = session.get(url, timeout=15)
            if response.status_code != 200:
                return None

            data = response.json()
            if not data.get('quoteResponse', {}).get('result'):
                return None

            # Extract the quote data (first result)
            shortpath_v7 = data['quoteResponse']['result'][0]

            # Build structured snapshot dictionary
            snapshot_data = {
                # Basic Info
                "Symbol": shortpath_v7.get('symbol'),
                "Name": shortpath_v7.get('longName'),

                # Current Trading Data
                "Current Price": shortpath_v7.get('regularMarketPrice'),
                "Open": shortpath_v7.get('regularMarketOpen'),
                "Prev Close": shortpath_v7.get('regularMarketPreviousClose'),
                "Day High": shortpath_v7.get('regularMarketDayHigh'),
                "Day Low": shortpath_v7.get('regularMarketDayLow'),

                # Volume Statistics
                "Volume": shortpath_v7.get('regularMarketVolume'),
                "Avg Volume (3M)": shortpath_v7.get('averageDailyVolume3Month'),
                "Avg Volume (10D)": shortpath_v7.get('averageDailyVolume10Day'),

                # Technical Indicators
                "50 Day Avg": shortpath_v7.get('fiftyDayAverage'),
                "200 Day Avg": shortpath_v7.get('twoHundredDayAverage'),
                "52W High": shortpath_v7.get('fiftyTwoWeekHigh'),
                "52W Low": shortpath_v7.get('fiftyTwoWeekLow'),

                # Valuation Metrics
                "Trailing PE": shortpath_v7.get('trailingPE'),
                "Forward PE": shortpath_v7.get('forwardPE'),
                "Market Cap": shortpath_v7.get('marketCap'),
                "Price to Book": shortpath_v7.get('priceToBook'),
                "EPS (TTM)": shortpath_v7.get('epsTrailingTwelveMonths')
            }

            return snapshot_data

        except Exception as e:
            print(f"Error v7: {e}")
            return None

    def check_proxy_ip(self, use_proxy=True):
        """
        Verify the external IP address to confirm proxy status.

        This method calls ipify.org to determine the public IP address
        visible to external services. Useful for verifying Tor proxy is active.

        Args:
            use_proxy (bool): If True, check IP through proxy; if False, check direct IP

        Returns:
            str or None: External IP address as string, or None if check fails

        Use Case:
            >>> scraper = YahooScraper()
            >>> direct_ip = scraper.check_proxy_ip(use_proxy=False)
            >>> proxy_ip = scraper.check_proxy_ip(use_proxy=True)
            >>> if direct_ip != proxy_ip:
            >>>     print("Proxy is working!")
        """
        try:
            session = requests.Session()

            if use_proxy:
                # Route through Tor SOCKS5 proxy
                session.proxies = {
                    "http": "socks5h://127.0.0.1:9050",
                    "https": "socks5h://127.0.0.1:9050"
                }

            # Call ipify API to get external IP
            response = session.get("https://api.ipify.org?format=json", timeout=15)
            if response.status_code == 200:
                return response.json().get("ip")

        except Exception as e:
            print(f"Error checking IP: {e}")
            return None

        return None

    def v8_formatter(self, ticker_data):
        """
        Convert v8 API response into a pandas DataFrame for analysis.

        This method transforms the raw dictionary output from data_v8() into
        a clean DataFrame indexed by date, ready for technical analysis.

        Args:
            ticker_data (dict): Output from scrape() method containing 'v8' key

        Returns:
            pandas.DataFrame or None: DataFrame with columns:
                - Date (index): Trading date
                - Close: Closing price
                - Open: Opening price
                - High: Highest price
                - Low: Lowest price
                - AdjClose: Adjusted closing price
                - Volume: Trading volume
                Returns None if formatting fails

        Transformations Applied:
            1. Convert Unix timestamps to datetime dates
            2. Set Date as DataFrame index
            3. Convert price columns to float32 for memory efficiency

        Example:
            >>> data = scraper.scrape("AAPL", v8=True, time_range="1mo")
            >>> df = scraper.v8_formatter(data)
            >>> print(df.tail())
        """
        try:
            if not ticker_data.get('v8'):
                raise Exception("Historical Data Not Present in ticker_data")

            # Convert dictionary to DataFrame
            data = pd.DataFrame(ticker_data.get('v8'))

            # Convert Unix timestamps to readable dates
            data["TimeStamp"] = pd.to_datetime(data["TimeStamp"], unit="s")
            data["TimeStamp"] = data["TimeStamp"].dt.date

            # Rename and set index
            data.rename(columns={"TimeStamp": "Date"}, inplace=True)
            data.set_index("Date", inplace=True)

            # Convert price columns to float32 for efficiency
            cols = ["Close", "Open", "High", "Low", "AdjClose"]
            for col in cols:
                data[col] = data[col].astype("float32")

            return data

        except Exception as e:
            print(f"Exception in v8_formatter: {e}")
            return None

    def scrape(self, ticker, ip_address=None, time_range="1d", interval="1d",
               use_proxy=False, v10=False, v8=False, v7=False,
               v10_full_access=False, max_retries=3):
        """
        Main scraping method - fetches data from multiple Yahoo Finance endpoints.

        This is the primary interface for the YahooScraper class. It establishes
        a session and fetches data from the requested API endpoints.

        Args:
            ticker (str): Stock ticker symbol (e.g., "AAPL", "GOOGL", "TCS.NS")
            ip_address (str, optional): Your direct IP for proxy verification
            time_range (str): Historical data range for v8 (default: "1d")
                Options: "1d", "5d", "1mo", "3mo", "6mo", "1y", "5y", "max"
            interval (str): Candle interval for v8 (default: "1d")
                Options: "1m", "5m", "15m", "1h", "1d", "1wk"
            use_proxy (bool): Route traffic through Tor proxy (default: False)
            v10 (bool): Fetch fundamental data (default: False)
            v8 (bool): Fetch historical price data (default: False)
            v7 (bool): Fetch real-time quote data (default: False)
            v10_full_access (bool): Get complete financial statements (default: False)
            max_retries (int): Session setup retry attempts (default: 3)

        Returns:
            dict or None: Dictionary containing requested data sections:
                {
                    "v10": {...},  # Fundamentals (if v10=True)
                    "v8": {...},   # Historical data (if v8=True)
                    "v7": {...}    # Real-time quote (if v7=True)
                }
                Returns None if session setup fails or all requests fail

        Example Usage:
            >>> scraper = YahooScraper()
            >>> 
            >>> # Get everything for a stock
            >>> data = scraper.scrape("AAPL", v7=True, v8=True, v10=True, time_range="1mo")
            >>> 
            >>> # Just get historical data
            >>> hist = scraper.scrape("TSLA", v8=True, time_range="1y", interval="1d")
            >>> 
            >>> # Use proxy for anonymity
            >>> data = scraper.scrape("GOOGL", v7=True, use_proxy=True)

        Side Effects:
            - Updates self.proxy attribute if ip_address is provided and use_proxy=True
        """
        result = {}

        try:
            # Step 1: Establish authenticated session with Yahoo Finance
            try:
                session, crumb = self._setup_session(use_proxy=use_proxy, max_retries=max_retries)
            except SessionSetupError as e:
                print(f"Session setup failed: {e}")
                return None

            # Step 2: Verify proxy status if requested
            if use_proxy and ip_address:
                current_ip = self.check_proxy_ip(use_proxy=True)
                if current_ip and current_ip != ip_address:
                    self.proxy = "Proxy Active"
                else:
                    self.proxy = "Proxy Inactive"

            # Step 3: Fetch requested data sections
            if v10:
                result["v10"] = self.data_v10(ticker, session, crumb, full_access=v10_full_access)

            if v8:
                result["v8"] = self.data_v8(ticker, session, time_range=time_range, interval=interval)

            if v7:
                result["v7"] = self.data_v7(ticker, session, crumb)

            return result

        except Exception as e:
            print(f"Scrape error for {ticker}: {e}")
            return None


# ============================================================================
# TEST / DEMO CODE
# ============================================================================

if __name__ == "__main__":
    """
    Demo script showing how to use YahooScraper.

    This example:
    1. Checks your direct IP address
    2. Fetches comprehensive data for an Indian stock (TCS.NS)
    3. Uses proxy for anonymity
    4. Displays results and timing
    """
    start = time.perf_counter()

    # Initialize scraper
    ys = YahooScraper()

    # Check your direct IP (without proxy)
    ip = ys.check_proxy_ip(use_proxy=False)
    print(f"My IP: {ip}")

    # Test with international stock (Indian market)
    print("\n=== Testing TCS.NS (Indian Stock) ===")
    s = ys.scrape(
        "TCS.NS",
        ip_address=ip,
        time_range="1mo",  # Last month of data
        use_proxy=True,  # Route through Tor
        v10=True,  # Get fundamentals
        v8=True,  # Get historical prices
        v7=True,  # Get current quote
        v10_full_access=False  # Just key metrics
    )

    end = time.perf_counter()

    # Display results
    print(f"\nProxy Status: {ys.proxy}")
    print(f"Time taken: {end - start:.2f} seconds")

    if s:
        print("\nv7 (Current Quote):", s.get("v7"))
        if s.get("v10"):
            print("\nDescription:", s.get("v10").get("Description"))
