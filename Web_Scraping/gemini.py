from google import genai
from google.genai import types
import json


class Gemini:

    def __init__(self):
        """
Gemini AI Integration Module

This module interfaces with Google's Gemini AI API to extract market intelligence
and contextual information about stocks that isn't available through traditional APIs.

Key Features:
- Identifies business partners and supply chain relationships
- Finds peer competitors in the same sector
- Determines appropriate market indices (sectoral and general)
- Classifies market regime (stable vs volatile)
- Returns structured JSON data for ML model consumption

Author: TradeAlchemy Team
"""

from google import genai
from google.genai import types
import json


class Gemini:
    """
    Interface for Gemini AI to provide market intelligence on stock tickers.

    This class uses large language models to extract contextual information
    that enhances stock predictions by considering ecosystem relationships.

    Attributes:
        None (stateless design for API calls)

    Use Cases:
        - Get competitors for relative strength analysis
        - Identify supply chain partners for correlation analysis
        - Determine if a stock is stable (blue-chip) or volatile (growth/speculative)
        - Find appropriate benchmark indices for performance comparison

    Example:
        >>> gemini = Gemini()
        >>> info = gemini.get_info("AAPL", api_key="your-key")
        >>> print(info['market_regime'])  # "stable" or "volatile"
        >>> print(info['peers'])  # [{"name": "Microsoft", "ticker": "MSFT"}, ...]
    """

    def __init__(self):
        """
        Initialize Gemini AI integration.

        Note: This is a stateless design. API key is passed per-call
        rather than stored as an instance variable for security.
        """
        pass

    def retrieve_data(self, ticker, api_key):
        """
        Call Gemini API to retrieve market intelligence for a stock ticker.

        This method constructs a detailed prompt for Gemini AI and requests
        structured JSON data about the stock's ecosystem.

        Args:
            ticker (str): Stock ticker symbol (e.g., "AAPL", "TCS.NS")
            api_key (str): Your Google Gemini API key

        Returns:
            str: Raw JSON string from Gemini containing:
                - partners: Top 3 business partners (clients/suppliers/alliances)
                - peers: Top 3 direct competitors
                - sectoral_index: Sector-specific market index (same country)
                - market_index: General market index (same country)
                - market_regime: "stable" or "volatile" classification

        Prompt Strategy:
            The prompt is engineered to:
            1. Request specific structured data (not general descriptions)
            2. Enforce Yahoo Finance ticker format for all symbols
            3. Require country-matched indices
            4. Return only JSON (no markdown, no explanations)
            5. Classify market regime based on company characteristics

        Implementation Notes:
            - Uses gemini-flash-latest model (fastest, optimized for structured output)
            - Forces JSON response with response_mime_type
            - Provides clear examples in prompt to guide output format

        Raises:
            May raise exceptions from genai client (network errors, invalid API key, etc.)
        """
        # Initialize Gemini client with API key
        client = genai.Client(api_key=api_key)

        # Construct detailed prompt for Gemini to analyze the stock
        # This prompt is carefully engineered to get consistent, structured output
        prompt = f"""
        Analyze the company with ticker symbol {ticker} and return a JSON object with FIVE sections:

        1. "partners": List the top 3 major business partners (Key Clients, Critical Suppliers, or Strategic Alliances).

        2. "peers": List the top 3 direct competitor/peer companies in the same market/sector.

        3. "sectoral_index": Provide the SECTOR-SPECIFIC stock market index from the SAME COUNTRY as {ticker}.
           - Examples: ^CNXIT (India IT), XLK (US Tech), XLV (US Healthcare).
           - If no specific index exists, leave as empty string "".

        4. "market_index": Provide the main general stock market index from the SAME COUNTRY as {ticker}.
           - Examples: ^GSPC (S&P 500), ^NSEI (Nifty 50), ^N225 (Nikkei 225).

        5. "market_regime": Classify the stock's price behavior into one of TWO categories:
           - "stable": Use for blue-chip companies, value stocks, low-beta entities, or established industry leaders with steady price action (e.g., JNJ, PG, KO, TCS.NS).
           - "volatile": Use for high-growth tech stocks, momentum stocks, speculative assets, or highly cyclical companies with large daily swings (e.g., TSLA, NVDA, coin-linked stocks).

        CRITICAL REQUIREMENTS:
        - ALL ticker symbols MUST be valid Yahoo Finance format (e.g., TCS.NS, AAPL, ^GSPC).
        - Use "Private" for non-public partners.
        - Verify the country of {ticker} and provide indices from that same country only.

        Return strictly a JSON object with this schema:
        {{
          "partners": [
            {{ 
                "name": "Company Name", 
                "role": "Role", 
                "ticker": "Ticker",
                "impact_reason": "Reason"
            }}
          ],
          "peers": [
            {{ "name": "Name", "ticker": "Ticker" }}
          ],
          "sectoral_index": "Ticker",
          "market_index": "Ticker",
          "market_regime": "stable" or "volatile"
        }}

        IMPORTANT: Return ONLY the JSON. No markdown, no code blocks, no explanations.
        """

        # Call Gemini API with the prompt and force JSON response
        response = client.models.generate_content(
            model="gemini-flash-latest",  # Fastest model, good for structured tasks
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"  # Force JSON response
            )
        )

        return response.text

    def format_info(self, response):
        """
        Parse and validate the JSON response from Gemini.

        This method converts the raw string response into a Python dictionary
        and handles potential parsing errors gracefully.

        Args:
            response (str): Raw JSON string from Gemini API

        Returns:
            dict or None: Parsed data structure containing:
                {
                    "partners": [...],
                    "peers": [...],
                    "sectoral_index": "...",
                    "market_index": "...",
                    "market_regime": "stable" or "volatile"
                }
                Returns None if JSON parsing fails

        Error Handling:
            - Catches json.JSONDecodeError for malformed JSON
            - Catches generic exceptions for unexpected errors
            - Prints error messages for debugging
            - Returns None on any failure (caller should handle this)
        """
        try:
            # Parse JSON string into Python dictionary
            data = json.loads(response)
            return data
        except json.JSONDecodeError as e:
            print(f"\n⚠️ Error parsing Gemini response: {e}")
            return None
        except Exception as e:
            print(f"\n⚠️ Unexpected error: {e}")
            return None

    def get_info(self, ticker, api_key):
        """
        High-level method to get complete market intelligence including regime.

        This is the main method for retrieving all contextual data about a stock.
        It handles the full pipeline: API call -> JSON parsing -> error handling.

        Args:
            ticker (str): Stock ticker symbol
            api_key (str): Your Google Gemini API key

        Returns:
            dict or None: Complete market intelligence data:
                {
                    "partners": [
                        {
                            "name": "Apple Inc",
                            "role": "Major Customer",
                            "ticker": "AAPL",
                            "impact_reason": "30% of revenue"
                        },
                        ...
                    ],
                    "peers": [
                        {"name": "Microsoft", "ticker": "MSFT"},
                        ...
                    ],
                    "sectoral_index": "XLK",
                    "market_index": "^GSPC",
                    "market_regime": "volatile"
                }
                Returns None if API call or parsing fails

        Example:
            >>> gemini = Gemini()
            >>> info = gemini.get_info("NVDA", "your-api-key")
            >>> if info:
            >>>     print(f"Regime: {info['market_regime']}")
            >>>     for peer in info['peers']:
            >>>         print(f"Competitor: {peer['name']} ({peer['ticker']})")
        """
        try:
            # Fetch data from Gemini API
            raw_response = self.retrieve_data(ticker, api_key)

            # Parse and return formatted data
            return self.format_info(raw_response)

        except Exception as e:
            print(f"⚠️ Gemini API error: {e}")
            return None

    def get_market_regime(self, ticker, api_key):
        """
        Flask-specific method: Get only market regime (stable or volatile).

        This convenience method extracts just the market regime classification
        from the full intelligence data. Used when you only need to know if
        a stock is stable or volatile without fetching competitors/partners.

        Args:
            ticker (str): Stock ticker symbol
            api_key (str): Your Google Gemini API key

        Returns:
            str: "stable" or "volatile"
                - "stable": Blue-chip, value stocks, low beta
                - "volatile": High-growth, momentum stocks, high beta
                Defaults to "volatile" if unable to determine

        Use Case:
            Used by ML models to adjust prediction thresholds:
            - Stable stocks: Look for smaller price movements (0.5% threshold)
            - Volatile stocks: Look for larger price movements (1.5% threshold)

        Example:
            >>> gemini = Gemini()
            >>> regime = gemini.get_market_regime("JNJ", api_key)
            >>> print(regime)  # "stable"
            >>> regime = gemini.get_market_regime("TSLA", api_key)
            >>> print(regime)  # "volatile"
        """
        try:
            # Get full info from Gemini
            info = self.get_info(ticker, api_key)

            if info and 'market_regime' in info:
                return info['market_regime']

            # Default to 'volatile' if unable to determine
            # This is a conservative choice - better to expect larger movements
            return 'volatile'

        except Exception as e:
            print(f"⚠️ Error getting market regime: {e}")
            return 'volatile'  # Default fallback

    def get_peers(self, ticker, api_key):
        """
        Flask-specific method: Get peers for comparison analysis.

        This convenience method extracts just the peer competitor list
        from the full intelligence data.

        Args:
            ticker (str): Stock ticker symbol
            api_key (str): Your Google Gemini API key

        Returns:
            list: List of peer dictionaries:
                [
                    {"name": "Microsoft", "ticker": "MSFT"},
                    {"name": "Google", "ticker": "GOOGL"},
                    {"name": "Amazon", "ticker": "AMZN"}
                ]
                Returns empty list [] if unable to retrieve peers

        Use Case:
            Used for relative strength analysis - comparing target stock's
            performance against its direct competitors.

        Example:
            >>> gemini = Gemini()
            >>> peers = gemini.get_peers("AAPL", api_key)
            >>> for peer in peers:
            >>>     print(f"Analyze {peer['ticker']} vs AAPL")
        """
        try:
            # Get full info from Gemini
            info = self.get_info(ticker, api_key)

            if info and 'peers' in info:
                return info['peers']

            return []

        except Exception as e:
            print(f"⚠️ Error getting peers: {e}")
            return []

    def get_partners(self, ticker, api_key):
        """
        Flask-specific method: Get business partners/suppliers.

        This convenience method extracts just the business partner list
        from the full intelligence data.

        Args:
            ticker (str): Stock ticker symbol
            api_key (str): Your Google Gemini API key

        Returns:
            list: List of partner dictionaries:
                [
                    {
                        "name": "TSMC",
                        "role": "Chip Manufacturer",
                        "ticker": "TSM",
                        "impact_reason": "Produces all Apple chips"
                    },
                    ...
                ]
                Returns empty list [] if unable to retrieve partners

        Use Case:
            Used for supply chain risk analysis - if a key supplier's stock
            drops, it may impact the target company.

        Example:
            >>> gemini = Gemini()
            >>> partners = gemini.get_partners("AAPL", api_key)
            >>> for partner in partners:
            >>>     print(f"Monitor {partner['name']}: {partner['impact_reason']}")
        """
        try:
            # Get full info from Gemini
            info = self.get_info(ticker, api_key)

            if info and 'partners' in info:
                return info['partners']

            return []

        except Exception as e:
            print(f"⚠️ Error getting partners: {e}")
            return []


# ============================================================================
# TEST / DEMO CODE
# ============================================================================

if __name__ == "__main__":
    """
    Demo script showing how to use the Gemini class.
    
    This example:
    1. Sets up API key (you need to provide your own)
    2. Tests with Apple (AAPL) ticker
    3. Displays all retrieved market intelligence
    """
    # You need to set your Gemini API key here
    # Get it from: https://makersuite.google.com/app/apikey
    GEMINI_API_KEY = "your-gemini-api-key-here"

    # Initialize Gemini
    gemini = Gemini()

    # Test with a stock ticker
    ticker = "AAPL"

    print(f"Testing Gemini AI for {ticker}...")
    print("=" * 60)

    # Get full market intelligence
    info = gemini.get_info(ticker, GEMINI_API_KEY)

    if info:
        print(f"\nMarket Regime: {info.get('market_regime')}")
        print(f"\nPeers: {info.get('peers')}")
        print(f"\nPartners: {info.get('partners')}")
        print(f"\nSector Index: {info.get('sectoral_index')}")
        print(f"\nMarket Index: {info.get('market_index')}")
    else:
        print("Failed to retrieve data from Gemini")
