"""
Stock Analyzer Module

This is the main orchestrator that combines all components:
- Web scraping (Yahoo Finance)
- AI intelligence (Gemini)
- Feature engineering (DataProcessor)
- ML predictions (LSTM)

This module provides a high-level interface for stock analysis with AI predictions.

Author: TradeAlchemy Team
"""

from typing import Dict, Optional
import yfinance as yf
import pandas as pd

# Import project modules
from Web_Scraping import YahooScraper
from Web_Scraping import Gemini
from Machine_Learning import MultiTimeframeLSTM
from Machine_Learning import FeatureCalculator


# ============================================================================
# MAIN STOCK ANALYZER CLASS
# ============================================================================

class StockAnalyzer:
    """
    High-level interface for comprehensive stock analysis with AI predictions.

    This class orchestrates the complete analysis pipeline:
    1. Fetch contextual data (competitors, partners) from Gemini AI
    2. Download price data for target stock and its ecosystem
    3. Engineer technical and contextual features
    4. Train LSTM model and generate prediction
    5. Return structured results for API/UI display

    Components:
        - scraper: Yahoo Finance data fetcher
        - gemini: Gemini AI for market intelligence
        - feature_calc: Technical indicator calculator
        - lstm: Neural network prediction model

    Attributes:
        scraper (YahooScraper): Web scraper instance
        gemini (Gemini): AI intelligence instance
        feature_calc (FeatureCalculator): Feature engineering instance
        lstm (MultiTimeframeLSTM): LSTM model instance
        gemini_api_key (str): Stored API key for Gemini

    Example:
        >>> analyzer = StockAnalyzer(gemini_api_key="your-key")
        >>> result = analyzer.ai_prediction("AAPL")
        >>>
        >>> if result:
        >>>     print(f"Direction: {result['direction']}")
        >>>     print(f"Confidence: {result['confidence']:.1%}")
        >>>     print(f"Current Price: ${result['current_price']:.2f}")
    """

    def __init__(self, gemini_api_key=None):
        """
        Initialize all analysis components.

        Args:
            gemini_api_key (str, optional): Google Gemini API key
                If not provided here, must be passed to analysis methods

        Side Effects:
            - Initializes YahooScraper (no network calls yet)
            - Initializes Gemini client
            - Initializes FeatureCalculator
            - Initializes LSTM model (not trained yet)
        """
        self.scraper = YahooScraper()
        self.gemini = Gemini()
        self.feature_calc = FeatureCalculator()
        self.lstm = MultiTimeframeLSTM()
        self.gemini_api_key = gemini_api_key

    def ai_prediction(self, ticker: str, gemini_api_key: str = None) -> Optional[Dict]:
        """
        Main analysis method - orchestrates complete AI prediction pipeline.

        This method performs ecosystem-aware stock analysis:
        1. Get context: competitors, partners, market regime
        2. Fetch data: target stock + all related entities
        3. Engineer features: technical indicators + relative strength
        4. Train LSTM: with advanced weighting
        5. Generate prediction: probability + confidence interpretation

        Args:
            ticker (str): Stock ticker symbol (e.g., "AAPL", "TCS.NS")
            gemini_api_key (str, optional): Gemini API key (overrides instance key)

        Returns:
            dict or None: Prediction results containing:
                {
                    "ticker": "AAPL",
                    "direction": "UP" or "DOWN",
                    "probability": 0.657,  # Raw LSTM output (0-1)
                    "confidence": 0.657,   # Actual confidence shown to user
                    "regime": "volatile" or "stable",
                    "accuracy": 73.5,      # Model's test set accuracy
                    "atr": 2.45,          # Current volatility (ATR)
                    "current_price": 178.50
                }
                Returns None if analysis fails at any stage

        Pipeline Details:

            STAGE 1: CONTEXT EXTRACTION (Gemini AI)
            - Identifies top 3 competitors (peers)
            - Identifies top 3 business partners
            - Classifies market regime (stable/volatile)
            - Determines appropriate benchmark indices

            STAGE 2: ECOSYSTEM DATA COLLECTION
            - Downloads 5 years of daily data for:
              * Target stock
              * All identified peers
              * All identified partners
            - Falls back to yfinance if custom scraper fails
            - Handles international stocks (.NS, .SZ, .KS, etc.)

            STAGE 3: FEATURE ENGINEERING
            - Calculates technical indicators (RSI, MACD, ATR, MA)
            - Computes relative strength vs peers
            - Computes relative strength vs partners
            - Generates binary target (significant move or not)

            STAGE 4: LSTM TRAINING & PREDICTION
            - Trains on 85% of data (chronological split)
            - Tests on 15% holdout set
            - Uses time-weighted + class-balanced sampling
            - Predicts tomorrow's volatility probability

            STAGE 5: INTERPRETATION
            - Converts probability to directional prediction
            - Calculates user-facing confidence
            - Packages results with current market data

        Confidence Calculation:
            - If prob > 0.5: Direction = "DOWN", Confidence = prob
            - If prob < 0.5: Direction = "UP", Confidence = 1 - prob

            This ensures confidence always represents certainty in the
            stated direction, not the raw model output.

        Error Handling:
            - Prints warnings for non-critical failures (e.g., Gemini timeout)
            - Returns None for critical failures (e.g., no data for target)
            - Uses fallbacks where possible (yfinance if scraper fails)

        Example Flow:
            >>> # Analyze Apple (US tech stock, volatile regime)
            >>> result = analyzer.ai_prediction("AAPL", api_key="your-key")
            >>>
            >>> if result:
            >>>     print(f"AAPL is in {result['regime']} regime")
            >>>     print(f"Prediction: {result['direction']} with {result['confidence']:.1%} confidence")
            >>>     print(f"Current ATR: ${result['atr']:.2f} (volatility)")
        """
        # Use provided API key or fall back to instance key
        api_key = gemini_api_key or self.gemini_api_key

        print(f"\n🤖 Running AI analysis for {ticker}...")

        try:
            # ================================================================
            # STAGE 1: GET CONTEXT (Peers & Partners)
            # ================================================================
            context = {
                'peers': [],
                'partners': [],
                'market_regime': 'volatile'  # Default fallback
            }

            if api_key:
                print("📊 Fetching market intelligence from Gemini...")
                gemini_data = self.gemini.get_info(ticker, api_key)

                if gemini_data:
                    # Extract top 3 peers and partners
                    context['peers'] = gemini_data.get('peers', [])[:3]
                    context['partners'] = gemini_data.get('partners', [])[:3]
                    context['market_regime'] = gemini_data.get('market_regime', 'volatile')

                    print(f"✓ Found {len(context['peers'])} peers, {len(context['partners'])} partners")
                    print(f"✓ Market regime: {context['market_regime']}")

            # ================================================================
            # STAGE 2: DATA FETCHING (Ecosystem)
            # ================================================================
            # Build list of all tickers to fetch (target + peers + partners)
            tickers_to_fetch = [ticker]

            # Add peer tickers
            for p in context['peers']:
                if isinstance(p, dict) and 'ticker' in p:
                    tickers_to_fetch.append(p['ticker'])
                elif isinstance(p, str):
                    tickers_to_fetch.append(p)

            # Add partner tickers
            for p in context['partners']:
                if isinstance(p, dict) and 'ticker' in p:
                    tickers_to_fetch.append(p['ticker'])
                elif isinstance(p, str):
                    tickers_to_fetch.append(p)

            # Remove duplicates
            tickers_to_fetch = list(set(tickers_to_fetch))
            print(f"📊 Fetching data for ecosystem: {tickers_to_fetch}")

            # Dictionary to store DataFrame for each ticker
            market_map = {}

            for t in tickers_to_fetch:
                try:
                    formatted = None

                    # Try 1: Custom scraper first (faster, more reliable)
                    df = self.scraper.scrape(t, v8=True, time_range="5y")
                    if df and 'v8' in df:
                        formatted = self.scraper.v8_formatter(df)

                    # Try 2: Fallback to yfinance if scraper fails
                    # This fixes issues with international stocks (.SZ, .KS, etc.)
                    if formatted is None or formatted.empty:
                        print(f"⚠️ Scraper failed for {t}, trying yfinance...")
                        yf_data = yf.download(t, period="5y", progress=False)

                        if not yf_data.empty:
                            # Standardize column name for DataProcessor
                            if 'Adj Close' in yf_data.columns:
                                yf_data['AdjClose'] = yf_data['Adj Close']
                            else:
                                yf_data['AdjClose'] = yf_data['Close']
                            formatted = yf_data

                    # Store if we got valid data
                    if formatted is not None and not formatted.empty:
                        market_map[t] = formatted
                        print(f"✓ Fetched {len(formatted)} days for {t}")

                except Exception as e:
                    print(f"❌ Failed to fetch data for {t}: {e}")

            # Critical check: Do we have data for the target stock?
            if ticker not in market_map:
                print("❌ Main ticker data not found.")
                return None

            # ================================================================
            # STAGE 3: FEATURE ENGINEERING
            # ================================================================
            print("🔧 Engineering features...")

            # Calculate technical indicators for target stock
            df_main = self.feature_calc.calculate_features(
                market_map[ticker],
                regime=context['market_regime']
            )

            if df_main is None:
                return None

            # Build simplified context for relative strength calculation
            # (just lists of ticker symbols)
            simple_context = {
                'peers': [
                    t for t in tickers_to_fetch
                    if t in context['peers'] or any(
                        p.get('ticker') == t for p in context['peers']
                        if isinstance(p, dict)
                    )
                ],
                'partners': [
                    t for t in tickers_to_fetch
                    if t in context['partners'] or any(
                        p.get('ticker') == t for p in context['partners']
                        if isinstance(p, dict)
                    )
                ]
            }

            # Add context features (relative strength vs peers/partners)
            df_final = self.feature_calc.add_context_features(
                df_main,
                market_map,
                simple_context
            )

            print(f"✓ Generated {len(df_final)} training samples")

            # ================================================================
            # STAGE 4: PREDICTION (LSTM Training)
            # ================================================================
            print("🧠 Training LSTM model...")

            result = self.lstm.train_and_predict(df_final)
            if not result:
                return None

            prob, acc, _ = result

            # ================================================================
            # STAGE 5: INTERPRETATION
            # ================================================================
            # The model outputs a probability of a significant move
            # We interpret this as:
            # - High probability (>0.5) → Bearish (expect DOWN move)
            # - Low probability (<0.5) → Bullish (expect UP move)

            is_risky = prob > 0.5

            # FIXED MATH: Confidence should be probability of the stated direction
            # If we say "DOWN", confidence = probability of down move
            # If we say "UP", confidence = probability of up move (1 - prob)
            actual_confidence = float(prob) if is_risky else float(1.0 - prob)

            # Package results
            return {
                'ticker': ticker,
                'direction': "DOWN" if is_risky else "UP",
                'probability': float(prob),  # Raw model output
                'confidence': float(actual_confidence),  # User-facing confidence
                'regime': context['market_regime'],
                'accuracy': float(acc * 100),  # Convert to percentage
                'atr': float(df_final['ATR'].iloc[-1]),
                'current_price': float(df_final['AdjClose'].iloc[-1])
            }

        except Exception as e:
            print(f"Analyzer Error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def analyze_for_api(self, ticker: str):
        """
        Flask API wrapper for ai_prediction().

        This method exists for API naming consistency and can be extended
        with API-specific logic (e.g., rate limiting, caching, logging).

        Args:
            ticker (str): Stock ticker symbol

        Returns:
            dict or None: Same as ai_prediction()

        Example:
            >>> analyzer = StockAnalyzer(gemini_api_key="key")
            >>> result = analyzer.analyze_for_api("TSLA")
            >>> return jsonify(result)
        """
        return self.ai_prediction(ticker)

    def get_fundamentals(self, ticker: str):
        """
        Fetch fundamental financial data (balance sheet, income statement).

        This method wraps the v10 endpoint from YahooScraper to provide
        company fundamentals in a consistent format.

        Args:
            ticker (str): Stock ticker symbol

        Returns:
            dict: Response with success flag and data
                {
                    "success": True,
                    "data": {
                        "Industry": "Technology",
                        "Sector": "Software",
                        "Description": "...",
                        "Profit Margins": 0.25,
                        "Debt to Equity": 1.2,
                        ... (30+ financial metrics)
                    }
                }
                OR
                {
                    "success": False
                }

        Use Case:
            Display company fundamentals on stock detail page
        """
        data = self.scraper.scrape(ticker, v10=True)

        if data and 'v10' in data:
            return {'success': True, 'data': data['v10']}
        else:
            return {'success': False}

    def get_quote(self, ticker: str):
        """
        Fetch real-time market quote (current price, volume, P/E ratio).

        This method wraps the v7 endpoint from YahooScraper to provide
        current market data in a consistent format.

        Args:
            ticker (str): Stock ticker symbol

        Returns:
            dict: Response with success flag and data
                {
                    "success": True,
                    "data": {
                        "Symbol": "AAPL",
                        "Name": "Apple Inc.",
                        "Current Price": 178.50,
                        "Open": 176.80,
                        "Day High": 179.20,
                        "Day Low": 176.50,
                        "Volume": 52000000,
                        "Market Cap": 2800000000000,
                        ... (20+ market metrics)
                    }
                }
                OR
                {
                    "success": False
                }

        Use Case:
            Display real-time price and statistics on stock cards
        """
        data = self.scraper.scrape(ticker, v7=True)

        if data and 'v7' in data:
            return {'success': True, 'data': data['v7']}
        else:
            return {'success': False}