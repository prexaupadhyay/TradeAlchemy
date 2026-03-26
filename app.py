"""
TradeAlchemy Flask Application

This is the main Flask application file that ties together all modules:
- Database management
- Authentication
- Stock analysis (scraping + ML)
- Watchlist management
- RESTful API endpoints
- Static file serving

The app provides both a web interface and API endpoints for:
- User authentication (sign up, sign in, password management)
- Stock data retrieval (fundamentals, quotes, historical data)
- AI predictions (LSTM-based volatility predictions)
- Watchlist management (add/remove stocks, get prices)

Author: TradeAlchemy Team
"""

import os
import secrets
import math
from functools import wraps
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_from_directory
import yfinance as yf
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env file (Gemini API key, etc.)
load_dotenv()

# Import project modules
from Database import DatabaseManager
from AccountServices import AuthManager
from AccountServices import WatchlistManager
from Machine_Learning import StockAnalyzer

# ============================================================================
# FLASK APP INITIALIZATION
# ============================================================================

# Create Flask application instance
app = Flask(__name__)

# Generate a random secret key for session management
# This key is used to encrypt session cookies
app.secret_key = secrets.token_hex(16)  # 32-character hex string

# ============================================================================
# INITIALIZE SYSTEM MODULES
# ============================================================================

try:
    # Initialize database manager (creates tables if not exist)
    db = DatabaseManager()

    # Initialize authentication manager (handles user accounts)
    auth = AuthManager(db)

    # Get Gemini API key from environment variables
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

    # Initialize stock analyzer (scraping + ML predictions)
    analyzer = StockAnalyzer(gemini_api_key=GEMINI_API_KEY)

    print("✓ System modules initialized successfully.")
except Exception as e:
    print(f"❌ Initialization Error: {e}")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def clean_data(data):
    """
    Recursively replace NaN/Infinity with None for valid JSON serialization.

    Python's math.nan and math.inf are not valid JSON. This function
    recursively traverses dictionaries and lists, replacing any NaN or
    Infinity values with None.

    Args:
        data: Any Python object (dict, list, float, etc.)

    Returns:
        Same structure with NaN/Inf replaced by None

    Use Case:
        Stock data from yfinance often contains NaN values. JSON
        serialization will fail unless these are converted to null.

    Example:
        >>> data = {'price': float('nan'), 'volume': 1000000}
        >>> clean_data(data)
        {'price': None, 'volume': 1000000}
    """
    if isinstance(data, dict):
        # Recursively clean dictionary values
        return {k: clean_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        # Recursively clean list elements
        return [clean_data(v) for v in data]
    elif isinstance(data, float):
        # Check if float is NaN or Infinity
        if math.isnan(data) or math.isinf(data):
            return None
        return data
    return data


# ============================================================================
# AUTHENTICATION DECORATOR
# ============================================================================

def login_required(f):
    """
    Decorator to protect routes that require authentication.

    This decorator checks if the user has a valid session. If not,
    redirects to the landing page.

    Args:
        f: The route function to protect

    Returns:
        Decorated function that checks authentication

    Usage:
        @app.route('/dashboard')
        @login_required
        def dashboard():
            return "Welcome to your dashboard!"

    How It Works:
        1. Checks if 'user_id' exists in session
        2. If yes: Allows access to the route
        3. If no: Redirects to landing page
    """

    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            # User not logged in - redirect to landing
            return redirect(url_for('landing'))
        # User authenticated - proceed to route
        return f(*args, **kwargs)

    return decorated_function


# ============================================================================
# PAGE ROUTES (HTML Pages)
# ============================================================================

@app.route('/')
def landing():
    """
    Landing page route (login/signup page).

    If user is already logged in, redirects to dashboard.
    Otherwise, serves the landing.html page.

    Returns:
        - Redirect to dashboard if logged in
        - landing.html if not logged in
    """
    if 'user_id' in session:
        # Already logged in - go to dashboard
        return redirect(url_for('dashboard'))
    return send_from_directory('templates', 'landing.html')


@app.route('/dashboard')
@app.route('/dashboard.html')
@login_required
def dashboard():
    """
    Main dashboard page (search interface).

    Protected route - requires authentication.

    Returns:
        dashboard.html
    """
    return send_from_directory('templates', 'dashboard.html')


@app.route('/search')
@app.route('/search.html')
@login_required
def search_page():
    """
    Stock search results page (general mode).

    Displays fundamentals, charts, and basic analysis.
    Protected route - requires authentication.

    Returns:
        search.html
    """
    return send_from_directory('templates', 'search.html')


@app.route('/ai_prediction')
@app.route('/ai_prediction.html')
@login_required
def ai_prediction_page():
    """
    AI prediction page (LSTM analysis mode).

    Displays ML-powered volatility predictions with confidence scores.
    Protected route - requires authentication.

    Returns:
        ai_prediction.html
    """
    return send_from_directory('templates', 'ai_prediction.html')


@app.route('/watchlist')
@app.route('/watchlist.html')
@login_required
def watchlist_page():
    """
    User's watchlist page (tracked stocks).

    Displays user's saved stocks with real-time prices.
    Protected route - requires authentication.

    Returns:
        watchlist.html
    """
    return send_from_directory('templates', 'watchlist.html')


@app.route('/account')
@app.route('/account.html')
@login_required
def account_page():
    """
    Account settings page (profile management).

    Allows user to change password, email, etc.
    Protected route - requires authentication.

    Returns:
        account.html
    """
    return send_from_directory('templates', 'account.html')


# --- EDUCATIONAL CONTENT PAGES (Public Access) ---

@app.route('/stock_market')
@app.route('/stock_market.html')
def stock_market_page():
    """
    Stock market education page.

    Educational content about stock market basics.
    Public access (no login required).

    Returns:
        stock_market.html
    """
    return send_from_directory('templates', 'stock_market.html')


@app.route('/ai_ml')
@app.route('/ai_ml.html')
def ai_ml_page():
    """
    AI/ML education page.

    Educational content about AI/ML in trading.
    Public access (no login required).

    Returns:
        ai_ml.html
    """
    return send_from_directory('templates', 'ai_ml.html')


# ============================================================================
# STATIC FILE ROUTES
# ============================================================================

@app.route('/style.css')
def serve_css():
    """
    Serve main CSS stylesheet.

    Returns:
        style.css from static/css directory
    """
    return send_from_directory('static/css', 'style.css')


@app.route('/static/images/<path:filename>')
def serve_images(filename):
    """
    Serve image files (logo, icons, etc.).

    Args:
        filename: Path to image file

    Returns:
        Image file from static/images directory
    """
    return send_from_directory('static/images', filename)


# ============================================================================
# AUTHENTICATION API ENDPOINTS
# ============================================================================

@app.route('/api/signup', methods=['POST'])
def signup():
    """
    Create new user account.

    Request Body:
        {
            "username": "john",
            "email": "john@example.com",
            "password": "SecurePass123"
        }

    Response:
        Success: {
            'success': True,
            'message': 'Account created! Check email for code.',
            'requires_verification': True
        }
        Failure: {
            'success': False,
            'message': 'Error message',
            'requires_verification': False
        }

    Side Effects:
        - Creates user in database (unverified)
        - Sends OTP verification email
    """
    data = request.json
    result = auth.sign_up(
        data.get('username'),
        data.get('email'),
        data.get('password')
    )
    return jsonify(result)


@app.route('/api/verify', methods=['POST'])
def verify_otp():
    """
    Verify email with OTP code.

    Request Body:
        {
            "email": "john@example.com",
            "otp": "1234"
        }

    Response:
        Success: {'success': True, 'message': 'Email verified successfully!'}
        Failure: {'success': False, 'message': 'Invalid or expired code'}

    Side Effects:
        - Activates user account (is_verified = 1)
        - Marks OTP as used
    """
    data = request.json
    result = auth.verify_email(
        data.get('email'),
        data.get('otp')
    )
    return jsonify(result)


@app.route('/api/login', methods=['POST'])
def login():
    """
    Authenticate user and create session.

    Request Body:
        {
            "login": "john@example.com",  // username or email
            "password": "SecurePass123"
        }

    Response:
        Success: {
            'success': True,
            'message': 'Login successful',
            'user_id': 123,
            'username': 'john'
        }
        Failure: {
            'success': False,
            'message': 'Invalid credentials'
        }

    Side Effects:
        - Creates session with user_id and username
        - Session cookie sent to client
    """
    data = request.json
    result = auth.sign_in(
        data.get('login'),
        data.get('password')
    )

    if result.get('success'):
        # Store user info in session
        session['user_id'] = result['user_id']
        session['username'] = result['username']
        return jsonify({'success': True, 'message': 'Login successful'})

    return jsonify(result)


@app.route('/api/logout', methods=['POST'])
def logout():
    """
    Destroy user session (log out).

    Response:
        {'success': True}

    Side Effects:
        - Clears all session data
        - Invalidates session cookie
    """
    session.clear()
    return jsonify({'success': True})


# ============================================================================
# USER ACCOUNT MANAGEMENT API
# ============================================================================

@app.route('/api/user_info', methods=['GET'])
@login_required
def user_info():
    """
    Get current user's account information.

    Response:
        Success: {
            'success': True,
            'user': {
                'id': 123,
                'username': 'john',
                'email': 'john@example.com',
                'is_verified': 1
            }
        }
        Failure: {'success': False, 'message': 'User not found'}
    """
    user = auth.get_user_info(session['user_id'])
    if user:
        return jsonify({'success': True, 'user': clean_data(dict(user))})
    return jsonify({'success': False, 'message': 'User not found'})


@app.route('/api/change_password', methods=['POST'])
@login_required
def change_password():
    """
    Change user's password.

    Request Body:
        {
            "old_password": "OldPass123",
            "new_password": "NewPass456"
        }

    Response:
        Success: {'success': True, 'message': 'Password updated successfully'}
        Failure: {'success': False, 'message': 'Error message'}
    """
    data = request.json
    result = auth.change_password(
        session['user_id'],
        data.get('old_password'),
        data.get('new_password')
    )
    return jsonify(result)


@app.route('/api/request_email_change', methods=['POST'])
@login_required
def request_email_change():
    """
    Request email change (sends OTP to new email).

    Request Body:
        {
            "new_email": "newemail@example.com"
        }

    Response:
        Success: {'success': True, 'message': 'Code sent to newemail@example.com'}
        Failure: {'success': False, 'message': 'Error message'}

    Side Effects:
        - Generates OTP
        - Sends verification email to NEW address
    """
    data = request.json
    result = auth.request_email_change(
        session['user_id'],
        data.get('new_email')
    )
    return jsonify(result)


@app.route('/api/verify_email_change', methods=['POST'])
@login_required
def verify_email_change():
    """
    Complete email change with OTP verification.

    Request Body:
        {
            "new_email": "newemail@example.com",
            "otp": "1234"
        }

    Response:
        Success: {'success': True, 'message': 'Email successfully updated!'}
        Failure: {'success': False, 'message': 'Invalid or expired code'}

    Side Effects:
        - Updates email in database
        - Marks OTP as used
    """
    data = request.json
    result = auth.verify_email_change(
        session['user_id'],
        data.get('new_email'),
        data.get('otp')
    )
    return jsonify(result)


# ============================================================================
# WATCHLIST API ENDPOINTS
# ============================================================================

@app.route('/api/watchlist', methods=['GET'])
@login_required
def get_watchlist():
    """
    Get user's watchlist with real-time prices.

    Response:
        {
            'success': True,
            'data': [
                {
                    'ticker': 'AAPL',
                    'added_at': '2024-02-19',
                    'current_price': '178.50',
                    'change_percent': 2.34,
                    'sparkline_data': [175.2, 176.3, ...]
                },
                ...
            ]
        }
    """
    wm = WatchlistManager(db, session['user_id'])
    return jsonify(clean_data({
        'success': True,
        'data': wm.get_watchlist_with_prices()
    }))


@app.route('/api/watchlist/add', methods=['POST'])
@login_required
def add_to_watchlist():
    """
    Add stock to watchlist.

    Request Body:
        {
            "ticker": "AAPL",
            "date": "2024-02-19"  // optional
        }

    Response:
        Success: {'success': True, 'message': 'AAPL added to watchlist'}
        Duplicate: {'success': False, 'message': 'AAPL is already in your watchlist'}
    """
    data = request.json
    wm = WatchlistManager(db, session['user_id'])
    result = wm.add_stock(
        data.get('ticker'),
        0.0,  # buy_price (not used currently)
        data.get('date')
    )
    return jsonify(result)


@app.route('/api/watchlist/remove', methods=['POST'])
@login_required
def remove_from_watchlist():
    """
    Remove stock from watchlist.

    Request Body:
        {
            "ticker": "AAPL"
        }

    Response:
        Success: {'success': True, 'message': 'AAPL removed.'}
        Not found: {'success': False, 'message': 'Stock not found in watchlist.'}
    """
    data = request.json
    wm = WatchlistManager(db, session['user_id'])
    result = wm.remove_stock(data.get('ticker'))
    return jsonify(result)


# ============================================================================
# STOCK DATA API ENDPOINTS
# ============================================================================

@app.route('/api/search_data', methods=['GET'])
@login_required
def get_search_data():
    """
    Get comprehensive stock data (fundamentals + historical prices).

    Query Parameters:
        - ticker: Stock ticker symbol (required)
        - period: Historical data period (default: 'max')
            Options: '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'max'

    Response:
        {
            'success': True,
            'ticker': 'AAPL',
            'fundamentals': {
                'Industry': 'Technology',
                'Sector': 'Consumer Electronics',
                'Description': '...',
                'Profit Margins': 0.25,
                ... (30+ metrics)
            },
            'chart_data': [
                {
                    'x': 1708355400000,  // Unix timestamp (ms)
                    'y': [175.0, 179.0, 174.5, 178.5],  // [O, H, L, C]
                    'v': 52000000  // Volume
                },
                ...
            ]
        }

    Data Sources:
        - Fundamentals: Custom Yahoo scraper (v10 endpoint)
        - Historical: yfinance library
    """
    ticker = request.args.get('ticker')
    period = request.args.get('period', 'max')

    if not ticker:
        return jsonify({'success': False, 'message': 'Ticker required'})

    try:
        # 1. FETCH FUNDAMENTALS
        scraper_data = {}
        try:
            data = analyzer.scraper.scrape(
                ticker,
                v7=True,  # Current quote
                v10=True,  # Fundamentals
                v10_full_access=False,
                use_proxy=False
            )
            if data:
                scraper_data = data
        except Exception as e:
            print(f"⚠️ Scraper warning for {ticker}: {e}")

        # 2. FETCH HISTORICAL DATA (Chart)
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period, interval="1d")

        chart_data = []
        if hasattr(hist, 'empty') and not hist.empty:
            hist = hist.reset_index()

            # Convert to candlestick format
            for _, row in hist.iterrows():
                if pd.isna(row['Close']):
                    continue

                # Unix timestamp in milliseconds
                ts = int(row['Date'].timestamp() * 1000)

                chart_data.append({
                    'x': ts,
                    'y': [
                        row['Open'],
                        row['High'],
                        row['Low'],
                        row['Close']
                    ],
                    'v': row['Volume']
                })

        # Check if we got any data
        if not scraper_data and not chart_data:
            return jsonify({
                'success': False,
                'message': 'No data found for this ticker'
            })

        # Package response
        response_data = {
            'success': True,
            'ticker': ticker.upper(),
            'fundamentals': scraper_data,
            'chart_data': chart_data
        }

        return jsonify(clean_data(response_data))

    except Exception as e:
        print(f"CRITICAL API ERROR: {e}")
        return jsonify({
            'success': False,
            'message': f"Server Error: {str(e)}"
        })


@app.route('/api/predict', methods=['GET'])
@login_required
def predict():
    """
    Get AI-powered volatility prediction for a stock.

    Query Parameters:
        - ticker: Stock ticker symbol (required)

    Response:
        {
            'ticker': 'AAPL',
            'direction': 'UP' or 'DOWN',
            'probability': 0.657,
            'confidence': 0.657,
            'regime': 'volatile',
            'accuracy': 73.5,
            'atr': 2.45,
            'current_price': 178.50
        }

    Process:
        1. Fetch ecosystem data (peers, partners) from Gemini
        2. Download 5 years of price data
        3. Calculate technical indicators + context features
        4. Train LSTM model
        5. Predict tomorrow's volatility

    Prediction Interpretation:
        - direction: "UP" (low volatility) or "DOWN" (high volatility)
        - confidence: How certain the model is (0-1)
        - probability: Raw LSTM output (0-1)
    """
    ticker = request.args.get('ticker')
    if not ticker:
        return jsonify({'success': False, 'message': 'Ticker required'})

    result = analyzer.analyze_for_api(ticker)
    return jsonify(result)


# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    """
    Run the Flask development server.

    Configuration:
        - Debug mode: Enabled (auto-reload on code changes)
        - Host: 127.0.0.1 (localhost only)
        - Port: 5000 (default Flask port)

    Note:
        For production deployment, use a production WSGI server like:
        - Gunicorn: gunicorn -w 4 app:app
        - uWSGI: uwsgi --http :5000 --wsgi-file app.py --callable app
    """
    app.run(debug=True)