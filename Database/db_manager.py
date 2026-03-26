"""
Database Manager Module

This module handles all database operations for the TradeAlchemy application.
It provides a clean interface for SQLite database operations including:
- User account management
- Email verification (OTP)
- Watchlist storage

Key Features:
- Thread-safe connection handling
- Automatic table initialization
- Row factory for dict-like access
- OTP cleanup utilities

Author: TradeAlchemy Team
"""

import sqlite3
from datetime import datetime


# ============================================================================
# DATABASE MANAGER CLASS
# ============================================================================

class DatabaseManager:
    """
    Manages SQLite database operations for the application.

    This class provides a centralized interface for all database operations.
    It handles connection management, table initialization, and provides
    utility methods for data cleanup.

    Database Schema:

        users table:
            - id: Primary key (auto-increment)
            - username: Unique username
            - email: Unique email address
            - password: SHA-256 hashed password
            - is_verified: Email verification status (0 or 1)
            - created_at: Account creation timestamp

        otp_verification table:
            - id: Primary key (auto-increment)
            - email: Email address for verification
            - otp_code: 4-digit verification code
            - created_at: OTP generation timestamp
            - expires_at: OTP expiration timestamp (10 minutes from creation)
            - is_used: Whether OTP has been consumed (0 or 1)

        watchlist table:
            - id: Primary key (auto-increment)
            - user_id: Foreign key to users.id
            - ticker: Stock ticker symbol (e.g., "AAPL")
            - buy_price: Optional purchase price
            - added_at: Timestamp when stock was added
            - UNIQUE constraint: (user_id, ticker) - prevents duplicates

    Attributes:
        db_path (str): Path to SQLite database file

    Example:
        >>> db = DatabaseManager("app.db")
        >>> conn = db.get_connection()
        >>> cursor = conn.cursor()
        >>> cursor.execute("SELECT * FROM users WHERE id = ?", (1,))
        >>> user = cursor.fetchone()
        >>> print(dict(user))  # Access as dictionary
        >>> conn.close()
    """

    def __init__(self, db_path: str = "app.db"):
        """
        Initialize database connection and create tables if they don't exist.

        This constructor is called once when the Flask app starts.
        It ensures all required tables exist before any operations.

        Args:
            db_path (str): Path to SQLite database file (default: "app.db")
                The file will be created if it doesn't exist.

        Side Effects:
            - Creates database file if not exists
            - Creates all tables (users, otp_verification, watchlist)
            - Sets up foreign key constraints
            - Creates unique constraints
        """
        self.db_path = db_path
        # Create all necessary tables when the database manager is initialized
        self._initialize_tables()

    def get_connection(self):
        """
        Create a thread-safe database connection for Flask.

        This method should be called at the start of each database operation.
        Each connection is independent and can be used in different threads.

        Returns:
            sqlite3.Connection: Database connection with row_factory set
                The row_factory allows dict-like access to rows:
                row['username'] instead of row[0]

        Configuration:
            - check_same_thread=True: Thread-safe for Flask
            - row_factory=sqlite3.Row: Enables dict-like row access

        Example:
            >>> conn = db.get_connection()
            >>> cursor = conn.cursor()
            >>> cursor.execute("SELECT * FROM users WHERE id = ?", (1,))
            >>> user = cursor.fetchone()
            >>> print(user['username'])  # Dict-like access
            >>> conn.close()  # Always close when done
        """
        conn = sqlite3.connect(self.db_path, check_same_thread=True)
        conn.row_factory = sqlite3.Row  # Return rows as dict-like objects
        return conn

    def _initialize_tables(self): # private method
        """
        Create all database tables if they don't already exist.

        This method is called automatically during __init__().
        It's idempotent - safe to call multiple times (IF NOT EXISTS).

        Tables Created:
            1. users: Account information
            2. otp_verification: Email verification codes
            3. watchlist: User's tracked stocks

        Side Effects:
            - Creates tables if missing
            - Sets up foreign keys
            - Creates unique constraints
            - Sets default values

        Schema Details:

            users table:
                - Unique username and email
                - Password stored as SHA-256 hash
                - Email verification required (is_verified flag)
                - Auto-timestamp on creation

            otp_verification table:
                - Stores 4-digit verification codes
                - 10-minute expiration window
                - One-time use (is_used flag)
                - Multiple OTPs can exist for same email

            watchlist table:
                - Links users to stock tickers
                - Unique constraint prevents duplicate entries
                - Foreign key ensures data integrity
                - Optional buy_price for tracking gains/losses
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        # ====================================================================
        # USERS TABLE
        # ====================================================================
        # Stores user account information
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                is_verified INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # ====================================================================
        # OTP VERIFICATION TABLE
        # ====================================================================
        # Stores email verification codes (4-digit OTPs)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS otp_verification (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL,
                otp_code TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                expires_at TEXT NOT NULL,
                is_used INTEGER DEFAULT 0
            )
        """)

        # ====================================================================
        # WATCHLIST TABLE - CORRECTED SCHEMA
        # ====================================================================
        # Stores user's tracked stocks with optional buy price and add date
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS watchlist (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                ticker TEXT NOT NULL,
                buy_price REAL,
                added_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id),
                UNIQUE(user_id, ticker)
            )
        """)

        conn.commit()
        conn.close()

    def cleanup_expired_otps(self):
        """
        Remove expired OTP codes from the database.

        This method should be called periodically (e.g., daily cron job)
        to prevent the otp_verification table from growing indefinitely.

        Deletion Criteria:
            - OTPs where expires_at < current time

        Side Effects:
            - Deletes rows from otp_verification table
            - Commits changes to database

        Use Case:
            >>> # In a scheduled task or maintenance script:
            >>> db = DatabaseManager()
            >>> db.cleanup_expired_otps()
            >>> print("Expired OTPs cleaned up")

        Note:
            This is a maintenance operation. The verification process
            already checks expiration, so cleanup is just for database hygiene.
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        # Delete OTPs that have expired
        cursor.execute(
            "DELETE FROM otp_verification WHERE expires_at < ?",
            (datetime.now().isoformat(),)
        )

        conn.commit()
        conn.close()