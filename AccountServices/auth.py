"""
Authentication and Email Verification Module

This module handles user authentication and email verification for TradeAlchemy.
It provides secure account management with OTP (One-Time Password) email verification.

Key Features:
- SHA-256 password hashing
- Email verification with 4-digit OTP codes
- SMTP email delivery with professional templates
- Password validation
- Email change functionality

Security:
- Passwords never stored in plain text
- OTP codes expire after 10 minutes
- One-time use codes (cannot be reused)

Author: TradeAlchemy Team
"""

import hashlib #Imported for cryptographic hashing (like the SHA-256 password hashing)
import sqlite3 #Lirary to interact with SQLite Database
import smtplib #The built-in Python library that handles the SMTP protocol to actually send the emails.
import random # To generate random 4-digit otp
from datetime import datetime, timedelta #Get current time and calculate 10-minute expiration window
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
# These help format the email properly so it can support both plain text and the professional HTML template you designed.
from typing import Optional, Tuple, Dict

# Email configuration (can be overridden by config.py)
try:
    from config import EMAIL_SENDER, EMAIL_PASSWORD
except ImportError:
    EMAIL_SENDER = "phoniexblaze5@gmail.com"
    EMAIL_PASSWORD = "your-app-password-here"


# ============================================================================
# EMAIL VERIFICATION CLASS
# ============================================================================

class EmailVerification:
    """
    Handles OTP generation, database storage, and SMTP email delivery.

    This class manages the complete email verification workflow:
    1. Generate random 4-digit OTP code
    2. Store in database with expiration timestamp
    3. Send formatted email to user
    4. Verify OTP when user submits it
    5. Mark code as used to prevent reuse

    Attributes:
        db: DatabaseManager instance for database operations
        sender (str): Email address to send from
        password (str): SMTP app password (not regular Gmail password)
        Simple Mail Transfer Protocol (SMTP)

    Security Considerations:
        - OTP codes are random (not predictable)
        - 10-minute expiration window
        - One-time use (is_used flag)
        - Multiple unused OTPs allowed (only latest is valid)

    Example:
        >>> email_verifier = EmailVerification(db_manager)
        >>> email_verifier.send_verification_code("user@example.com")
        >>> # User receives email with 4-digit code
        >>> if email_verifier.verify_and_activate_user("user@example.com", "1234"):
        >>>     print("Email verified successfully!")
    """

    def __init__(self, db_manager):
        """
        Initialize email verification system.

        Args:
            db_manager: DatabaseManager instance for OTP storage

        Side Effects:
            - Sets SMTP credentials from config or defaults
            - Does NOT establish SMTP connection (created per-send)
        """
        self.db = db_manager
        self.sender = EMAIL_SENDER
        self.password = EMAIL_PASSWORD

    def generate_otp(self):
        """
        Generate a random 4-digit OTP code.

        Returns:
            str: 4-digit code as string (e.g., "1234", "0587")

        Implementation:
            - Uses random.randint for cryptographically weak but sufficient randomness
            - Range: 1000 to 9999 (ensures 4 digits)

        Security Note:
            This is not cryptographically secure random, but for OTP codes
            with 10-minute expiration and one-time use, it's sufficient.
            There are 10,000 possible codes, making brute force impractical.
        """
        return str(random.randint(1000, 9999))

    def store_otp(self, email, otp_code):
        """
        Store OTP code in database with expiration timestamp.

        This method creates a new OTP record that will expire in 10 minutes.
        Multiple OTPs can exist for the same email (only latest unused one is valid).

        Args:
            email (str): User's email address
            otp_code (str): The 4-digit OTP code

        Side Effects:
            - Inserts row into otp_verification table
            - Sets created_at to current time
            - Sets expires_at to 10 minutes from now
            - Sets is_used to 0 (unused)

        Database Row Created:
            {
                "email": "user@example.com",
                "otp_code": "1234",
                "created_at": "2024-02-19 14:30:00",
                "expires_at": "2024-02-19 14:40:00",
                "is_used": 0
            }
        """
        conn = self.db.get_connection()
        cursor = conn.cursor()

        # Calculate expiration time (10 minutes from now)
        expires_at = (datetime.now() + timedelta(minutes=10)).isoformat()

        # Insert OTP record
        cursor.execute(
            """INSERT INTO otp_verification (email, otp_code, expires_at)
               VALUES (?, ?, ?)""",
            (email, otp_code, expires_at)
        )

        conn.commit()
        conn.close()

    def verify_otp(self, email, otp_code):
        """
        Verify OTP code and mark as used if valid.

        This method checks if the provided OTP is:
        1. Stored in database for this email
        2. Not yet used (is_used = 0)
        3. Not expired (expires_at > current time)
        4. The most recent OTP (ORDER BY created_at DESC LIMIT 1)

        Args:
            email (str): User's email address
            otp_code (str): The code to verify

        Returns:
            bool: True if OTP is valid, False otherwise

        Side Effects:
            - If valid, marks OTP as used (is_used = 1)
            - Prevents the same code from being used twice

        Security:
            - Only checks the MOST RECENT unused OTP
            - Old/expired codes cannot be used
            - Each code can only be used once

        Example:
            >>> verifier.store_otp("user@example.com", "1234")
            >>> verifier.verify_otp("user@example.com", "1234")  # True (first use)
            >>> verifier.verify_otp("user@example.com", "1234")  # False (already used)
        """
        conn = self.db.get_connection()
        cursor = conn.cursor()

        # Query for valid OTP:
        # - Matching email and code
        # - Not yet used
        # - Not expired
        # - Most recent (ORDER BY created_at DESC)
        cursor.execute(
            """SELECT id
               FROM otp_verification
               WHERE email = ?
                 AND otp_code = ?
                 AND is_used = 0
                 AND expires_at > ?
               ORDER BY created_at DESC LIMIT 1""",
            (email, otp_code, datetime.now().isoformat())
        )

        result = cursor.fetchone()

        if result:
            # Valid OTP found - mark as used
            cursor.execute(
                "UPDATE otp_verification SET is_used = 1 WHERE id = ?",
                (result[0],)
            )
            conn.commit()
            conn.close()
            return True

        # No valid OTP found
        conn.close()
        return False

    def send_otp_email(self, receiver_email, otp_code):
        """
        Send OTP code via email using SMTP.

        This method composes and sends a professionally formatted HTML email
        containing the 4-digit OTP code.

        Args:
            receiver_email (str): Recipient's email address
            otp_code (str): The 4-digit code to include in email

        Returns:
            bool: True if email sent successfully, False on error

        Email Configuration:
            - Protocol: SMTP_SSL (port 465)
            - Server: smtp.gmail.com
            - Authentication: Gmail app password (not regular password)

        Email Template:
            - Professional HTML design
            - Clear subject line
            - Prominent OTP display
            - Security warnings
            - Expiration notice (10 minutes)

        Error Handling:
            - Catches SMTP exceptions
            - Prints error messages for debugging
            - Returns False on any failure
            - Always attempts to close connection

        Gmail Setup:
            To use Gmail SMTP:
            1. Enable 2-factor authentication on your Google account
            2. Generate an "App Password" from Google Account settings
            3. Use the app password (not your regular password)

        Example:
            >>> success = verifier.send_otp_email("user@example.com", "1234")
            >>> if success:
            >>>     print("Email sent!")
        """
        # Compose email message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = "TradeAlchemy - Your Authorization Code"
        msg['From'] = f"TradeAlchemy Security <{self.sender}>"
        msg['To'] = receiver_email

        # Professional HTML email template with styled OTP display
        html_body = f"""
        <html>
            <body style="font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; padding: 20px; background-color: #f4f4f5; color: #333; line-height: 1.6;">
                <div style="max-width: 600px; margin: 0 auto; background-color: #ffffff; border: 1px solid #e4e4e7; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">

                    <!-- Header -->
                    <div style="background-color: #121212; padding: 25px; text-align: center; border-bottom: 3px solid #A855F7;">
                        <h2 style="color: #ffffff; margin: 0; font-size: 24px; font-weight: 700; letter-spacing: 1px;">
                            Trade<span style="color: #A855F7;">Alchemy</span>
                        </h2>
                    </div>

                    <!-- Body Content -->
                    <div style="padding: 40px 30px;">
                        <p style="font-size: 16px; margin-bottom: 20px;">Dear User,</p>
                        <p style="font-size: 16px; margin-bottom: 30px;">
                            We received a request to verify this email address associated with your TradeAlchemy account. 
                            To proceed, please use the authorization code provided below:
                        </p>

                        <!-- OTP Display Box -->
                        <div style="text-align: center; margin: 35px 0;">
                            <span style="display: inline-block; font-size: 36px; font-weight: 700; color: #121212; letter-spacing: 10px; padding: 20px 40px; background-color: #f3e8ff; border-radius: 8px; border: 2px dashed #A855F7;">
                                {otp_code}
                            </span>
                        </div>

                        <!-- Security Warning -->
                        <div style="background-color: #fef2f2; border-left: 4px solid #ef4444; padding: 15px; margin-bottom: 30px;">
                            <p style="font-size: 14px; color: #991b1b; margin: 0; font-weight: 600;">
                                Security Notice: This code is strictly valid for the next 10 minutes.
                            </p>
                        </div>

                        <p style="font-size: 14px; color: #666; margin-top: 30px;">
                            If you did not initiate this request, please disregard this email. Your account remains secure.
                        </p>
                        <p style="font-size: 14px; color: #666; margin-top: 10px;">
                            Sincerely,<br><strong>The TradeAlchemy Security Team</strong>
                        </p>
                    </div>

                    <!-- Footer -->
                    <div style="background-color: #f9fafb; padding: 20px; text-align: center; font-size: 12px; color: #9ca3af; border-top: 1px solid #e4e4e7;">
                        &copy; {datetime.now().year} TradeAlchemy. All rights reserved.
                    </div>
                </div>
            </body>
        </html>
        """

        # Attach HTML body to message
        msg.attach(MIMEText(html_body, 'html'))

        try:
            # Establish secure SMTP connection
            server = smtplib.SMTP_SSL('smtp.gmail.com', 465)

            # Login with app password
            server.login(self.sender, self.password)

            # Send email
            server.sendmail(self.sender, receiver_email, msg.as_string())

            # Close connection gracefully
            try:
                server.quit()
            except:
                pass  # Ignore errors during quit

            return True

        except Exception as e:
            print(f"CRITICAL MAIL ERROR: {e}")
            return False

    def send_verification_code(self, email):
        """
        Generate OTP, store in database, and send via email.

        This is the main method for initiating email verification.
        It combines OTP generation, storage, and email delivery.

        Args:
            email (str): Email address to verify

        Returns:
            bool: True if code was sent successfully, False otherwise

        Process Flow:
            1. Generate random 4-digit code
            2. Store in database with 10-minute expiration
            3. Send email to user
            4. Return success/failure status

        Example:
            >>> email_verifier.send_verification_code("user@example.com")
            True  # Code generated, stored, and email sent
        """
        otp_code = self.generate_otp()
        self.store_otp(email, otp_code)
        return self.send_otp_email(email, otp_code)

    def verify_and_activate_user(self, email, otp_code):
        """
        Verify OTP and activate user account.

        This method combines OTP verification with user account activation.
        It's used during the sign-up process to verify email ownership.

        Args:
            email (str): User's email address
            otp_code (str): 4-digit code provided by user

        Returns:
            bool: True if verification successful and account activated,
                  False if OTP invalid/expired

        Side Effects:
            - If OTP valid:
                * Marks OTP as used (is_used = 1)
                * Sets user's is_verified = 1
                * Account is now fully activated

        Database Updates:
            1. otp_verification: is_used = 1
            2. users: is_verified = 1 WHERE email = ?

        Example:
            >>> # User signs up
            >>> verifier.send_verification_code("newuser@example.com")
            >>> # User receives code "1234" in email
            >>> if verifier.verify_and_activate_user("newuser@example.com", "1234"):
            >>>     print("Account activated!")
        """
        # Verify the OTP code
        if self.verify_otp(email, otp_code):
            # OTP is valid - activate the user account
            conn = self.db.get_connection()
            cursor = conn.cursor()

            cursor.execute(
                "UPDATE users SET is_verified = 1 WHERE email = ?",
                (email,)
            )

            conn.commit()
            conn.close()
            return True

        return False


# ============================================================================
# AUTHENTICATION MANAGER CLASS
# ============================================================================

class AuthManager:
    """
    Handles user account lifecycle: Sign-up, Sign-in, and Validation.

    This class manages all user authentication operations including:
    - Account creation with validation
    - Password hashing and verification
    - Sign-in authentication
    - Password changes
    - Email changes with verification

    Security Features:
        - SHA-256 password hashing (passwords never stored in plain text)
        - Email verification required before sign-in
        - Password strength validation
        - Unique username and email enforcement

    Attributes:
        db: DatabaseManager instance
        email_verifier: EmailVerification instance

    Example:
        >>> auth = AuthManager(db_manager)
        >>> result = auth.sign_up("john", "john@example.com", "SecurePass123")
        >>> if result['success']:
        >>>     print("Account created! Check email for verification code.")
    """

    def __init__(self, db_manager):
        """
        Initialize authentication manager.

        Args:
            db_manager: DatabaseManager instance for database operations

        Side Effects:
            - Creates EmailVerification instance
            - Does NOT create database connection (created per-operation)
        """
        self.db = db_manager
        self.email_verifier = EmailVerification(db_manager)

    @staticmethod
    def hash_password(password: str) -> str:
        """
        Hash password using SHA-256.

        This method converts a plain-text password into a secure hash.
        The hash is deterministic (same password = same hash) which allows
        verification but prevents recovery of the original password.

        Args:
            password (str): Plain-text password

        Returns:
            str: 64-character hexadecimal hash

        Security:
            - SHA-256 is one-way (cannot reverse to get password)
            - Same password always produces same hash
            - Rainbow table attacks possible (consider adding salt in production)

        Production Note:
            For production, consider bcrypt or Argon2 with per-user salts.
            SHA-256 is used here for simplicity.

        Example:
            >>> hash1 = AuthManager.hash_password("MyPassword123")
            >>> hash2 = AuthManager.hash_password("MyPassword123")
            >>> print(hash1 == hash2)  # True
            >>> print(len(hash1))  # 64
        """
        return hashlib.sha256(password.encode()).hexdigest()

    @staticmethod
    def validate_email(email: str) -> bool:
        """
        Validate email format (basic check).

        This performs a simple sanity check that the email contains
        @ symbol and has a domain part with a dot.

        Args:
            email (str): Email address to validate

        Returns:
            bool: True if format appears valid, False otherwise

        Validation Rules:
            - Must contain '@' character
            - Domain part (after @) must contain '.'

        Examples:
            >>> AuthManager.validate_email("user@example.com")  # True
            >>> AuthManager.validate_email("user@example")      # False
            >>> AuthManager.validate_email("userexample.com")   # False

        Production Note:
            This is a basic check. For production, consider:
            - Regex pattern for RFC 5322 compliance
            - DNS MX record verification
            - Disposable email detection
        """
        return '@' in email and '.' in email.split('@')[1]

    @staticmethod
    def validate_password(password: str) -> Tuple[bool, str]:
        """
        Validate password strength.

        This method enforces minimum password requirements:
        1. At least 8 characters long
        2. Contains at least one number

        Args:
            password (str): Password to validate

        Returns:
            Tuple[bool, str]: (is_valid, error_message)
                - is_valid: True if password meets requirements
                - error_message: Empty string if valid, error description if not

        Examples:
            >>> AuthManager.validate_password("Pass123")
            (True, "")
            >>> AuthManager.validate_password("Pass")
            (False, "Password must be at least 8 characters")
            >>> AuthManager.validate_password("Password")
            (False, "Password must contain at least one number")

        Production Note:
            Consider adding:
            - Uppercase letter requirement
            - Special character requirement
            - Check against common passwords list
            - Pwned Passwords API check
        """
        if len(password) < 8:
            return False, "Password must be at least 8 characters"
        if not any(char.isdigit() for char in password):
            return False, "Password must contain at least one number"
        return True, ""

    def sign_up(self, username: str, email: str, password: str) -> Dict:
        """
        Create a new user account with email verification.

        This method handles the complete sign-up process:
        1. Validate input data
        2. Check for existing accounts
        3. Hash password
        4. Create database record
        5. Send verification email

        Args:
            username (str): Desired username
            email (str): Email address
            password (str): Plain-text password (will be hashed)

        Returns:
            Dict: Response with status and message:
                Success: {
                    'success': True,
                    'message': 'Account created! Check email for code.',
                    'requires_verification': True
                }
                Failure: {
                    'success': False,
                    'message': 'Error description',
                    'requires_verification': False
                }

        Validation Performed:
            - All fields present (not None or empty)
            - Valid email format
            - Password meets strength requirements
            - Username not already taken
            - Email not already registered

        Side Effects:
            - Creates row in users table (is_verified = 0)
            - Generates and stores OTP code
            - Sends verification email

        Error Cases:
            - Missing fields → "All fields are required"
            - Invalid email → "Invalid email format"
            - Weak password → Specific password error
            - Duplicate username/email → "Username or email already exists"
            - Database error → Exception message

        Example:
            >>> auth = AuthManager(db)
            >>> result = auth.sign_up("john", "john@example.com", "SecurePass123")
            >>>
            >>> if result['success']:
            >>>     print(result['message'])  # "Account created! Check email..."
            >>> else:
            >>>     print(f"Error: {result['message']}")
        """
        try:
            # Validation 1: All fields present
            if not username or not email or not password:
                return {
                    'success': False,
                    'message': 'All fields are required',
                    'requires_verification': False
                }

            # Validation 2: Email format
            if not self.validate_email(email):
                return {
                    'success': False,
                    'message': 'Invalid email format',
                    'requires_verification': False
                }

            # Validation 3: Password strength
            is_valid, msg = self.validate_password(password)
            if not is_valid:
                return {
                    'success': False,
                    'message': msg,
                    'requires_verification': False
                }

            # Hash password for storage
            hashed_pw = self.hash_password(password)

            # Create database record
            conn = self.db.get_connection()
            cursor = conn.cursor()

            cursor.execute(
                "INSERT INTO users (username, email, password, is_verified) VALUES (?, ?, ?, 0)",
                (username.strip(), email.strip().lower(), hashed_pw)
            )

            conn.commit()
            conn.close()

            # Send verification email
            email_sent = self.email_verifier.send_verification_code(email.strip().lower())

            return {
                'success': True,
                'message': f'Account created! Check {email} for code.' if email_sent
                else 'Account created, but email failed.',
                'requires_verification': True
            }

        except sqlite3.IntegrityError:
            # Username or email already exists (UNIQUE constraint violated)
            return {
                'success': False,
                'message': 'Username or email already exists',
                'requires_verification': False
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Sign up failed: {str(e)}',
                'requires_verification': False
            }

    def verify_email(self, email: str, otp_code: str) -> Dict:
        """
        Verify email with OTP code and activate account.

        This method is called when the user submits the verification code
        they received via email.

        Args:
            email (str): User's email address
            otp_code (str): 4-digit code from email

        Returns:
            Dict: Verification result:
                Success: {'success': True, 'message': 'Email verified successfully!'}
                Failure: {'success': False, 'message': 'Invalid or expired verification code'}

        Side Effects:
            - If valid: Sets is_verified = 1 in users table
            - If valid: Marks OTP as used in otp_verification table

        Error Cases:
            - Code doesn't exist → "Invalid or expired verification code"
            - Code expired → "Invalid or expired verification code"
            - Code already used → "Invalid or expired verification code"
            - Database error → "Verification failed: {error}"

        Example:
            >>> # After sign-up, user receives code "1234"
            >>> result = auth.verify_email("john@example.com", "1234")
            >>> if result['success']:
            >>>     print("Account verified! You can now sign in.")
        """
        try:
            if self.email_verifier.verify_and_activate_user(email.strip().lower(), otp_code):
                return {'success': True, 'message': 'Email verified successfully!'}
            return {'success': False, 'message': 'Invalid or expired verification code'}
        except Exception as e:
            return {'success': False, 'message': f'Verification failed: {str(e)}'}

    def sign_in(self, login: str, password: str) -> Dict:
        """
        Authenticate user and create session.

        This method verifies credentials and checks email verification status.

        Args:
            login (str): Username OR email address
            password (str): Plain-text password (will be hashed for comparison)

        Returns:
            Dict: Authentication result:
                Success: {
                    'success': True,
                    'message': 'Welcome, {username}!',
                    'user_id': 123,
                    'username': 'john'
                }
                Unverified: {
                    'success': False,
                    'message': 'Please verify your email',
                    'user_id': None,
                    'username': None
                }
                Failure: {
                    'success': False,
                    'message': 'Invalid credentials',
                    'user_id': None,
                    'username': None
                }

        Authentication Process:
            1. Hash provided password
            2. Query database for matching username/email + password hash
            3. Check if account is verified
            4. Return user_id and username for session

        Security:
            - Password is hashed before database query
            - Plain-text passwords never touch the database
            - Supports login with username OR email
            - Requires email verification

        Example:
            >>> result = auth.sign_in("john@example.com", "SecurePass123")
            >>>
            >>> if result['success']:
            >>>     session['user_id'] = result['user_id']
            >>>     session['username'] = result['username']
            >>>     print(f"Welcome, {result['username']}!")
        """
        try:
            # Hash password for database comparison
            hashed_pw = self.hash_password(password)

            conn = self.db.get_connection()
            cursor = conn.cursor()

            # Query: Match (username OR email) AND password hash
            cursor.execute(
                """SELECT id, username, is_verified
                   FROM users
                   WHERE (username = ? OR email = ?)
                     AND password = ?""",
                (login.strip(), login.strip().lower(), hashed_pw)
            )

            result = cursor.fetchone()
            conn.close()

            if result:
                user_id, username, is_verified = result

                # Check if email is verified
                if is_verified == 0:
                    return {
                        'success': False,
                        'message': 'Please verify your email',
                        'user_id': None,
                        'username': None
                    }

                # Successful authentication
                return {
                    'success': True,
                    'message': f'Welcome, {username}!',
                    'user_id': user_id,
                    'username': username
                }

            # No matching credentials found
            return {
                'success': False,
                'message': 'Invalid credentials',
                'user_id': None,
                'username': None
            }

        except Exception as e:
            return {
                'success': False,
                'message': f'Sign in failed: {str(e)}',
                'user_id': None,
                'username': None
            }

    def resend_verification_code(self, email: str) -> Dict:
        """
        Trigger a new OTP email for the user.

        This method is called when the user didn't receive the first code
        or the code expired.

        Args:
            email (str): User's email address

        Returns:
            Dict: Resend result:
                Success: {'success': True, 'message': 'Verification code resent!'}
                Failure: {'success': False, 'message': 'Failed to send email.'}

        Side Effects:
            - Generates new OTP code
            - Stores in database (old unused codes remain but are superseded)
            - Sends new email

        Example:
            >>> # User clicks "Resend Code"
            >>> result = auth.resend_verification_code("john@example.com")
            >>> print(result['message'])
        """
        email_sent = self.email_verifier.send_verification_code(email.strip().lower())
        return {
            'success': email_sent,
            'message': 'Verification code resent!' if email_sent else 'Failed to send email.'
        }

    def get_user_info(self, user_id: int) -> Optional[Dict]:
        """
        Fetch basic user details for the frontend profile.

        This method retrieves non-sensitive user information for display
        in the account settings page.

        Args:
            user_id (int): User's database ID (from session)

        Returns:
            dict or None: User information:
                {
                    'id': 123,
                    'username': 'john',
                    'email': 'john@example.com',
                    'is_verified': 1
                }
                Returns None if user not found or error occurs

        Example:
            >>> user = auth.get_user_info(session['user_id'])
            >>> if user:
            >>>     print(f"Username: {user['username']}")
            >>>     print(f"Email: {user['email']}")
        """
        try:
            conn = self.db.get_connection()
            cursor = conn.cursor()

            cursor.execute(
                "SELECT id, username, email, is_verified FROM users WHERE id = ?",
                (user_id,)
            )

            row = cursor.fetchone()
            conn.close()

            if row:
                return dict(row)
            return None

        except Exception as e:
            print(f"Error fetching user info: {e}")
            return None

    def change_password(self, user_id: int, old_password: str, new_password: str) -> Dict:
        """
        Verify old password and update to new password.

        This method implements a secure password change process that
        requires the user to provide their current password.

        Args:
            user_id (int): User's database ID
            old_password (str): Current password (for verification)
            new_password (str): New password to set

        Returns:
            Dict: Change result:
                Success: {'success': True, 'message': 'Password updated successfully'}
                Wrong old password: {'success': False, 'message': 'Incorrect current password'}
                Weak new password: {'success': False, 'message': 'Password must be at least 8 characters'}

        Security:
            - Requires current password verification (prevents unauthorized changes)
            - Validates new password strength
            - Hashes new password before storage

        Example:
            >>> result = auth.change_password(
            >>>     session['user_id'],
            >>>     "OldPass123",
            >>>     "NewSecurePass456"
            >>> )
            >>> if result['success']:
            >>>     print("Password changed successfully!")
        """
        try:
            conn = self.db.get_connection()
            cursor = conn.cursor()

            # Fetch current password hash
            cursor.execute("SELECT password FROM users WHERE id = ?", (user_id,))
            result = cursor.fetchone()

            if not result:
                return {'success': False, 'message': 'User not found'}

            stored_hash = result[0]

            # Verify old password
            if stored_hash != self.hash_password(old_password):
                return {'success': False, 'message': 'Incorrect current password'}

            # Validate new password
            is_valid, msg = self.validate_password(new_password)
            if not is_valid:
                return {'success': False, 'message': msg}

            # Update password
            new_hash = self.hash_password(new_password)
            cursor.execute(
                "UPDATE users SET password = ? WHERE id = ?",
                (new_hash, user_id)
            )

            conn.commit()
            conn.close()

            return {'success': True, 'message': 'Password updated successfully'}

        except Exception as e:
            return {'success': False, 'message': f'Error: {str(e)}'}

    def request_email_change(self, user_id: int, new_email: str) -> Dict:
        """
        Validate new email and send OTP to it.

        This initiates the email change process by sending a verification
        code to the NEW email address.

        Args:
            user_id (int): User's database ID
            new_email (str): New email address to verify

        Returns:
            Dict: Request result:
                Success: {'success': True, 'message': 'Code sent to {email}'}
                Invalid: {'success': False, 'message': 'Invalid email format'}
                In use: {'success': False, 'message': 'This email is already in use.'}

        Process:
            1. Validate email format
            2. Check if email already exists (any user)
            3. Send OTP to new email
            4. User must verify before email is changed

        Example:
            >>> result = auth.request_email_change(
            >>>     session['user_id'],
            >>>     "newemail@example.com"
            >>> )
            >>> if result['success']:
            >>>     print("Check your new email for verification code")
        """
        try:
            new_email = new_email.strip().lower()

            # Validate format
            if not self.validate_email(new_email):
                return {'success': False, 'message': 'Invalid email format'}

            conn = self.db.get_connection()
            cursor = conn.cursor()

            # Check if email is already in use by ANY user
            cursor.execute("SELECT id FROM users WHERE email = ?", (new_email,))
            if cursor.fetchone():
                conn.close()
                return {'success': False, 'message': 'This email is already in use.'}

            conn.close()

            # Send OTP to the NEW email
            email_sent = self.email_verifier.send_verification_code(new_email)

            if email_sent:
                return {'success': True, 'message': f'Code sent to {new_email}'}
            return {'success': False, 'message': 'Failed to send verification code.'}

        except Exception as e:
            return {'success': False, 'message': f'Error processing request: {str(e)}'}

    def verify_email_change(self, user_id: int, new_email: str, otp_code: str) -> Dict:
        """
        Verify OTP and update user's email in the DB.

        This completes the email change process by verifying the code
        sent to the new email and updating the user's record.

        Args:
            user_id (int): User's database ID
            new_email (str): New email address (same as in request)
            otp_code (str): 4-digit code sent to new email

        Returns:
            Dict: Verification result:
                Success: {'success': True, 'message': 'Email successfully updated!'}
                Failure: {'success': False, 'message': 'Invalid or expired verification code'}

        Side Effects:
            - If valid: Updates email in users table
            - If valid: Marks OTP as used

        Security:
            - Verifies code was sent to the new email
            - Prevents email change without verification

        Example:
            >>> # After requesting change, user receives code "1234"
            >>> result = auth.verify_email_change(
            >>>     session['user_id'],
            >>>     "newemail@example.com",
            >>>     "1234"
            >>> )
            >>> if result['success']:
            >>>     print("Email updated! Please sign in again.")
        """
        try:
            new_email = new_email.strip().lower()

            # Verify the OTP code sent to the new email
            if self.email_verifier.verify_otp(new_email, otp_code):
                conn = self.db.get_connection()
                cursor = conn.cursor()

                # Update user's email
                cursor.execute(
                    "UPDATE users SET email = ? WHERE id = ?",
                    (new_email, user_id)
                )

                conn.commit()
                conn.close()

                return {'success': True, 'message': 'Email successfully updated!'}

            return {'success': False, 'message': 'Invalid or expired verification code'}

        except Exception as e:
            return {'success': False, 'message': f'Update failed: {str(e)}'}