"""
LSTM Confidence Model Module

This module implements a Bidirectional LSTM neural network for predicting
significant stock price movements (volatility spikes). It uses advanced
weighting techniques to handle imbalanced data and emphasize recent patterns.

Key Features:
- Bidirectional LSTM for pattern recognition in both directions
- Time-weighted training (recent data more important)
- Class-balanced sampling (handles rare volatility spikes)
- Early stopping and learning rate reduction
- Sequence-based predictions (60-day lookback window)

Author: TradeAlchemy Team
"""

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import time


# ============================================================================
# MULTI-TIMEFRAME LSTM MODEL
# ============================================================================

class MultiTimeframeLSTM:
    """
    Bidirectional LSTM model for predicting stock volatility spikes.

    This model uses a sequence of past observations (lookback window) to predict
    if the next day will have a significant price movement.

    Architecture:
        - Input: 60-day sequences of technical indicators
        - Layer 1: Bidirectional LSTM (128 units) with return sequences
        - Layer 2: Batch Normalization
        - Layer 3: Dropout (30%)
        - Layer 4: Bidirectional LSTM (64 units)
        - Layer 5: Batch Normalization
        - Layer 6: Dropout (30%)
        - Layer 7: Dense (32 units, swish activation)
        - Output: Dense (1 unit, sigmoid activation) → probability

    Key Design Decisions:

        1. Bidirectional LSTM:
           - Reads sequences forward AND backward
           - Captures patterns that depend on future context
           - Useful for financial data where patterns may be symmetric

        2. Sequence Length (60 days):
           - Approximately 3 months of trading data
           - Long enough to capture medium-term trends
           - Short enough to train efficiently

        3. Dropout (30%):
           - Prevents overfitting to training data
           - Forces model to learn robust features

        4. Batch Normalization:
           - Stabilizes training
           - Allows higher learning rates
           - Reduces internal covariate shift

        5. Sigmoid Output:
           - Outputs probability (0 to 1)
           - > 0.5: Predicts significant move (high volatility)
           - < 0.5: Predicts quiet day (low volatility)

    Attributes:
        lookback (int): Number of days to look back (default: 60)
        seed (int): Random seed for reproducibility
        scaler (RobustScaler): Feature scaler (robust to outliers)
        model (Sequential): Trained Keras model

    Example:
        >>> lstm = MultiTimeframeLSTM(lookback=60)
        >>> result = lstm.train_and_predict(df_with_features)
        >>> if result:
        >>>     prob, accuracy, _ = result
        >>>     print(f"Tomorrow's volatility probability: {prob:.1%}")
        >>>     print(f"Model accuracy: {accuracy:.1%}")
    """

    def __init__(self, lookback=60, seed=42):
        """
        Initialize the LSTM model with configuration.

        Args:
            lookback (int): Number of historical days to use as input (default: 60)
                - Too small: May miss longer-term patterns
                - Too large: Harder to train, more data required
            seed (int): Random seed for reproducibility (default: 42)
                - Ensures consistent results across runs

        Side Effects:
            - Sets NumPy and TensorFlow random seeds
            - Ensures deterministic behavior (same results each time)
        """
        self.lookback = lookback
        self.seed = seed
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        self.model = None

        # Set seeds for reproducibility
        np.random.seed(seed)
        tf.random.set_seed(seed)

    def create_sequences(self, X, y):
        """
        Convert flat time series into sequences for LSTM input.

        LSTM models require 3D input: (samples, timesteps, features)
        This method creates overlapping sequences from the time series.

        Args:
            X (numpy.ndarray): Feature matrix, shape (n_days, n_features)
            y (numpy.ndarray): Target vector, shape (n_days,)

        Returns:
            tuple: (X_sequences, y_sequences)
                - X_sequences: shape (n_sequences, lookback, n_features)
                - y_sequences: shape (n_sequences,)

        Example:
            If we have 100 days of data and lookback=60:
            - Day 0-59: First sequence → predicts Day 60
            - Day 1-60: Second sequence → predicts Day 61
            - ...
            - Day 39-98: Last sequence → predicts Day 99

            Result: 40 sequences of length 60

        Visualization:
            Input X: [Day0, Day1, Day2, ..., Day99]

            Sequence 0: [Day0...Day59] → Target: Day60
            Sequence 1: [Day1...Day60] → Target: Day61
            Sequence 2: [Day2...Day61] → Target: Day62
            ...
        """
        Xs, ys = [], []

        # Create sliding window sequences
        for i in range(len(X) - self.lookback):
            # Extract lookback-day window
            Xs.append(X[i:(i + self.lookback)])
            # Target is the next day after the sequence
            ys.append(y[i + self.lookback])

        return np.array(Xs), np.array(ys)

    def train_and_predict(self, df, verbose=0):
        """
        Train LSTM model on historical data and predict tomorrow's volatility.

        This method implements the complete training pipeline:
        1. Feature selection and validation
        2. Data scaling (RobustScaler)
        3. Sequence creation
        4. Advanced sample weighting (time + class balance)
        5. Model training with callbacks
        6. Evaluation on test set
        7. Tomorrow's prediction

        Args:
            df (pd.DataFrame): Enhanced DataFrame from FeatureCalculator
                Must contain: RSI, MACD, ATR, Dist_MA50, Rel_Str_Peers,
                             Rel_Str_Partners, Target
            verbose (int): Training verbosity (0=silent, 1=progress bar, 2=one line per epoch)

        Returns:
            tuple or None: (probability, accuracy, metadata)
                - probability (float): Tomorrow's volatility probability (0 to 1)
                    > 0.5: Expect significant move
                    < 0.5: Expect quiet day
                - accuracy (float): Model's test set accuracy (0 to 1)
                - metadata (dict): Empty dict (reserved for future metrics)
                Returns None if training fails or insufficient data

        Advanced Weighting Strategy:

            The model uses TWO types of weights combined:

            1. Time Decay Weights:
               - Recent data is 3.0x more important than old data
               - Formula: weight = exp(t * 3.0) where t ∈ [0, 1]
               - Rationale: Market patterns change over time

            2. Class Balance Weights:
               - Handles imbalanced targets (rare volatility spikes)
               - Formula: weight_class = n_samples / (n_classes * n_samples_in_class)
               - Rationale: Model would ignore rare events without this

            3. Combined Weight:
               - final_weight = time_weight * class_weight
               - Both factors work together to focus on recent volatility patterns

        Training Process:
            1. Split: 85% train, 15% test (chronological split)
            2. Epochs: Max 35, early stopping if no improvement
            3. Batch size: 32 sequences per update
            4. Learning rate: 0.0005 with reduction on plateau
            5. Optimizer: Adam (adaptive learning rate)
            6. Loss: Binary crossentropy (log loss)

        Callbacks:
            - EarlyStopping: Stops if validation loss doesn't improve for 5 epochs
            - ReduceLROnPlateau: Reduces learning rate by 50% if stuck for 3 epochs

        Example:
            >>> lstm = MultiTimeframeLSTM()
            >>> result = lstm.train_and_predict(df_features, verbose=1)
            >>>
            >>> if result:
            >>>     prob, acc, _ = result
            >>>
            >>>     if prob > 0.5:
            >>>         print(f"⚠️ High volatility risk: {prob:.1%}")
            >>>     else:
            >>>         print(f"✅ Stable conditions expected: {(1-prob):.1%}")
            >>>
            >>>     print(f"Model accuracy: {acc:.1%}")
        """
        try:
            # ================================================================
            # 1. FEATURE SELECTION
            # ================================================================
            # Ensure Context Features are included if present
            feature_cols = ['RSI', 'MACD', 'ATR', 'Dist_MA50', 'Rel_Str_Peers', 'Rel_Str_Partners']
            available_cols = [c for c in feature_cols if c in df.columns]

            # Need at least 3 features to train
            if len(available_cols) < 3:
                print("Not enough features for training")
                return None

            # Extract features and target
            X = df[available_cols].values  # Shape: (n_days, n_features)
            y = df['Target'].values  # Shape: (n_days,)

            # ================================================================
            # 2. SCALE FEATURES
            # ================================================================
            # RobustScaler uses median and IQR, less affected by outliers
            X_scaled = self.scaler.fit_transform(X)

            # ================================================================
            # 3. CREATE SEQUENCES
            # ================================================================
            X_seq, y_seq = self.create_sequences(X_scaled, y)

            # Need at least 100 sequences to train meaningful patterns
            if len(X_seq) < 100:
                return None

            # ================================================================
            # 4. TRAIN/TEST SPLIT (Chronological)
            # ================================================================
            # Important: Always split chronologically in time series!
            # Never shuffle - this would leak future information into training
            split = int(len(X_seq) * 0.85)
            X_train, X_test = X_seq[:split], X_seq[split:]
            y_train, y_test = y_seq[:split], y_seq[split:]

            # ================================================================
            # 5. ADVANCED SAMPLE WEIGHTING
            # ================================================================

            # Weight 1: Time Decay (Recent data 3.0x more important)
            # Creates exponential curve from start to end
            t = np.linspace(0, 1, len(y_train))
            time_weights = np.exp(t * 3.0)
            time_weights /= time_weights.mean()  # Normalize to mean=1.0

            # Weight 2: Class Balance (Handle rare volatility spikes)
            classes = np.unique(y_train)
            if len(classes) > 1:
                # Compute balanced class weights
                cw = compute_class_weight('balanced', classes=classes, y=y_train)
                cw_dict = dict(zip(classes, cw))
                # Map each sample to its class weight
                sample_cw = np.array([cw_dict[c] for c in y_train])
            else:
                # All samples same class (shouldn't happen)
                sample_cw = np.ones(len(y_train))

            # Combine both weighting strategies
            final_weights = time_weights * sample_cw

            # ================================================================
            # 6. BUILD MODEL ARCHITECTURE
            # ================================================================
            model = Sequential([
                # First Bidirectional LSTM layer with return_sequences=True
                # This outputs sequences for the next layer to process
                Bidirectional(
                    LSTM(128, return_sequences=True),
                    input_shape=(self.lookback, len(available_cols))
                ),
                BatchNormalization(),  # Stabilize training
                Dropout(0.3),  # Prevent overfitting

                # Second Bidirectional LSTM layer (no return_sequences)
                # This outputs a single vector per sequence
                Bidirectional(LSTM(64)),
                BatchNormalization(),
                Dropout(0.3),

                # Dense hidden layer with swish activation
                Dense(32, activation='swish'),

                # Output layer: sigmoid for binary probability
                Dense(1, activation='sigmoid')
            ])

            # ================================================================
            # 7. COMPILE MODEL
            # ================================================================
            model.compile(
                optimizer=tf.keras.optimizers.Adam(0.0005),  # Conservative learning rate
                loss='binary_crossentropy',  # Log loss for binary classification
                metrics=['accuracy']  # Track accuracy
            )

            # ================================================================
            # 8. TRAIN MODEL WITH CALLBACKS
            # ================================================================
            model.fit(
                X_train, y_train,
                epochs=35,  # Maximum 35 epochs
                batch_size=32,  # 32 sequences per update
                validation_data=(X_test, y_test),  # Monitor test performance
                sample_weight=final_weights,  # Apply our advanced weighting
                callbacks=[
                    # Stop early if validation loss doesn't improve for 5 epochs
                    EarlyStopping(patience=5, restore_best_weights=True),
                    # Reduce learning rate if stuck for 3 epochs
                    ReduceLROnPlateau(factor=0.5, patience=3)
                ],
                verbose=verbose  # 0=silent, 1=progress bar, 2=one line per epoch
            )

            self.model = model

            # ================================================================
            # 9. EVALUATE ON TEST SET
            # ================================================================
            preds = model.predict(X_test, verbose=0)
            pred_classes = (preds > 0.5).astype(int)
            acc = accuracy_score(y_test, pred_classes)

            # ================================================================
            # 10. PREDICT TOMORROW
            # ================================================================
            # Take the last 60 days as input sequence
            last_seq = X_scaled[-self.lookback:].reshape(1, self.lookback, len(available_cols))
            prob = model.predict(last_seq, verbose=0)[0][0]

            return prob, acc, {}

        except Exception as e:
            print(f"LSTM Training Error: {e}")
            return None

    def predict_next(self, df):
        """
        Helper method for existing calls (if any).

        This method is a placeholder for compatibility with older code
        that may be calling predict_next() directly.

        Args:
            df: Input data

        Returns:
            None (not implemented, use train_and_predict instead)

        Note:
            Use train_and_predict() method instead, which handles
            the complete training and prediction pipeline.
        """
        pass