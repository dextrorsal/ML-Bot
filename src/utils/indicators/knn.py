# src/utils/indicators/knn.py

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

class knnStrategy:
    def __init__(self, short_period=14, long_period=28, base_neighbors=252, 
                 bar_threshold=300, volatility_filter=False):
        """
        K-Nearest Neighbors trading strategy implementation.

        Parameters:
        -----------
        short_period : int
            The short period for technical indicator calculation.
        long_period : int
            The long period for technical indicator calculation.
        base_neighbors : int
            Maximum number of neighbors to use in knn algorithm.
        bar_threshold : int
            Minimum number of bars needed before generating signals.
        volatility_filter : bool
            Whether to apply volatility filtering.
        """
        self.short_period = short_period
        self.long_period = long_period
        self.base_neighbors = base_neighbors
        self.bar_threshold = bar_threshold
        self.volatility_filter = volatility_filter
        self.scaler = StandardScaler()
        self.model = None  # We'll create it dynamically in 'calculate()'

    def calculate_features(self, prices, volumes):
        """Calculate technical features for knn."""
        df = pd.DataFrame({'close': prices, 'volume': volumes})

        # Price momentum
        df['returns'] = df['close'].pct_change()
        df['mom_short'] = df['close'].pct_change(self.short_period)
        df['mom_long'] = df['close'].pct_change(self.long_period)

        # Volume features
        df['vol_short'] = df['volume'].rolling(self.short_period).mean()
        df['vol_long'] = df['volume'].rolling(self.long_period).mean()

        # Volatility
        df['volatility'] = df['returns'].rolling(self.long_period).std()

        # Price levels
        df['price_sma_short'] = df['close'].rolling(self.short_period).mean()
        df['price_sma_long'] = df['close'].rolling(self.long_period).mean()

        # Instead of dropping rows with NaN, skip the first <long_period> bars
        start_idx = self.long_period
        return df.iloc[start_idx:]

    def generate_labels(self, df):
        """Generate training labels based on future returns (shift -1)."""
        future_returns = df['close'].pct_change().shift(-1)
        labels = np.where(future_returns > 0, 1, -1)
        return labels[:-1]  # remove last point, no future return

    def calculate(self, prices, volumes):
        """
        Calculate a trading signal for the latest bar using knn.

        Returns:
            -1 (sell), 0 (hold), 1 (buy), or None if insufficient data.
        """
        try:
            # Basic data length check
            if len(prices) < self.long_period + 10:
                return None

            # Build feature DataFrame
            features_df = self.calculate_features(prices, volumes)
            if len(features_df) < 2:
                return None

            # Choose which columns to feed to the knn
            feature_cols = [
                'mom_short', 'mom_long', 'vol_short', 'vol_long', 
                'volatility', 'price_sma_short', 'price_sma_long'
            ]
            X = features_df[feature_cols].values

            # Scale the features
            X_scaled = self.scaler.fit_transform(X)

            # Generate labels
            y = self.generate_labels(features_df)
            if len(X_scaled) < 2 or len(y) < 1:
                return None

            # Figure out how many neighbors we can effectively use
            effective_neighbors = min(self.base_neighbors, len(X_scaled[:-1]))
            if effective_neighbors < 1:
                return None

            # Build and train the knn
            self.model = KNeighborsClassifier(n_neighbors=effective_neighbors)
            # Train on all but the last row
            self.model.fit(X_scaled[:-1], y)

            # Predict on the last row
            latest_scaled = X_scaled[-1].reshape(1, -1)
            prediction = self.model.predict(latest_scaled)[0]

            # Optional volatility filter
            if self.volatility_filter:
                current_vol = features_df['volatility'].iloc[-1]
                vol_avg = features_df['volatility'].mean()
                # If volatility is more than double the average, override to 0
                if current_vol > vol_avg * 2:
                    return 0

            return prediction

        except Exception as e:
            print(f"Error in knn calculation: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
