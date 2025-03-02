# src/utils/indicators/lorentzian.py

import math  # Ensure this is imported at the top
import numpy as np
import pandas as pd
import talib
from dataclasses import dataclass
from typing import List, Dict, Tuple

@dataclass
class Feature:
    type: str
    parameter_a: int
    parameter_b: int

@dataclass
class LorentzianSettings:
    # General Settings
    source: str = "close"
    neighbors_count: int = 8
    max_bars_back: int = 2000
    
    # Feature Engineering
    feature_count: int = 5
    color_compression: int = 1
    
    # Filters
    use_volatility_filter: bool = True
    use_regime_filter: bool = False
    use_adx_filter: bool = False
    use_ema_filter: bool = False
    use_sma_filter: bool = False

    # Additional fields
    regime_threshold: float = -0.1
    adx_threshold: int = 20
    ema_period: int = 200
    sma_period: int = 200
    use_dynamic_exits: bool = False
    use_worst_case_estimates: bool = False
    enhance_kernel_smoothing: bool = False
    lag: int = 1

class LorentzianClassification:
    """
    A unified LorentzianClassification that:
      1) Calculates TA-based features (RSI, WT, CCI, ADX).
      2) Can generate signals using an ML-core approach (kernel regression + kNN).
      3) Has a `.run()` method for your wrapper_lorentzian.py.
    """

    def __init__(self, settings: LorentzianSettings = None):
        self.settings = settings or LorentzianSettings()
        
        # Features for TA-based calculations
        self.features = [
            Feature("RSI", 14, 1),
            Feature("WT", 10, 11),
            Feature("CCI", 20, 1),
            Feature("ADX", 20, 2),
            Feature("RSI", 9, 1)
        ]

    # ----------------------------------------------------------------
    # (A) TA-Based Feature Calculations
    # ----------------------------------------------------------------
    def calculate_rsi_feature(self, data: pd.DataFrame, period: int, parameter_b: int) -> pd.Series:
        """Calculate RSI feature."""
        return talib.RSI(data['close'], timeperiod=period)

    def calculate_wt_feature(self, data: pd.DataFrame, parameter_a: int, parameter_b: int) -> pd.Series:
        """Calculate Wave Trend feature."""
        hlc3 = (data['high'] + data['low'] + data['close']) / 3
        esa = talib.EMA(hlc3, timeperiod=parameter_a)
        d = talib.EMA(abs(hlc3 - esa), timeperiod=parameter_a)
        ci = (hlc3 - esa) / (0.015 * d)
        wt = talib.EMA(ci, timeperiod=parameter_b)
        return wt

    def calculate_cci_feature(self, data: pd.DataFrame, period: int, parameter_b: int) -> pd.Series:
        """Calculate CCI feature."""
        return talib.CCI(data['high'], data['low'], data['close'], timeperiod=period)

    def calculate_adx_feature(self, data: pd.DataFrame, period: int, parameter_b: int) -> pd.Series:
        """Calculate ADX feature."""
        return talib.ADX(data['high'], data['low'], data['close'], timeperiod=period)

    def calculate_feature(self, feature: Feature, data: pd.DataFrame) -> pd.Series:
        """Calculate a single feature based on Feature(type, parameter_a, parameter_b)."""
        if feature.type == "RSI":
            return self.calculate_rsi_feature(data, feature.parameter_a, feature.parameter_b)
        elif feature.type == "WT":
            return self.calculate_wt_feature(data, feature.parameter_a, feature.parameter_b)
        elif feature.type == "CCI":
            return self.calculate_cci_feature(data, feature.parameter_a, feature.parameter_b)
        elif feature.type == "ADX":
            return self.calculate_adx_feature(data, feature.parameter_a, feature.parameter_b)
        else:
            raise ValueError(f"Unknown feature type: {feature.type}")

    def calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all features (RSI, WT, CCI, ADX, etc.) and return a DataFrame."""
        features_df = pd.DataFrame(index=data.index)
        for i, feat in enumerate(self.features):
            features_df[f'feature_{i+1}'] = self.calculate_feature(feat, data)
        return features_df

    # ----------------------------------------------------------------
    # (B) Filters & Lorentzian Distance
    # ----------------------------------------------------------------
    def apply_volatility_filter(self, data: pd.DataFrame) -> pd.Series:
        """Return True/False for each bar based on whether ATR(14) > ATR(28)."""
        atr14 = talib.ATR(data['high'], data['low'], data['close'], timeperiod=14)
        atr28 = talib.ATR(data['high'], data['low'], data['close'], timeperiod=28)
        return atr14 > atr28

    def calculate_lorentzian_distance(self, features: pd.DataFrame, idx: int) -> float:
        """
        Compute the Lorentzian distance between the most recent feature vector (the last row of features)
        and the historical feature vector at index `idx` by summing the logarithm of (1 + absolute difference)
        for each feature. This emulates the Pine Script logic.
        """
        if idx >= len(features):
            return float('inf')
        
        current_features = features.iloc[-1]
        historical_features = features.iloc[idx]
        
        distance = 0.0
        for col in features.columns:
            distance += math.log(1 + abs(current_features[col] - historical_features[col]))
        
        return distance

    # ----------------------------------------------------------------
    # (C) ML-Core: Kernel Regression, KNN, etc.
    # ----------------------------------------------------------------
    def calculate_kernel_regression(self, data: pd.DataFrame, h: int = 8, r: float = 8.0, x: int = 25) -> pd.Series:
        """
        Example Nadaraya-Watson Kernel Regression with Rational Quadratic Kernel.
        """
        kernel_sum = np.zeros(len(data))
        weight_sum = np.zeros(len(data))
        for i in range(len(data)):
            if i < h:
                continue
            for j in range(max(0, i - h), i):
                time_diff = i - j
                weight = (1 + (time_diff ** 2) / (2 * r * x)) ** (-r)
                kernel_sum[i] += weight * data['close'].iloc[j]
                weight_sum[i] += weight

        kernel_est = kernel_sum / np.where(weight_sum == 0, 1, weight_sum)
        return pd.Series(kernel_est, index=data.index)

    def calculate_training_labels(self, data: pd.DataFrame, lookforward: int = 4) -> np.ndarray:
        """
        Label each bar as +1 if future price is higher after 'lookforward' bars,
        otherwise -1.
        """
        labels = np.zeros(len(data))
        for i in range(len(data) - lookforward):
            if data['close'].iloc[i + lookforward] > data['close'].iloc[i]:
                labels[i] = 1
            else:
                labels[i] = -1
        return labels

    def find_nearest_neighbors(self, features: pd.DataFrame, current_idx: int) -> List[Tuple[float, int]]:
        """
        KNN approach: find the k-nearest neighbors using 'calculate_lorentzian_distance'.
        """
        distances = []
        max_bars_back = min(current_idx, self.settings.max_bars_back)
        for i in range(max(0, current_idx - max_bars_back), current_idx):
            if i % 4 == 0:  # skip some bars for speed
                dist = self.calculate_lorentzian_distance(features, i)
                distances.append((dist, i))
        distances.sort(key=lambda x: x[0])
        return distances[: self.settings.neighbors_count]

    def predict_next_bar(self, features: pd.DataFrame, labels: np.ndarray, current_idx: int) -> float:
        """
        Return an average of the neighbors' labels => +1 or -1 if bullish/bearish.
        """
        if current_idx < self.settings.max_bars_back:
            return 0
        neighbors = self.find_nearest_neighbors(features, current_idx)
        preds = [labels[n_idx] for _, n_idx in neighbors]
        return np.mean(preds)

    # ============= New Filters =============
    def apply_regime_filter(self, data: pd.DataFrame) -> pd.Series:
        """
        Simple example: slope over 5 bars > regime_threshold => True, else False.
        """
        threshold = self.settings.regime_threshold
        slope = data['close'].diff(5)
        return slope > threshold

    def apply_adx_filter(self, data: pd.DataFrame) -> pd.Series:
        """Return True if ADX >= adx_threshold, else False."""
        adx_vals = talib.ADX(data['high'], data['low'], data['close'], timeperiod=14)
        return adx_vals >= self.settings.adx_threshold

    def apply_ema_filter(self, data: pd.DataFrame) -> pd.Series:
        """
        Return True if close > EMA(ema_period), else False
        for an uptrend-only filter.
        """
        ema_vals = talib.EMA(data['close'], timeperiod=self.settings.ema_period)
        return data['close'] > ema_vals

    def apply_sma_filter(self, data: pd.DataFrame) -> pd.Series:
        sma_vals = talib.SMA(data['close'], timeperiod=self.settings.sma_period)
        return data['close'] > sma_vals

    def apply_dynamic_exits(self, signals: pd.Series, data: pd.DataFrame) -> pd.Series:
        """
        Stub: If use_dynamic_exits is True, we can do a 4-bar exit or kernel-based exit logic.
        For now, just return signals as-is.
        """
        # Example logic might track how many bars since the last signal
        # and force neutral after 4 bars, or watch for kernel cross, etc.
        return signals

    # ----------------------------------------------------------------
    # (D) generate_signals_ml & run
    # ----------------------------------------------------------------
    def generate_signals_ml(self, data: pd.DataFrame) -> pd.Series:
        # 1) Calculate features, training labels, and kernel estimate
        feats = self.calculate_features(data)
        labels = self.calculate_training_labels(data)
        kernel_est = self.calculate_kernel_regression(data)

        predictions = pd.Series(0, index=data.index, dtype=float)
        
        # 2) KNN-like approach with Lorentzian distance
        for i in range(len(data)):
            if i < self.settings.max_bars_back:
                predictions.iloc[i] = 0
            else:
                neighbors = self.find_nearest_neighbors(feats, i)
                if not neighbors:
                    predictions.iloc[i] = 0
                    continue
                
                neighbor_distances = [dist for dist, _ in neighbors]
                mean_dist = np.mean(neighbor_distances)
                std_dist = np.std(neighbor_distances)
                
                current_distance = self.calculate_lorentzian_distance(feats, i)
                
                # Voting logic
                if current_distance < mean_dist - std_dist:
                    predictions.iloc[i] = 1
                elif current_distance > mean_dist + std_dist:
                    predictions.iloc[i] = -1
                else:
                    predictions.iloc[i] = 0

        # 3) Apply filters
        if self.settings.use_volatility_filter:
            vol_filter = self.apply_volatility_filter(data)
            predictions = predictions.where(vol_filter, 0)

        # Kernel trend up
        kernel_filter = (kernel_est > kernel_est.shift(1))
        predictions = predictions.where(kernel_filter, 0)

        # Additional filters
        if self.settings.use_regime_filter:
            regime_filter = self.apply_regime_filter(data)
            predictions = predictions.where(regime_filter, 0)

        if self.settings.use_adx_filter:
            adx_filter = self.apply_adx_filter(data)
            predictions = predictions.where(adx_filter, 0)

        if self.settings.use_ema_filter:
            ema_filter = self.apply_ema_filter(data)
            predictions = predictions.where(ema_filter, 0)

        if self.settings.use_sma_filter:
            sma_filter = self.apply_sma_filter(data)
            predictions = predictions.where(sma_filter, 0)

        # Convert float predictions to discrete signals
        signals = predictions.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

        # 4) Handle dynamic exits
        if self.settings.use_dynamic_exits:
            signals = self.apply_dynamic_exits(signals, data)

        return signals

    def run(self, data: pd.DataFrame) -> Dict:
        """
        The default "run" method. 
        We'll assume you want the ML-based approach by default.
        """
        signals = self.generate_signals_ml(data)
        kernel = self.calculate_kernel_regression(data)
        feats = self.calculate_features(data)
        return {
            'signals': signals,
            'kernel_estimate': kernel,
            'features': feats
        }
