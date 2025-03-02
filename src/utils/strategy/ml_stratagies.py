# src/strategy/ml_strategies.py
"""
ML-focused trading strategies that provide rich feature extraction
and adaptive mechanisms for machine learning integration.
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from .base import BaseStrategy


class EnhancedMultiIndicatorStrategy(BaseStrategy):
    """
    Strategy with dynamic weight adaptation and feature extraction capabilities,
    designed for ML compatibility and continuous learning.
    """
    
    def __init__(self, config: Dict, indicators: List, weights: Optional[List[float]] = None) -> None:
        """
        Initialize the enhanced strategy with adaptive weighting capabilities.
        
        Parameters:
            config: Configuration dictionary with strategy parameters
            indicators: List of indicator objects implementing generate_signals(df)
            weights: Optional custom weights for indicators (default: auto-assigned)
        """
        super().__init__(config)
        self.indicators = indicators
        self.weights = self._initialize_weights(indicators, weights)
        
        # Configuration parameters
        self.threshold = self.config.get("consensus_threshold", 0)
        self.extract_features = self.config.get("extract_features", False)
        self.adaptive_weights = self.config.get("adaptive_weights", False)
        self.lookback = self.config.get("lookback", 20)
        self.weight_history = []  # Track weight changes over time
        
        # Performance tracking
        self.performance_metrics = {
            'indicator_correlations': [],
            'weight_adjustments': [],
            'signal_quality': []
        }
        
    def _initialize_weights(self, indicators: List, weights: Optional[List[float]]) -> List[float]:
        """
        Initialize indicator weights based on provided values or defaults.
        
        Parameters:
            indicators: List of indicator objects
            weights: Optional custom weights
            
        Returns:
            List of initialized weights
        """
        if weights is not None:
            if len(weights) != len(indicators):
                raise ValueError("Length of weights must match number of indicators.")
            return weights.copy()  # Return copy to avoid external modification
        
        # Default weights based on indicator types
        if len(indicators) == 4:
            # Default priority: Lorentzian:2, Supertrend:2, Logistic:1, KNN:1
            return [2, 2, 1, 1]
        
        # Equal weights for other configurations
        return [1] * len(indicators)
        
    def _adjust_weights(self, df: pd.DataFrame) -> None:
        """
        Dynamically adjust indicator weights based on recent performance.
        
        This method evaluates each indicator's predictive power over recent data
        and increases weights for indicators showing stronger predictive ability.
        
        Parameters:
            df: DataFrame with market data
        """
        if not self.adaptive_weights or len(df) < self.lookback * 2:
            return
            
        # Store original weights for tracking changes
        original_weights = self.weights.copy()
        
        # Get recent price movement direction
        recent_return = df['close'].pct_change().iloc[-self.lookback:].sum()
        
        # Track correlation data for performance analysis
        correlations = []
        adjustments = []
        
        # Assess each indicator's recent performance
        for i, indicator in enumerate(self.indicators):
            # Generate signals on recent data window
            recent_data = df.iloc[-self.lookback*2:]
            signals = indicator.generate_signals(recent_data)
            
            # Calculate correlation with future returns (predictive power)
            future_returns = df['close'].pct_change().shift(-1).iloc[-self.lookback*2+1:]
            signal_effectiveness = signals.iloc[:-1].corr(future_returns)
            correlations.append(signal_effectiveness)
            
            # Adjust weight based on recent effectiveness
            if not pd.isna(signal_effectiveness):
                direction = 1 if recent_return > 0 else -1
                adjustment = abs(signal_effectiveness) * 0.5  # Scale adjustment
                
                # Increase weight if indicator is correctly predicting direction
                if (signal_effectiveness > 0 and direction > 0) or (signal_effectiveness < 0 and direction < 0):
                    self.weights[i] *= (1 + adjustment)
                    adjustments.append(adjustment)
                else:
                    self.weights[i] *= (1 - adjustment)
                    adjustments.append(-adjustment)
            else:
                adjustments.append(0)
        
        # Normalize weights to maintain scale
        total = sum(self.weights)
        if total > 0:  # Avoid division by zero
            self.weights = [w/total * len(self.weights) for w in self.weights]
        
        # Store performance metrics
        self.performance_metrics['indicator_correlations'].append(correlations)
        self.performance_metrics['weight_adjustments'].append(
            [(new - old) for old, new in zip(original_weights, self.weights)]
        )
        
        # Store weight history for analysis
        self.weight_history.append(self.weights.copy())
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals and optionally extract ML-compatible features.
        
        Parameters:
            df: DataFrame with OHLCV market data
            
        Returns:
            pd.Series: Trading signals (1=long, -1=short, 0=neutral)
        """
        # Adjust weights if adaptive mode is enabled
        if self.adaptive_weights:
            self._adjust_weights(df)
            
        # Get individual indicator signals
        indicator_signals = []
        for indicator in self.indicators:
            signals = indicator.generate_signals(df)
            indicator_signals.append(signals)
        
        # Apply weights to signals
        weighted_signals = [signals * weight for signals, weight in zip(indicator_signals, self.weights)]
        
        # If no signals are produced, default to a neutral series
        if not weighted_signals:
            return pd.Series(0, index=df.index)
        
        # Combine weighted signals
        combined_signals_df = pd.concat(weighted_signals, axis=1)
        summed_signals = combined_signals_df.sum(axis=1)
        
        # Final signals based on threshold
        final_signals = summed_signals.apply(
            lambda x: 1 if x > self.threshold else (-1 if x < -self.threshold else 0)
        )
        
        # Generate additional ML-compatible features if requested
        if self.extract_features:
            # Create feature set for ML training
            feature_set = pd.DataFrame(index=df.index)
            
            # Include raw indicators and metadata
            for i, (indicator, signals) in enumerate(zip(self.indicators, indicator_signals)):
                feature_set[f'indicator_{i}_raw'] = signals
                feature_set[f'indicator_{i}_weight'] = self.weights[i]
            
            # Add consensus info
            feature_set['sum_signal'] = summed_signals
            feature_set['signal_confidence'] = abs(summed_signals)
            feature_set['signal_direction'] = summed_signals.apply(
                lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
            )
            
            # Store features for later ML processing
            self.feature_set = feature_set
            
        # Track signal quality for performance analysis
        if len(df) > 1:
            # Calculate if signals correctly predicted next bar's direction
            returns = df['close'].pct_change().shift(-1)
            signal_quality = (final_signals * returns).iloc[:-1]  # Exclude last bar (no future return)
            self.performance_metrics['signal_quality'].append(signal_quality.mean())
            
        return final_signals
    
    def get_performance_metrics(self) -> Dict:
        """
        Get performance metrics for analysis and ML model training.
        
        Returns:
            Dict: Performance metrics including indicator correlations and signal quality
        """
        if not self.performance_metrics['signal_quality']:
            return {'error': 'No performance data available yet'}
            
        return {
            'avg_signal_quality': np.mean(self.performance_metrics['signal_quality']),
            'weight_stability': self._calculate_weight_stability(),
            'indicator_performance': self._calculate_indicator_performance(),
            'current_weights': self.weights
        }
    
    def _calculate_weight_stability(self) -> float:
        """
        Calculate the stability of weights over time (lower values = more stable).
        
        Returns:
            float: Weight stability metric
        """
        if len(self.weight_history) < 2:
            return 1.0  # Perfect stability if weights haven't changed
            
        # Calculate standard deviation of each weight over time
        weight_variations = []
        for i in range(len(self.weights)):
            weight_over_time = [weights[i] for weights in self.weight_history]
            variation = np.std(weight_over_time) if len(weight_over_time) > 1 else 0
            weight_variations.append(variation)
            
        # Average variation across all weights
        return np.mean(weight_variations)
    
    def _calculate_indicator_performance(self) -> List[Dict]:
        """
        Calculate performance metrics for each indicator.
        
        Returns:
            List[Dict]: Performance data for each indicator
        """
        if not self.performance_metrics['indicator_correlations']:
            return []
            
        result = []
        for i in range(len(self.indicators)):
            correlations = [corrs[i] for corrs in self.performance_metrics['indicator_correlations'] 
                           if i < len(corrs) and not pd.isna(corrs[i])]
            
            adjustments = [adjs[i] for adjs in self.performance_metrics['weight_adjustments'] 
                          if i < len(adjs)]
            
            result.append({
                'index': i,
                'avg_correlation': np.mean(correlations) if correlations else 0,
                'correlation_stability': np.std(correlations) if len(correlations) > 1 else 0,
                'net_weight_change': sum(adjustments),
                'current_weight': self.weights[i] if i < len(self.weights) else 0
            })
            
        return result


class FeatureExtractionStrategy(BaseStrategy):
    """
    Creates rich feature sets from indicators specifically designed for ML model training.
    This strategy focuses on extracting valuable features rather than optimizing trading performance.
    """
    
    def __init__(self, config: Dict, indicators: List) -> None:
        """
        Initialize feature extraction strategy.
        
        Parameters:
            config: Configuration dictionary with strategy parameters
            indicators: List of indicator objects implementing generate_signals(df)
        """
        super().__init__(config)
        self.indicators = indicators
        self.feature_window = self.config.get("feature_window", 10)
        
        # Advanced configuration
        self.include_technical_features = self.config.get("include_technical_features", True)
        self.include_price_features = self.config.get("include_price_features", True)
        self.include_volume_features = self.config.get("include_volume_features", True)
        self.normalize_features = self.config.get("normalize_features", False)
        
        # Track feature importance
        self.feature_importance = {}
        
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate basic trading signals and prepare comprehensive feature dataset for ML.
        
        Parameters:
            df: DataFrame with OHLCV market data
            
        Returns:
            pd.Series: Basic trading signals (1=long, -1=short, 0=neutral)
        """
        if len(df) < self.feature_window:
            return pd.Series(0, index=df.index)
            
        # Get raw signals from all indicators
        indicator_signals = []
        for indicator in self.indicators:
            signals = indicator.generate_signals(df)
            indicator_signals.append(signals)
            
        # Simple consensus for basic signals (unweighted)
        if indicator_signals:
            combined_df = pd.concat(indicator_signals, axis=1)
            consensus = combined_df.sum(axis=1)
            signals = consensus.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        else:
            signals = pd.Series(0, index=df.index)
            
        # Create feature matrix for ML
        features = pd.DataFrame(index=df.index)
        
        # 1. Raw indicator signals
        for i, ind_signals in enumerate(indicator_signals):
            features[f'ind_{i}_signal'] = ind_signals
            
        # 2. Rolling statistics of each indicator
        for i, ind_signals in enumerate(indicator_signals):
            if len(ind_signals) >= self.feature_window:
                features[f'ind_{i}_mean'] = ind_signals.rolling(self.feature_window).mean()
                features[f'ind_{i}_std'] = ind_signals.rolling(self.feature_window).std()
                features[f'ind_{i}_max'] = ind_signals.rolling(self.feature_window).max()
                features[f'ind_{i}_min'] = ind_signals.rolling(self.feature_window).min()
                
        # 3. Signal change frequency
        for i, ind_signals in enumerate(indicator_signals):
            features[f'ind_{i}_changes'] = ind_signals.diff().abs().rolling(self.feature_window).sum()
            
        # 4. Indicator agreement ratio
        if len(indicator_signals) > 1:
            agreement_ratios = []
            for i in range(len(df)):
                if i >= self.feature_window:
                    agreement_count = 0
                    total_pairs = 0
                    for j in range(len(indicator_signals)):
                        for k in range(j+1, len(indicator_signals)):
                            total_pairs += 1
                            if indicator_signals[j].iloc[i] == indicator_signals[k].iloc[i]:
                                agreement_count += 1
                    agreement_ratio = agreement_count / total_pairs if total_pairs > 0 else 0
                    agreement_ratios.append(agreement_ratio)
                else:
                    agreement_ratios.append(None)
            features['indicator_agreement'] = agreement_ratios
            
        # 5. Price-based features (if enabled)
        if self.include_price_features and len(df) >= self.feature_window:
            # Momentum features
            for window in [self.feature_window, self.feature_window*2, self.feature_window//2]:
                if window > 0:  # Ensure positive window size
                    features[f'price_momentum_{window}'] = df['close'].pct_change(window)
            
            # Volatility features
            features['price_volatility'] = df['close'].pct_change().rolling(self.feature_window).std()
            
            # Price relative to moving averages
            for ma_period in [self.feature_window, self.feature_window*2, 50]:
                if len(df) >= ma_period:
                    ma = df['close'].rolling(ma_period).mean()
                    features[f'price_rel_ma_{ma_period}'] = df['close'] / ma - 1
            
            # High-low range relative to historical
            hl_range = (df['high'] - df['low']) / df['close']
            features['hl_range_relative'] = hl_range / hl_range.rolling(self.feature_window*2).mean()
            
            # Distance from highest high and lowest low
            features['dist_from_high'] = df['close'] / df['high'].rolling(self.feature_window).max() - 1
            features['dist_from_low'] = df['close'] / df['low'].rolling(self.feature_window).min() - 1
            
        # 6. Volume-based features (if enabled)
        if self.include_volume_features and 'volume' in df.columns:
            # Volume change
            features['volume_change'] = df['volume'].pct_change()
            
            # Volume relative to average
            vol_ma = df['volume'].rolling(self.feature_window).mean()
            features['volume_rel_avg'] = df['volume'] / vol_ma
            
            # Volume acceleration
            features['volume_accel'] = features['volume_change'].diff()
            
            # Price-volume correlation
            for window in [self.feature_window, self.feature_window*2]:
                if len(df) >= window:
                    # Rolling correlation between price changes and volume
                    price_changes = df['close'].pct_change()
                    vol_price_corr = price_changes.rolling(window).corr(df['volume'])
                    features[f'price_vol_corr_{window}'] = vol_price_corr
            
        # Normalize features if requested
        if self.normalize_features:
            for col in features.columns:
                if col not in ['indicator_agreement']:  # Skip columns that are already normalized
                    features[col] = self._normalize_feature(features[col])
        
        # Store features for ML use
        self.feature_matrix = features.copy()
        
        return signals
    
    def _normalize_feature(self, series: pd.Series) -> pd.Series:
        """
        Normalize a feature series using min-max scaling.
        
        Parameters:
            series: Series to normalize
            
        Returns:
            pd.Series: Normalized series
        """
        min_val = series.min()
        max_val = series.max()
        
        if min_val == max_val:
            return pd.Series(0, index=series.index)
            
        return (series - min_val) / (max_val - min_val)
    
    def get_feature_matrix(self) -> pd.DataFrame:
        """
        Get the generated feature matrix for ML model training.
        
        Returns:
            pd.DataFrame: Feature matrix
        """
        if not hasattr(self, 'feature_matrix'):
            return pd.DataFrame()
        return self.feature_matrix
    
    def set_feature_importance(self, importance_dict: Dict[str, float]) -> None:
        """
        Set importance scores for features (from external ML model).
        
        Parameters:
            importance_dict: Dictionary mapping feature names to importance scores
        """
        self.feature_importance = importance_dict
    
    def get_top_features(self, n: int = 10) -> List[Tuple[str, float]]:
        """
        Get the top n most important features.
        
        Parameters:
            n: Number of top features to return
            
        Returns:
            List[Tuple[str, float]]: List of (feature_name, importance_score) pairs
        """
        if not self.feature_importance:
            return []
            
        # Sort features by importance and get top n
        sorted_features = sorted(self.feature_importance.items(), 
                                key=lambda x: x[1], reverse=True)
        return sorted_features[:n]