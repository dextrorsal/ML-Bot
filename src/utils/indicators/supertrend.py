# src/utils/inidcators/supertrend.py

import numpy as np
import pandas as pd

def atr(high, low, close, length=10):
    """
    Calculate ATR as a rolling mean of the True Range.
    Returns a NumPy array.
    """
    # Convert inputs to Pandas Series if needed, then to NumPy
    if not isinstance(high, pd.Series):
        high = pd.Series(high)
    if not isinstance(low, pd.Series):
        low = pd.Series(low)
    if not isinstance(close, pd.Series):
        close = pd.Series(close)

    tr = np.maximum(
        high - low,
        np.maximum((high - close.shift()).abs(), (low - close.shift()).abs())
    )
    # Rolling mean -> convert final result to NumPy
    return tr.rolling(window=length).mean().to_numpy()


def supertrend(high, low, close, atr_length=10, factor=3):
    """
    Standard Supertrend calculation.
    Returns:
      - supertrend line (NumPy array)
      - direction array (NumPy array of 1 or -1)
    """
    # Convert to NumPy arrays for consistent integer indexing
    high_arr = np.array(high)
    low_arr = np.array(low)
    close_arr = np.array(close)

    atr_values = atr(high_arr, low_arr, close_arr, atr_length)
    hl2 = (high_arr + low_arr) / 2

    upper_band = hl2 + factor * atr_values
    lower_band = hl2 - factor * atr_values

    st = np.zeros_like(close_arr)
    direction = np.ones_like(close_arr)

    for i in range(1, len(close_arr)):
        if close_arr[i - 1] > upper_band[i - 1]:
            direction[i] = -1
        elif close_arr[i - 1] < lower_band[i - 1]:
            direction[i] = 1
        else:
            direction[i] = direction[i - 1]
            if direction[i] == -1:
                # Use Python's built-in max/min, not np.max/min, to compare single floats
                lower_band[i] = max(lower_band[i], lower_band[i - 1])
            else:
                upper_band[i] = min(upper_band[i], upper_band[i - 1])

        st[i] = lower_band[i] if direction[i] == -1 else upper_band[i]

    return st, direction


def k_means_volatility(atr_values, training_period=100, highvol=0.75, midvol=0.50, lowvol=0.25):
    """
    Assign each bar to one of three volatility clusters (high, medium, low)
    based on ATR values.
    Returns:
      - clusters (NumPy int array)
      - centroids (list of NumPy arrays)
    """
    # Ensure atr_values is a NumPy array
    if isinstance(atr_values, pd.Series):
        atr_values = atr_values.to_numpy()

    # We'll use a rolling max/min but do it with Pandas, then convert to arrays
    s = pd.Series(atr_values)
    upper_series = s.rolling(window=training_period).max()
    lower_series = s.rolling(window=training_period).min()

    upper = upper_series.to_numpy()
    lower = lower_series.to_numpy()

    high_volatility = lower + (upper - lower) * highvol
    medium_volatility = lower + (upper - lower) * midvol
    low_volatility = lower + (upper - lower) * lowvol

    # Ensure clusters is an int array
    clusters = np.zeros(len(atr_values), dtype=int)
    centroids = [
        high_volatility,
        medium_volatility,
        low_volatility
    ]

    for i in range(len(atr_values)):
        # c[i] is also valid because c is a NumPy array
        distances = [abs(atr_values[i] - c[i]) for c in centroids]
        clusters[i] = np.argmin(distances)

    return clusters, centroids


def adaptive_supertrend(high, low, close,
                        atr_length=10, factor=3,
                        training_period=100, highvol=0.75, midvol=0.50, lowvol=0.25):
    """
    Adaptive Supertrend that first clusters volatility using k_means_volatility,
    then calls the standard Supertrend function.
    """
    # 1. Compute ATR as a NumPy array
    atr_values = atr(high, low, close, atr_length)

    # 2. Cluster volatility
    clusters, centroids = k_means_volatility(atr_values,
                                             training_period=training_period,
                                             highvol=highvol,
                                             midvol=midvol,
                                             lowvol=lowvol)

    # 3. assigned_centroid is optional. If you don't need it, comment it out.
    #    This line can cause "list indices must be integers or slices" if
    #    there's any mismatch. Now that we've converted everything to arrays,
    #    it should work:
    assigned_centroid = np.array([
        centroids[c][i] for i, c in enumerate(clusters)
    ])

    # 4. Return standard supertrend
    return supertrend(high, low, close, atr_length, factor)


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    data = pd.DataFrame({
        "high": np.random.rand(200) * 100,
        "low": np.random.rand(200) * 100,
        "close": np.random.rand(200) * 100
    })

    st_values, st_direction = adaptive_supertrend(data['high'], data['low'], data['close'])

    print("Supertrend values shape:", st_values.shape)
    print("Direction array shape:  ", st_direction.shape)
    print("First 10 directions:    ", st_direction[:10])
