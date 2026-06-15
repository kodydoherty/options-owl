"""Multi-timeframe support level detection via wick clustering.

Support is NOT just "the lowest low" — it's a price level where multiple candles
refused to drop below, as shown by wick bottoms clustering near the same level.

Algorithm:
1. For each timeframe (5m, 15m, 1h, 4h), collect candle lows (wick bottoms)
2. Cluster nearby lows within a tolerance band (% of price)
3. Rank clusters by touch count (more touches = stronger support)
4. Score multi-timeframe confluence (same zone on multiple TFs = strongest)

Usage::

    from options_owl.collectors.support_levels import find_support_levels
    from options_owl.collectors.candle_cache import CandleBar

    levels = find_support_levels(candle_data, current_price=605.50)
    # levels = [SupportLevel(price=604.80, strength=8, timeframes=['5m','15m','1h'], ...)]

    at_support = any(lv.distance_pct <= 0.3 for lv in levels if lv.strength >= 3)
"""

from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class SupportLevel:
    """A detected support zone."""
    price: float                    # center of the cluster
    strength: int                   # total touch count across all timeframes
    timeframes: list[str]           # which TFs contributed
    touches_per_tf: dict[str, int]  # e.g. {"5m": 4, "15m": 2, "1h": 1}
    distance_pct: float             # distance from current price (%)
    confluence: int                 # number of timeframes (more = stronger)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Tolerance for clustering wick lows (% of price).
# Two lows within this band are considered "same level".
CLUSTER_TOLERANCE_PCT = 0.15

# Minimum touches required for a valid support level per timeframe
MIN_TOUCHES_PER_TF = 2

# How many candles to look back per timeframe
LOOKBACK = {
    "5m": 36,    # 3 hours of 5m candles
    "15m": 20,   # 5 hours of 15m candles
    "30m": 12,   # 6 hours of 30m candles
    "1h": 10,    # 10 hours (spans multiple days)
    "4h": 10,    # 40 hours (spans ~5 trading days)
}

# Timeframes to analyze (order = priority for display)
ANALYSIS_TFS = ["5m", "15m", "1h", "4h"]


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------

def _cluster_lows(
    lows: list[float],
    tolerance_pct: float = CLUSTER_TOLERANCE_PCT,
) -> list[tuple[float, int]]:
    """Cluster nearby price levels and count touches.

    Returns list of (center_price, touch_count) sorted by touch count desc.

    Algorithm:
    - Sort lows ascending
    - Walk through, grouping any low within tolerance_pct of the cluster center
    - When a low is too far from the current cluster, start a new one
    """
    if not lows:
        return []

    sorted_lows = sorted(lows)
    clusters: list[tuple[float, int]] = []

    cluster_prices: list[float] = [sorted_lows[0]]

    for i in range(1, len(sorted_lows)):
        center = sum(cluster_prices) / len(cluster_prices)
        pct_diff = abs(sorted_lows[i] - center) / center * 100 if center > 0 else 999

        if pct_diff <= tolerance_pct:
            # Same cluster
            cluster_prices.append(sorted_lows[i])
        else:
            # Save old cluster, start new one
            if len(cluster_prices) >= 1:
                clusters.append((
                    sum(cluster_prices) / len(cluster_prices),
                    len(cluster_prices),
                ))
            cluster_prices = [sorted_lows[i]]

    # Don't forget the last cluster
    if cluster_prices:
        clusters.append((
            sum(cluster_prices) / len(cluster_prices),
            len(cluster_prices),
        ))

    # Sort by touch count descending
    clusters.sort(key=lambda x: -x[1])
    return clusters


def find_support_levels(
    candle_data: dict,
    current_price: float,
    max_distance_pct: float = 2.0,
) -> list[SupportLevel]:
    """Find support levels from multi-timeframe candle data.

    Args:
        candle_data: Dict from CandleCache.get_candle_data() with keys
                     "5m", "15m", "1h", "4h" containing lists of CandleBar.
        current_price: Current underlying price.
        max_distance_pct: Only return levels within this % of current price.

    Returns:
        List of SupportLevel sorted by strength (strongest first).
        Only includes levels BELOW the current price (support, not resistance).
    """
    if current_price <= 0:
        return []

    # Step 1: Find clusters per timeframe
    tf_clusters: dict[str, list[tuple[float, int]]] = {}

    for tf in ANALYSIS_TFS:
        bars = candle_data.get(tf, [])
        if not bars:
            continue

        lookback = min(LOOKBACK.get(tf, 20), len(bars))
        recent = bars[-lookback:]

        # Collect wick lows (only below current price = potential support)
        lows = [b.low for b in recent if b.low < current_price and b.low > 0]
        if not lows:
            continue

        clusters = _cluster_lows(lows)
        # Only keep clusters with minimum touches
        min_touches = MIN_TOUCHES_PER_TF if tf != "4h" else 1  # 4h has fewer bars
        clusters = [(p, c) for p, c in clusters if c >= min_touches]
        if clusters:
            tf_clusters[tf] = clusters

    if not tf_clusters:
        return []

    # Step 2: Merge clusters across timeframes into unified support zones
    # Collect all cluster centers with their TF and touch count
    all_clusters: list[tuple[float, str, int]] = []
    for tf, clusters in tf_clusters.items():
        for price, count in clusters:
            all_clusters.append((price, tf, count))

    # Sort by price
    all_clusters.sort(key=lambda x: x[0])

    # Merge clusters across TFs that are within tolerance of each other
    merged: list[SupportLevel] = []
    used = set()

    for i, (price_i, tf_i, count_i) in enumerate(all_clusters):
        if i in used:
            continue

        # Start a new merged zone
        zone_prices = [price_i]
        zone_counts = [count_i]
        zone_tfs = {tf_i: count_i}
        used.add(i)

        # Find other clusters within tolerance
        for j, (price_j, tf_j, count_j) in enumerate(all_clusters):
            if j in used:
                continue
            center = sum(zone_prices) / len(zone_prices)
            pct_diff = abs(price_j - center) / center * 100 if center > 0 else 999
            if pct_diff <= CLUSTER_TOLERANCE_PCT * 2:  # wider band for cross-TF merge
                zone_prices.append(price_j)
                zone_counts.append(count_j)
                if tf_j in zone_tfs:
                    zone_tfs[tf_j] += count_j
                else:
                    zone_tfs[tf_j] = count_j
                used.add(j)

        center = sum(zone_prices) / len(zone_prices)
        total_strength = sum(zone_counts)
        dist_pct = (current_price - center) / current_price * 100

        # Only include levels below current price and within max distance
        if dist_pct < 0 or dist_pct > max_distance_pct:
            continue

        merged.append(SupportLevel(
            price=round(center, 2),
            strength=total_strength,
            timeframes=sorted(zone_tfs.keys()),
            touches_per_tf=zone_tfs,
            distance_pct=round(dist_pct, 3),
            confluence=len(zone_tfs),
        ))

    # Sort by strength descending (strongest support first)
    merged.sort(key=lambda lv: (-lv.confluence, -lv.strength))
    return merged


def is_at_support(
    candle_data: dict,
    current_price: float,
    max_distance_pct: float = 0.3,
    min_strength: int = 3,
    min_confluence: int = 1,
) -> tuple[bool, str]:
    """Check if current price is near a multi-timeframe support level.

    Args:
        candle_data: From CandleCache.get_candle_data()
        current_price: Current underlying price
        max_distance_pct: Max distance from support to count as "at support"
        min_strength: Min total touches across timeframes
        min_confluence: Min number of timeframes agreeing

    Returns:
        (is_at_support, detail_string)
    """
    levels = find_support_levels(candle_data, current_price)

    if not levels:
        return False, "no support levels found"

    # Find the nearest qualifying support
    for lv in levels:
        if lv.distance_pct <= max_distance_pct and lv.strength >= min_strength:
            if lv.confluence >= min_confluence:
                tf_detail = ", ".join(
                    f"{tf}:{lv.touches_per_tf[tf]}t"
                    for tf in lv.timeframes
                )
                return True, (
                    f"support=${lv.price:.2f} "
                    f"strength={lv.strength} "
                    f"confluence={lv.confluence}TF "
                    f"dist={lv.distance_pct:.2f}% "
                    f"[{tf_detail}]"
                )

    # Report nearest level even if it doesn't qualify
    nearest = min(levels, key=lambda lv: lv.distance_pct)
    tf_detail = ", ".join(
        f"{tf}:{nearest.touches_per_tf[tf]}t"
        for tf in nearest.timeframes
    )
    return False, (
        f"nearest_support=${nearest.price:.2f} "
        f"strength={nearest.strength} "
        f"confluence={nearest.confluence}TF "
        f"dist={nearest.distance_pct:.2f}% "
        f"[{tf_detail}] "
        f"(need: dist<={max_distance_pct}% strength>={min_strength} confluence>={min_confluence})"
    )
