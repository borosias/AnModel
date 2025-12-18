"""micro_trend.py
===================

This module contains a simple implementation of the *micro‑trend* feature
generator.  Micro‑trend features are designed to capture short‑term
fluctuations in user behaviour over a sliding time horizon.  While
traditional snapshot features summarise a user’s entire history up to a
given date, micro‑trends focus on the most recent window and compare it
against a longer baseline window.  This allows the model to detect
accelerations or decelerations in activity that may precede a purchase or
conversion.

The implementation below is intentionally lightweight.  It does not rely
on external libraries beyond NumPy and Pandas and can operate on raw
arrays.  The choice of window lengths (short vs long) and the exact
feature definitions can be adjusted as needed.  When integrating into
the snapshot builder the resulting dictionary of features is merged
directly into the snapshot record.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, Optional

import numpy as np


class MicroTrend:
    """Compute micro‑trend features for a sequence of user events.

    A micro‑trend is defined as a comparison between activity observed in a
    short recent window versus activity observed in a longer baseline
    window.  For example, if a user has generated many events in the last
    few days compared with the previous week, this may indicate an
    upcoming purchase.  Conversely, a drop in short‑term activity may
    signal disengagement.

    Parameters
    ----------
    short_window : int, default=3
        Length of the recent window in days.  Events strictly greater than
        ``snapshot_date - short_window`` and less than or equal to
        ``snapshot_date`` are counted in the numerator.
    long_window : int, default=7
        Length of the baseline window in days.  Events strictly greater
        than ``snapshot_date - long_window`` and less than or equal to
        ``snapshot_date`` are counted in the denominator.

    Notes
    -----
    * The windows are inclusive on the right edge and exclusive on the
      left edge.  This matches the conventions used in the snapshot
      builder where rolling counts include the snapshot day itself.
    * When there is no activity in the baseline window the corresponding
      ratio defaults to 0.0 to avoid division by zero.  Likewise, if
      there is no activity in the short window, ratios will evaluate to
      zero.
    * Additional micro‑trend features (e.g. category‑specific growth) can
      be added by extending the returned dictionary.
    """

    def __init__(self, short_window: int = 3, long_window: int = 7) -> None:
        if short_window <= 0 or long_window <= 0:
            raise ValueError("Window sizes must be positive")
        if short_window > long_window:
            raise ValueError("short_window must be ≤ long_window")
        self.short_window = short_window
        self.long_window = long_window

    def compute(
        self,
        event_dates: np.ndarray,
        event_types: np.ndarray,
        event_prices: Optional[np.ndarray],
        snapshot_date: date,
    ) -> Dict[str, float]:
        """Compute micro‑trend metrics for the given user history.

        Parameters
        ----------
        event_dates : np.ndarray
            Array of event dates (dtype ``object`` or ``datetime64``) for a
            single user.  These should correspond to the ``date`` column
            from the events table and *not* include future events after
            ``snapshot_date``.
        event_types : np.ndarray
            Array of event type strings (e.g. ``"click"``, ``"purchase"``)
            aligned with ``event_dates``.
        event_prices : Optional[np.ndarray]
            Array of purchase amounts aligned with ``event_dates``.
            Non‑purchase events should have ``0`` or ``NaN`` at the same
            index.  This parameter is optional and may be ``None`` if
            price information is unavailable.
        snapshot_date : date
            The snapshot date for which the micro‑trend features are
            calculated.  Events occurring after this date must not be
            included in the windows.

        Returns
        -------
        Dict[str, float]
            A dictionary of micro‑trend feature names to values.  At a
            minimum the following keys are returned:

            * ``micro_event_growth`` – ratio of the number of events in
              the short window to the number of events in the long window.
            * ``micro_purchase_growth`` – ratio of purchases in the short
              window to purchases in the long window.
            * ``micro_purchase_ratio`` – ratio of purchases to events in
              the short window.
            * ``micro_spent_growth`` – (optional) ratio of total spend in
              the short window to that in the long window.  Returned only
              if ``event_prices`` is not ``None``.
        """
        # Convert to numpy datetime64 for efficient comparisons at day precision
        dates_np = np.array(event_dates, dtype="datetime64[D]")
        snapshot_np = np.datetime64(snapshot_date)

        # Define window boundaries
        short_start = snapshot_np - np.timedelta64(self.short_window, 'D')
        long_start = snapshot_np - np.timedelta64(self.long_window, 'D')

        # Build masks: strictly greater than start and ≤ snapshot_date
        mask_short = (dates_np > short_start) & (dates_np <= snapshot_np)
        mask_long = (dates_np > long_start) & (dates_np <= snapshot_np)

        # Count events
        short_events = float(mask_short.sum())
        long_events = float(mask_long.sum())

        # Count purchases
        purchase_mask_short = (event_types[mask_short] == 'purchase')
        purchase_mask_long = (event_types[mask_long] == 'purchase')

        short_purchases = float(purchase_mask_short.sum())
        long_purchases = float(purchase_mask_long.sum())

        result: Dict[str, float] = {}

        # Growth ratios; if denominator is zero, result is 0.0
        result['micro_event_growth'] = (
            short_events / long_events if long_events > 0 else 0.0
        )
        result['micro_purchase_growth'] = (
            short_purchases / long_purchases if long_purchases > 0 else 0.0
        )
        result['micro_purchase_ratio'] = (
            short_purchases / short_events if short_events > 0 else 0.0
        )

        # Spend ratios if price information is available
        if event_prices is not None:
            # Convert to float; NaNs are treated as zeros by nansum
            prices_np = np.array(event_prices, dtype=float)
            short_spent = float(np.nansum(prices_np[mask_short]))
            long_spent = float(np.nansum(prices_np[mask_long]))
            result['micro_spent_growth'] = (
                short_spent / long_spent if long_spent > 0 else 0.0
            )

        return result
