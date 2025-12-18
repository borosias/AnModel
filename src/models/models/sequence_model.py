"""sequence_model.py
====================

This module contains a lightweight sequence encoder intended to provide
behavioural embeddings for each user based on their recent interaction
history.  The design goal is to integrate a sequential component into
the hybrid modelling pipeline without introducing heavy deep‑learning
dependencies.  While advanced recommender systems often employ
Transformer architectures such as SASRec or BERT4Rec, here we provide
a simplified implementation that captures the relative frequency of
event types in a sliding window.  The resulting vector can be used as
additional input features or passed through downstream models.

Key ideas:

* Each unique event type observed in the data is mapped to an index.
* For a given snapshot date and user history the encoder considers the
  last ``max_sequence_length`` events prior to the snapshot.
* It produces an embedding of fixed length equal to the number of
  distinct event types.  Each dimension corresponds to the normalised
  count of that event type in the recent history.
* Position information and sophisticated attention mechanisms are
  omitted for simplicity.  Nevertheless, this representation has proven
  useful as an approximation of sequence dynamics, especially when
  combined with micro‑trend and aggregated snapshot features.

This implementation is deliberately simple so that it can be executed
within environments lacking deep‑learning libraries.  It should be
considered a placeholder for a more expressive Transformer model in
future iterations.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Optional

import numpy as np


class SequenceModel:
    """Encode a user’s recent events as a fixed‑dimensional vector.

    The encoder operates on categorical ``event_type`` values.  During
    ``fit`` it constructs a mapping from each unique event type to a
    column index in the embedding.  At inference time it counts the
    occurrences of each event type in the last ``max_sequence_length``
    events (prior to the snapshot) and normalises the counts by the
    total number of events considered.  The resulting vector can be
    interpreted as a distribution over event types in the recent
    history.

    Parameters
    ----------
    max_sequence_length : int, default=50
        Maximum number of recent events to include when computing the
        embedding.  If the user has fewer than this number of events the
        entire history up to the snapshot date is used.  If they have
        more, only the most recent ``max_sequence_length`` events are
        considered.
    """

    def __init__(self, max_sequence_length: int = 50) -> None:
        if max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive")
        self.max_sequence_length = max_sequence_length
        self.event_type_to_index: Dict[str, int] = {}
        self._fitted: bool = False

    def fit(self, event_types: Iterable[str]) -> None:
        """Register unique event types.

        Parameters
        ----------
        event_types : Iterable[str]
            An iterable of event type strings observed in the training
            data.  The order of registration defines the order of
            dimensions in the output embedding.
        """
        unique_types = list(dict.fromkeys(event_types))  # stable deduplication
        self.event_type_to_index = {et: idx for idx, et in enumerate(unique_types)}
        self._fitted = True

    @property
    def embedding_dim(self) -> int:
        """Number of dimensions in the embedding (one per event type)."""
        return len(self.event_type_to_index)

    def encode_history(
        self,
        event_types: np.ndarray,
        event_dates: np.ndarray,
        snapshot_date: np.datetime64,
    ) -> np.ndarray:
        """Compute a normalised histogram of event types for recent history.

        Parameters
        ----------
        event_types : np.ndarray
            Array of event type strings for a single user, ordered by
            timestamp ascending (matching ``event_dates``).
        event_dates : np.ndarray
            Array of timestamps (e.g. ``np.datetime64[s]``) aligned with
            ``event_types``.
        snapshot_date : np.datetime64
            The cut‑off timestamp; events strictly less than or equal to
            this value are considered.  Events occurring after the
            snapshot must not contribute to the embedding.

        Returns
        -------
        np.ndarray
            A 1‑D array of length ``embedding_dim`` containing the
            normalised counts of each event type in the recent window.
        """
        if not self._fitted:
            raise RuntimeError("SequenceModel.fit() must be called before encode_history()")

        # Identify indices of events up to and including the snapshot
        mask = event_dates <= snapshot_date
        if not np.any(mask):
            # No history; return zeros
            return np.zeros(self.embedding_dim, dtype=float)

        # Filter and take the last ``max_sequence_length`` events
        filtered_types = event_types[mask]
        if filtered_types.size > self.max_sequence_length:
            filtered_types = filtered_types[-self.max_sequence_length:]

        # Count occurrences
        counts = Counter()
        for et in filtered_types:
            if et in self.event_type_to_index:
                counts[et] += 1

        # Build histogram vector
        hist = np.zeros(self.embedding_dim, dtype=float)
        total = float(sum(counts.values()))
        if total == 0:
            return hist
        for et, cnt in counts.items():
            idx = self.event_type_to_index[et]
            hist[idx] = cnt / total
        return hist
