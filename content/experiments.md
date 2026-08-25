---
title: Experiments
---

# Experiments

A collection of experiments, code dump etc, that are more for the vibe than for
the writing (i.e. these may be heavily written by AI, but are still "cool").

# 2026

## Speech Diarizer

A fairly vanilla speech diarizer without torch or unusual dependencies. Only
`onnxruntime`. See https://github.com/NoRaincheck/Speech-Segmentation

```py
import urllib.request
import wave
import numpy as np
import onnxruntime as ort

model_id = "onnx-community/pyannote-segmentation-3.0"

model_path = "model.onnx"
urllib.request.urlretrieve(
    f"https://huggingface.co/{model_id}/resolve/main/onnx/model.onnx",
    model_path,
)

url = "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/mlk.wav"
audio_path = "mlk.wav"
urllib.request.urlretrieve(url, audio_path)

with wave.open(audio_path, "rb") as wf:
    sr = wf.getframerate()
    frames = wf.readframes(wf.getnframes())
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

target_sr = 16000
step = 270

if sr != target_sr:
    old_len = len(audio)
    new_len = int(old_len * target_sr / sr)
    audio = np.interp(np.linspace(0, old_len - 1, new_len), np.arange(old_len), audio)

session = ort.InferenceSession(model_path)
logits = session.run(
    None, {"input_values": audio[np.newaxis, np.newaxis, :].astype(np.float32)}
)[0]

frame_logits = logits[0]
exps = np.exp(frame_logits - frame_logits.max(axis=1, keepdims=True))
probs = exps / exps.sum(axis=1, keepdims=True)

preds = probs.argmax(axis=1)
confidence = probs.max(axis=1)

segments = []
current_spk = None
current_start = None
max_conf = 0.0

for i, (cls, conf) in enumerate(zip(preds, confidence)):
    if cls in (1, 2, 3):
        if cls != current_spk:
            if current_spk is not None:
                segments.append(
                    (
                        current_spk,
                        current_start * step / target_sr,
                        i * step / target_sr,
                        float(max_conf),
                    )
                )
            current_spk = cls
            current_start = i
            max_conf = conf
        else:
            max_conf = max(max_conf, conf)
    else:
        if current_spk is not None:
            segments.append(
                (
                    current_spk,
                    current_start * step / target_sr,
                    i * step / target_sr,
                    float(max_conf),
                )
            )
            current_spk = None
            current_start = None
            max_conf = 0.0

if current_spk is not None:
    segments.append(
        (
            current_spk,
            current_start * step / target_sr,
            len(preds) * step / target_sr,
            float(max_conf),
        )
    )

for spk_id, start, end, conf in segments:
    print(f"  SPEAKER_{spk_id:02d}  {start:7.2f}s - {end:7.2f}s  (conf={conf:.3f})")
```

## pre-commit hook for anonymous git commits

```sh
#!/bin/sh
# Pre-commit hook to check that git config contains $NAME
NAME='~myname'

# Define color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if user.name
USER_NAME=$(git config user.name)
if ! echo "$USER_NAME" | grep -q "${NAME}"; then
    echo "${RED}❌ Error: git config user.name does not contain '${NAME}'${NC}"
    echo "${YELLOW}Current user.name: $USER_NAME${NC}"
    exit 1
fi

echo "${GREEN}✅ Git config validation passed: user.name and user.email contain '${NAME}'${NC}"
```

## Full Text Search (sqlite3 FTS5)

Semantic search gets all the attention, but full-text search with BM25 scoring
is fast, dependency-free, and surprisingly effective for local RAG pipelines.
Python's built-in `sqlite3` ships with **FTS5** support — no pip installs, no
external databases. Just create a regular table for your documents (plus JSON
metadata), an FTS5 virtual table for indexing, and a few triggers to keep them
in sync.

```py
"""
Minimal full-text search client using sqlite3 FTS5 with BM25 scoring.

Stdlib only — no pip installs required. Python 3.12+ (union types).
Uses JSON functions (SQLite 3.38+) for metadata filtering.
"""

import json
import sqlite3
from typing import Any, Sequence


class FTSClient:
    """
    A full-text search client backed by sqlite3 FTS5.

    Documents are stored in a regular table (id + content + JSON metadata).
    An FTS5 virtual table indexes only the ``content`` column. Triggers keep
    the index in sync automatically.

    BM25 relevance scores are returned via ``bm25(fts_table)`` — lower is better.
    """

    def __init__(self, db: str = ":memory:"):
        self.con = sqlite3.connect(db)
        self._ensure_schema()

    # ------------------------------------------------------------------ schema
    def _ensure_schema(self) -> None:
        self.con.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                id       TEXT PRIMARY KEY,
                content  TEXT,
                metadata TEXT   -- JSON string stored in the regular table
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
                content,
                tokenize='unicode61'
            );

            /* Triggers keep the FTS index synced with the main table. */
            CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN
                INSERT INTO documents_fts(_rowid_, content) VALUES (new.rowid, new.content);
            END;

            CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN
                DELETE FROM documents_fts WHERE rowid = old.rowid;
            END;

            CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE ON documents BEGIN
                DELETE FROM documents_fts WHERE rowid = old.rowid;
                INSERT INTO documents_fts(_rowid_, content) VALUES (new.rowid, new.content);
            END;
        """)

    # -------------------------------------------------------------- CRUD helpers
    def insert(
        self,
        data: Sequence[tuple[str, str, dict[str, Any]]],
        replace: bool = False,
    ) -> None:
        """Insert documents.

        Args:
            data:  iterable of ``(id, content, metadata_dict)``.
            replace: if True, existing rows with the same ``id`` are overwritten.

        Note:
            We explicitly DELETE before INSERT when ``replace=True`` because
            SQLite's ``INSERT OR REPLACE`` does not fire AFTER DELETE triggers,
            which would leave stale FTS entries behind.
        """
        serialized = [
            (doc_id, content, json.dumps(metadata))
            for doc_id, content, metadata in data
        ]
        if replace and serialized:
            ids = tuple(d[0] for d in serialized)
            placeholders = ",".join("?" for _ in ids)
            self.con.execute(
                f"DELETE FROM documents WHERE id IN ({placeholders})", ids,
            )
        sql = "INSERT INTO documents VALUES (?, ?, ?)"
        self.con.executemany(sql, serialized)

    def delete(self, ids: Sequence[str]) -> None:
        placeholders = ",".join("?" for _ in ids)
        self.con.execute(
            f"DELETE FROM documents WHERE id IN ({placeholders})", ids,
        )

    # -------------------------------------------------------------- search API
    def search(
        self,
        query: str = "",
        limit: int = 10,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[tuple[str, str, str, float]]:
        """
        Search documents by full-text query (BM25) with optional metadata filter.

        Returns a list of ``(id, content, metadata_json, bm25_score)`` tuples.
        Scores are lower for better matches.
        """
        if not query and not metadata_filter:
            return []

        meta_clauses: list[str] = []
        params: list[Any] = []
        if metadata_filter:
            for key, val in metadata_filter.items():
                path = f"$.{key}"
                # Normalize booleans to lowercase (SQLite JSON stores true/false)
                norm = str(val).lower() if isinstance(val, bool) else val
                meta_clauses.append("json_extract(d.metadata, ?) = ?")
                params.extend([path, norm])

        has_fts = bool(query)
        has_meta = bool(metadata_filter)

        if has_fts and has_meta:
            sql = f"""
                SELECT d.id, d.content, d.metadata,
                       bm25(documents_fts) AS score
                  FROM documents d
                 JOIN documents_fts ON documents_fts._rowid_ = d.rowid
                 WHERE documents_fts MATCH ?
                   AND {' AND '.join(meta_clauses)}
                 ORDER BY score ASC LIMIT ?"""
            params = [query, *params, limit]

        elif has_meta and not has_fts:
            sql = f"""
                SELECT d.id, d.content, d.metadata, NULL AS score
                  FROM documents d
                 WHERE {' AND '.join(meta_clauses)}
                 ORDER BY d.id ASC LIMIT ?"""
            params = [*params, limit]

        elif has_fts and not has_meta:
            sql = """
                SELECT d.id, d.content, d.metadata,
                       bm25(documents_fts) AS score
                  FROM documents d
                 JOIN documents_fts ON documents_fts._rowid_ = d.rowid
                 WHERE documents_fts MATCH ?
                 ORDER BY score ASC LIMIT ?"""
            params = [query, limit]

        else:
            return []

        rows = self.con.execute(sql, params).fetchall()
        return [(rid, content, meta, score) for rid, content, meta, score in rows]
```

# 2025

## Lightweight SHAP Implementation

_December 2025_

When performing SHAP, sometimes the requirements to install are overly onerous.
This is a lightweight implementation of Permutation SHAP including helper
functions matching the style of `scikit-learn` interface.

```py
"""
Permutation SHAP with scikit-learn style API.

This module provides permutation-based SHAP feature importance computation
following scikit-learn's API conventions, with feature selection utilities
including Boruta.

Example usage:

    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    >>> model = RandomForestClassifier(random_state=42).fit(X, y)

    # Global importance (sklearn permutation_importance style)
    >>> result = permutation_shap(model, X, n_repeats=5, random_state=42)
    >>> print(result.importances_mean)

    # Per-instance SHAP values
    >>> shap_values = permutation_shap_values(model, X[:5], background=X, random_state=42)
    >>> print(shap_values.shape)  # (5, 10) or (5, 10, n_classes)

    # Feature selection
    >>> selector = SelectFromShap(model, threshold='median', random_state=42)
    >>> X_selected = selector.fit_transform(X)
"""

import numpy as np
from sklearn.base import BaseEstimator, clone, is_classifier
from sklearn.feature_selection import SelectorMixin
from sklearn.utils import Bunch, check_random_state
from sklearn.cluster import KMeans, MiniBatchKMeans


def _get_predict_fn(estimator):
    """
    Get a prediction function from an estimator.

    For classifiers, uses predict_proba if available, otherwise predict.
    For regressors, uses predict.

    Parameters
    ----------
    estimator : estimator object
        A fitted estimator.

    Returns
    -------
    predict_fn : callable
        Function that takes X and returns predictions.
    """
    if hasattr(estimator, 'predict_proba'):
        return estimator.predict_proba
    return estimator.predict


def _summarize_data(X, max_samples, random_state):
    """
    Summarize data using k-means clustering if max_samples is specified.

    Parameters
    ----------
    X : np.ndarray
        Input data to summarize. Shape: (n_samples, n_features)
    max_samples : int, float, or None
        If None, return X unchanged.
        If int, reduce to that many centroids via k-means.
        If float (0-1), reduce to that fraction of samples.
    random_state : RandomState
        Random state for k-means.

    Returns
    -------
    np.ndarray
        Either the original X or k-means cluster centroids.
    """
    if max_samples is None:
        return X

    # Determine number of clusters
    if isinstance(max_samples, float):
        if not 0 < max_samples <= 1:
            raise ValueError("max_samples as float must be in (0, 1]")
        n_clusters = max(1, int(max_samples * X.shape[0]))
    else:
        n_clusters = int(max_samples)

    # No reduction needed if n_clusters >= n_samples
    if n_clusters >= X.shape[0]:
        return X

    if n_clusters < 1:
        raise ValueError("max_samples must result in at least 1 cluster")

    # Use MiniBatchKMeans for large datasets (faster), KMeans otherwise (more accurate)
    threshold = 10000
    KMeansClass = MiniBatchKMeans if X.shape[0] > threshold else KMeans
    seed = random_state.randint(0, 2**31) if hasattr(random_state, 'randint') else random_state
    kmeans = KMeansClass(n_clusters=n_clusters, random_state=seed, n_init='auto')
    kmeans.fit(X)
    return kmeans.cluster_centers_


def _compute_shap_single_instance(x, predict_fn, background, n_permutations, rng):
    """
    Compute SHAP values for a single instance using permutation method
    with antithetic sampling.

    Parameters
    ----------
    x : np.ndarray
        Instance to explain. Shape: (n_features,)
    predict_fn : callable
        Prediction function.
    background : np.ndarray
        Background data. Shape: (n_background, n_features)
    n_permutations : int
        Number of permutations to sample.
    rng : RandomState
        Random number generator.

    Returns
    -------
    shap_values : np.ndarray
        SHAP values for each feature.
    """
    n_features = x.shape[0]
    n_background = background.shape[0]

    # Determine output dimensionality
    sample_pred = predict_fn(background[:1])
    if sample_pred.ndim == 1:
        n_outputs = 1
        contributions = [[] for _ in range(n_features)]
    else:
        n_outputs = sample_pred.shape[1]
        contributions = [[] for _ in range(n_features)]

    for _ in range(n_permutations):
        # Sample a random permutation of feature indices
        perm = rng.permutation(n_features)

        # Sample a background instance for absent features
        bg_idx = rng.randint(n_background)
        bg_sample = background[bg_idx].copy()

        # Forward pass: add features according to permutation
        _compute_marginal_contributions(x, perm, bg_sample, predict_fn, contributions)

        # Backward pass (antithetic): use reverse permutation to reduce variance
        reverse_perm = perm[::-1]
        _compute_marginal_contributions(x, reverse_perm, bg_sample, predict_fn, contributions)

    # Average all marginal contributions for each feature
    if n_outputs == 1:
        shap_values = np.array([np.mean(contribs) for contribs in contributions])
    else:
        shap_values = np.array([
            np.mean(contribs, axis=0) for contribs in contributions
        ])

    return shap_values


def _compute_marginal_contributions(x, permutation, background_sample, predict_fn,
                                    contributions):
    """
    Compute marginal contributions for a single permutation pass.

    Starting from a coalition containing no features from x (all from background),
    we add features one at a time according to the permutation order.
    Each addition gives us a marginal contribution for that feature.

    Parameters
    ----------
    x : np.ndarray
        Instance to explain. Shape: (n_features,)
    permutation : np.ndarray
        Order in which to add features. Shape: (n_features,)
    background_sample : np.ndarray
        Background sample for absent features. Shape: (n_features,)
    predict_fn : callable
        Prediction function.
    contributions : list of lists
        Contribution accumulator. contributions[i] is a list of marginal
        contributions for feature i.
    """
    # Start with all features from background (empty coalition w.r.t. x)
    current_sample = background_sample.copy()
    prev_pred = predict_fn(current_sample.reshape(1, -1))[0]

    for feat_idx in permutation:
        # Add this feature to the coalition (use value from x)
        current_sample[feat_idx] = x[feat_idx]
        current_pred = predict_fn(current_sample.reshape(1, -1))[0]

        # Marginal contribution: v(S ∪ {i}) - v(S)
        marginal = current_pred - prev_pred
        contributions[feat_idx].append(marginal)

        # Reuse this prediction as the "before" for the next feature
        prev_pred = current_pred


def permutation_shap_values(estimator, X, *, background=None, n_permutations=10,
                            max_samples=None, random_state=None):
    """
    Compute per-instance SHAP values using permutation method.

    This function computes SHAP values for each instance in X using the
    permutation method with antithetic sampling for variance reduction.

    Parameters
    ----------
    estimator : estimator object
        A fitted estimator. Must have a predict or predict_proba method.
    X : array-like of shape (n_samples, n_features)
        Data for which to compute SHAP values.
    background : array-like of shape (n_background, n_features), optional
        Background data used for computing baseline expectations.
        If None, uses X as the background.
    n_permutations : int, default=10
        Number of permutations to sample. Each permutation generates both
        forward and backward (antithetic) passes.
    max_samples : int, float, or None, default=None
        The number of samples to use from background. If background is large,
        k-means clustering is used to summarize it to centroids.
        - If None, use the full background unchanged.
        - If int, reduce background to that many centroids via k-means.
        - If float (0-1), reduce background to that fraction of samples.
    random_state : int, RandomState instance, or None, default=None
        Controls the randomness. Pass an int for reproducible results.

    Returns
    -------
    shap_values : np.ndarray
        SHAP values for each instance and feature.
        - If estimator outputs 1D predictions: shape (n_samples, n_features)
        - If estimator outputs 2D predictions: shape (n_samples, n_features, n_outputs)

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    >>> model = RandomForestClassifier(random_state=42).fit(X, y)
    >>> shap_values = permutation_shap_values(model, X[:5], background=X, random_state=42)
    >>> shap_values.shape
    (5, 10, 2)
    """
    X = np.asarray(X)
    rng = check_random_state(random_state)

    if background is None:
        background = X
    else:
        background = np.asarray(background)

    # Summarize background if requested
    background = _summarize_data(background, max_samples, rng)

    predict_fn = _get_predict_fn(estimator)

    # Compute SHAP values for each instance
    results = []
    for i in range(X.shape[0]):
        shap_vals = _compute_shap_single_instance(
            X[i], predict_fn, background, n_permutations, rng
        )
        results.append(shap_vals)

    return np.array(results)


def permutation_shap(estimator, X, y=None, *, background=None, n_repeats=5,
                     n_permutations=10, max_samples=None, random_state=None):
    """
    Compute permutation-based SHAP feature importance.

    This function follows the scikit-learn permutation_importance API pattern.
    It computes SHAP values and aggregates them to global feature importance
    using mean absolute SHAP values.

    Parameters
    ----------
    estimator : estimator object
        A fitted estimator. Must have a predict or predict_proba method.
    X : array-like of shape (n_samples, n_features)
        Data on which to compute feature importance.
    y : array-like of shape (n_samples,), optional
        Target values. Not used in computation but accepted for API compatibility.
    background : array-like of shape (n_background, n_features), optional
        Background data used for computing baseline expectations.
        If None, uses X as the background.
    n_repeats : int, default=5
        Number of times to repeat the importance computation with different
        random seeds for variance estimation.
    n_permutations : int, default=10
        Number of permutations to sample per repeat for SHAP value estimation.
    max_samples : int, float, or None, default=None
        The number of samples to use from background. If background is large,
        k-means clustering is used to summarize it to centroids.
        - If None, use the full background unchanged.
        - If int, reduce background to that many centroids via k-means.
        - If float (0-1), reduce background to that fraction of samples.
    random_state : int, RandomState instance, or None, default=None
        Controls the randomness. Pass an int for reproducible results.

    Returns
    -------
    result : Bunch
        Dictionary-like object with the following attributes:

        importances_mean : np.ndarray of shape (n_features,)
            Mean of feature importances across repeats.
        importances_std : np.ndarray of shape (n_features,)
            Standard deviation of feature importances across repeats.
        importances : np.ndarray of shape (n_features, n_repeats)
            Raw feature importances for each repeat.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    >>> model = RandomForestClassifier(random_state=42).fit(X, y)
    >>> result = permutation_shap(model, X, n_repeats=5, random_state=42)
    >>> result.importances_mean.shape
    (10,)
    >>> for i in result.importances_mean.argsort()[::-1][:3]:
    ...     print(f"Feature {i}: {result.importances_mean[i]:.4f} +/- {result.importances_std[i]:.4f}")
    """
    X = np.asarray(X)
    rng = check_random_state(random_state)
    n_features = X.shape[1]

    if background is None:
        background = X
    else:
        background = np.asarray(background)

    # Summarize background if requested (do this once, not per repeat)
    background = _summarize_data(background, max_samples, rng)

    # Collect importances across repeats
    importances = np.zeros((n_features, n_repeats))

    for repeat in range(n_repeats):
        # Compute SHAP values for all instances
        shap_values = permutation_shap_values(
            estimator, X,
            background=background,
            n_permutations=n_permutations,
            max_samples=None,  # Already summarized
            random_state=rng
        )

        # For multi-output, sum absolute values across outputs
        if shap_values.ndim == 3:
            # Shape: (n_samples, n_features, n_outputs)
            # Take mean absolute across samples, sum across outputs
            feature_importance = np.mean(np.sum(np.abs(shap_values), axis=2), axis=0)
        else:
            # Shape: (n_samples, n_features)
            # Take mean absolute across samples
            feature_importance = np.mean(np.abs(shap_values), axis=0)

        importances[:, repeat] = feature_importance

    return Bunch(
        importances_mean=np.mean(importances, axis=1),
        importances_std=np.std(importances, axis=1),
        importances=importances
    )


class SelectFromShap(SelectorMixin, BaseEstimator):
    """
    Feature selector based on permutation SHAP importance.

    This transformer uses permutation-based SHAP values to select features
    based on their importance. It supports threshold-based selection and
    Boruta feature selection.

    Parameters
    ----------
    estimator : estimator object
        A fitted estimator to compute SHAP importance for.
    threshold : float, str, or None, default=None
        The threshold value to use for feature selection. Features with
        importance greater than or equal to the threshold are selected.
        - If float, features with importance >= threshold are selected.
        - If "median", uses the median of feature importances.
        - If "mean", uses the mean of feature importances.
        - If None and mode='threshold', defaults to "median".
        Ignored when mode='boruta'.
    max_features : int or None, default=None
        The maximum number of features to select. If not None, only the
        top max_features features are selected. If both threshold and
        max_features are specified, max_features takes precedence.
    mode : {'threshold', 'boruta'}, default='threshold'
        Feature selection mode:
        - 'threshold': Select features above importance threshold.
        - 'boruta': Use Boruta algorithm with shadow features.
    n_permutations : int, default=10
        Number of permutations for SHAP value estimation.
    n_repeats : int, default=5
        Number of repeats for importance variance estimation.
    max_samples : int, float, or None, default=None
        Maximum samples for background summarization.
    boruta_max_iter : int, default=100
        Maximum iterations for Boruta algorithm. Only used when mode='boruta'.
    boruta_alpha : float, default=0.05
        Significance level for Boruta statistical test. Only used when mode='boruta'.
    random_state : int, RandomState instance, or None, default=None
        Controls randomness.

    Attributes
    ----------
    importances_ : np.ndarray of shape (n_features,)
        Feature importances (mean absolute SHAP values).
    importances_std_ : np.ndarray of shape (n_features,)
        Standard deviation of feature importances.
    support_ : np.ndarray of shape (n_features,)
        Boolean mask of selected features.
    n_features_in_ : int
        Number of features seen during fit.
    ranking_ : np.ndarray of shape (n_features,)
        Feature ranking (1 = selected, higher = less important).
        Only available when mode='boruta'.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=100, n_features=20,
    ...                            n_informative=5, random_state=42)
    >>> model = RandomForestClassifier(random_state=42).fit(X, y)

    # Threshold-based selection
    >>> selector = SelectFromShap(model, threshold='median', random_state=42)
    >>> X_selected = selector.fit_transform(X)
    >>> X_selected.shape[1]  # About half the features

    # Top-k selection
    >>> selector = SelectFromShap(model, max_features=5, random_state=42)
    >>> X_selected = selector.fit_transform(X)
    >>> X_selected.shape[1]
    5

    # Boruta selection
    >>> selector = SelectFromShap(model, mode='boruta', random_state=42)
    >>> X_selected = selector.fit_transform(X)
    """

    def __init__(self, estimator, *, threshold=None, max_features=None,
                 mode='threshold', n_permutations=10, n_repeats=5,
                 max_samples=None, boruta_max_iter=100, boruta_alpha=0.05,
                 random_state=None):
        self.estimator = estimator
        self.threshold = threshold
        self.max_features = max_features
        self.mode = mode
        self.n_permutations = n_permutations
        self.n_repeats = n_repeats
        self.max_samples = max_samples
        self.boruta_max_iter = boruta_max_iter
        self.boruta_alpha = boruta_alpha
        self.random_state = random_state

    def fit(self, X, y=None):
        """
        Fit the feature selector.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,), optional
            Target values. Not used but accepted for pipeline compatibility.

        Returns
        -------
        self : object
            Returns self.
        """
        X = np.asarray(X)
        self.n_features_in_ = X.shape[1]

        if self.mode == 'boruta':
            self._fit_boruta(X, y)
        else:
            self._fit_threshold(X, y)

        return self

    def _fit_threshold(self, X, y):
        """Fit using threshold-based selection."""
        rng = check_random_state(self.random_state)

        # Compute SHAP importance
        result = permutation_shap(
            self.estimator, X, y,
            background=X,
            n_repeats=self.n_repeats,
            n_permutations=self.n_permutations,
            max_samples=self.max_samples,
            random_state=rng
        )

        self.importances_ = result.importances_mean
        self.importances_std_ = result.importances_std

        # Determine support mask
        if self.max_features is not None:
            # Select top max_features
            n_select = min(self.max_features, self.n_features_in_)
            top_indices = np.argsort(self.importances_)[::-1][:n_select]
            self.support_ = np.zeros(self.n_features_in_, dtype=bool)
            self.support_[top_indices] = True
        else:
            # Use threshold
            threshold = self.threshold
            if threshold is None or threshold == 'median':
                threshold = np.median(self.importances_)
            elif threshold == 'mean':
                threshold = np.mean(self.importances_)

            self.support_ = self.importances_ >= threshold

            # Ensure at least one feature is selected
            if not np.any(self.support_):
                best_idx = np.argmax(self.importances_)
                self.support_[best_idx] = True

    def _fit_boruta(self, X, y):
        """
        Fit using Boruta algorithm.

        The Boruta algorithm:
        1. Create shadow features (shuffled copies of all original features)
        2. Train model on real + shadow features
        3. Compute SHAP importance for all features
        4. Compare each real feature's importance to max shadow importance
        5. Use statistical test to classify features as confirmed/rejected/tentative
        6. Repeat until all features are classified or max_iter reached
        """
        rng = check_random_state(self.random_state)
        n_features = X.shape[1]

        # Track feature status: 0=tentative, 1=confirmed, -1=rejected
        status = np.zeros(n_features, dtype=int)

        # Track hits (times feature beat max shadow)
        hits = np.zeros(n_features, dtype=int)

        # Track total trials
        n_trials = 0

        for iteration in range(self.boruta_max_iter):
            # Check if all features are classified
            if np.all(status != 0):
                break

            # Get tentative feature indices
            tentative_mask = (status == 0)

            # Create shadow features by shuffling each column independently
            X_shadow = X.copy()
            for j in range(n_features):
                rng.shuffle(X_shadow[:, j])

            # Combine real and shadow features
            X_combined = np.hstack([X, X_shadow])

            # Clone and refit estimator on combined data
            combined_estimator = clone(self.estimator)
            combined_estimator.fit(X_combined, y)

            # Compute SHAP importance for combined features
            result = permutation_shap(
                combined_estimator, X_combined, y,
                background=X_combined,
                n_repeats=1,  # Single repeat per iteration
                n_permutations=self.n_permutations,
                max_samples=self.max_samples,
                random_state=rng
            )

            importances = result.importances_mean
            real_importances = importances[:n_features]
            shadow_importances = importances[n_features:]

            # Max shadow importance (threshold)
            max_shadow = np.max(shadow_importances)

            # Update hits for tentative features
            for j in range(n_features):
                if status[j] == 0:  # Tentative
                    if real_importances[j] > max_shadow:
                        hits[j] += 1

            n_trials += 1

            # Statistical test using binomial distribution
            # Under null hypothesis, P(beat max shadow) = 0.5
            # Use two-tailed test with Bonferroni correction
            if n_trials >= 5:  # Need minimum trials for stable test
                for j in range(n_features):
                    if status[j] == 0:  # Tentative
                        # Binomial test: probability of getting this many hits by chance
                        # Using normal approximation for binomial
                        p_value = self._binomial_test(hits[j], n_trials)

                        if p_value < self.boruta_alpha / n_features:  # Bonferroni correction
                            if hits[j] > n_trials / 2:
                                status[j] = 1  # Confirmed
                            else:
                                status[j] = -1  # Rejected

        # Final classification: treat remaining tentative as rejected
        self.support_ = (status == 1)

        # If no features confirmed, select the one with most hits
        if not np.any(self.support_):
            best_idx = np.argmax(hits)
            self.support_[best_idx] = True

        # Compute final importances on original features
        result = permutation_shap(
            self.estimator, X, y,
            background=X,
            n_repeats=self.n_repeats,
            n_permutations=self.n_permutations,
            max_samples=self.max_samples,
            random_state=rng
        )

        self.importances_ = result.importances_mean
        self.importances_std_ = result.importances_std

        # Create ranking: 1 for confirmed, 2 for tentative, 3 for rejected
        self.ranking_ = np.where(status == 1, 1, np.where(status == 0, 2, 3))

    def _binomial_test(self, k, n, p=0.5):
        """
        Two-tailed binomial test using normal approximation.

        Tests whether the number of successes k in n trials is
        significantly different from expected under p=0.5.

        Parameters
        ----------
        k : int
            Number of successes (hits).
        n : int
            Number of trials.
        p : float, default=0.5
            Null hypothesis probability.

        Returns
        -------
        p_value : float
            Two-tailed p-value.
        """
        # Normal approximation to binomial
        mean = n * p
        std = np.sqrt(n * p * (1 - p))

        if std == 0:
            return 1.0

        # Z-score
        z = abs(k - mean) / std

        # Two-tailed p-value using normal CDF approximation
        # Using error function approximation
        p_value = 2 * (1 - self._norm_cdf(z))

        return p_value

    def _norm_cdf(self, x):
        """
        Standard normal CDF approximation.

        Uses the error function approximation.
        """
        # Approximation using the error function
        # CDF(x) = 0.5 * (1 + erf(x / sqrt(2)))
        return 0.5 * (1 + self._erf(x / np.sqrt(2)))

    def _erf(self, x):
        """
        Error function approximation.

        Abramowitz and Stegun approximation (max error ~ 1.5e-7).
        """
        # Constants
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        p = 0.3275911

        sign = np.sign(x)
        x = np.abs(x)

        t = 1.0 / (1.0 + p * x)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)

        return sign * y

    def _get_support_mask(self):
        """Get the boolean mask of selected features."""
        return self.support_
```

## Tree Ensemble Hashing

_August 2025_

I've been thinking about how to encode decision trees (on and off). This is a
vibe-coded approach that tries to encode the path of every instance based on
what leaf of the decision tree(s) the instance lands on. The rough idea is that
it also gets 'weight' from every parent node (weighted based on distance to
leaf) and is encoded by feature hashing, which allows it to be aggregated by all
trees. It somewhat works. I think this kind of approach may be useful for
scenarios where online learning with just a linear head is used, especially when
feature engineering (from the trees) is done in an online manner, for example
via Mondrian Trees.

```py
import mmh3
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from onnx import ModelProto
import logging

logger = logging.getLogger(__name__)


class TreeEnsembleHash:
    """
    Implements the tree ensemble hashing algorithm described in the README.

    The algorithm works as follows:
    1. Extract all nodes, splits (thresholds), and paths from an ONNX tree ensemble model
    2. For each input sample, traverse the trees to find which leaves are visited
    3. Hash only the visited leaves based on {node_id} {operation} {threshold}
    4. Compute weights based on inverse distance to leaf nodes using 1/d(x,y)^p, 0<p<1
    5. Aggregate all hashed leaf features to create the final representation
    """

    def __init__(self, onnx_model: Union[ModelProto, Tuple[ModelProto, Any]], hash_dim: int = 128, distance_power: float = 0.5):
        """
        Initialize the TreeEnsembleHash with an ONNX model.

        Args:
            onnx_model: ONNX model containing tree ensemble operators, or tuple (ModelProto, Topology)
            hash_dim: Dimension for murmurhash3 output
            distance_power: Power parameter for distance weighting (0 < p < 1)
        """
        if not 0 < distance_power < 1:
            raise ValueError("distance_power must be between 0 and 1")
        if hash_dim <= 0:
            raise ValueError("hash_dim must be positive")

        self.hash_dim = hash_dim
        self.distance_power = distance_power

        # Extract and store tree ensemble information
        self.tree_ensembles = self._extract_tree_ensemble_info(onnx_model)
        if not self.tree_ensembles:
            raise ValueError("No tree ensemble operators found in the model")

        # Pre-compute distances for all trees
        self.tree_distances = []
        for tree_info in self.tree_ensembles:
            distances = self._compute_node_distances(tree_info)
            self.tree_distances.append(distances)

    def _extract_tree_ensemble_info(self, model: Union[ModelProto, Tuple[ModelProto, Any]]) -> List[Dict]:
        """
        Extract tree ensemble information from ONNX model.

        Args:
            model: ONNX model containing tree ensemble operators, or tuple (ModelProto, Topology)

        Returns:
            List of dictionaries containing tree ensemble information
        """
        # Handle case where convert_sklearn returns (ModelProto, Topology)
        if isinstance(model, tuple):
            model = model[0]

        tree_ensembles = []

        for node in model.graph.node:
            # Check both the op_type and domain for tree ensemble operators
            if (node.op_type in ['TreeEnsembleClassifier', 'TreeEnsembleRegressor'] or
                (hasattr(node, 'domain') and node.domain == 'ai.onnx.ml' and
                 node.op_type in ['TreeEnsembleClassifier', 'TreeEnsembleRegressor'])):
                # Extract attributes
                attrs = {attr.name: attr for attr in node.attribute}

                # Get node information
                node_ids = self._get_attribute_values(attrs, 'nodes_nodeids')
                feature_ids = self._get_attribute_values(attrs, 'nodes_featureids')
                values = self._get_attribute_values(attrs, 'nodes_values')
                modes = self._get_attribute_values(attrs, 'nodes_modes')
                true_node_ids = self._get_attribute_values(attrs, 'nodes_truenodeids')
                false_node_ids = self._get_attribute_values(attrs, 'nodes_falsenodeids')
                tree_ids = self._get_attribute_values(attrs, 'nodes_treeids')

                if all(v is not None for v in [node_ids, feature_ids, values, modes, true_node_ids, false_node_ids, tree_ids]):
                    tree_ensembles.append({
                        'node_ids': node_ids,
                        'feature_ids': feature_ids,
                        'values': values,
                        'modes': modes,
                        'true_node_ids': true_node_ids,
                        'false_node_ids': false_node_ids,
                        'tree_ids': tree_ids
                    })

        return tree_ensembles

    def _get_attribute_values(self, attrs: Dict, name: str) -> Optional[List]:
        """Extract attribute values from ONNX node attributes."""
        if name not in attrs:
            logger.debug(f"Attribute {name} not found in {list(attrs.keys())}")
            return None

        attr = attrs[name]
        logger.debug(f"Attribute {name} type: {attr.type}")

        try:
            if attr.type == 1:  # FLOAT
                result = list(attr.floats)
                return result
            elif attr.type == 2:  # INT
                result = list(attr.ints)
                return result
            elif attr.type == 3:  # STRING
                result = list(attr.strings)
                return result
            elif attr.type == 6:  # FLOATS (alternative representation)
                result = list(attr.floats)
                return result
            elif attr.type == 7:  # INTS (alternative representation)
                result = list(attr.ints)
                return result
            elif attr.type == 8:  # STRINGS (alternative representation)
                result = list(attr.strings)
                return result
            else:
                return None
        except Exception as e:
            return None

    def _compute_node_distances(self, tree_info: Dict) -> Dict[int, Union[int, float]]:
        """
        Compute the distance from each node to its nearest leaf node.

        Args:
            tree_info: Dictionary containing tree structure information

        Returns:
            Dictionary mapping node_id to distance to nearest leaf
        """
        node_ids = tree_info['node_ids']
        modes = tree_info['modes']
        true_node_ids = tree_info['true_node_ids']
        false_node_ids = tree_info['false_node_ids']

        # Create adjacency list
        adjacency = {}
        for i, node_id in enumerate(node_ids):
            if modes[i] != b'LEAF' and modes[i] != 'LEAF':
                adjacency[node_id] = []
                if true_node_ids[i] != 0:
                    adjacency[node_id].append(true_node_ids[i])
                if false_node_ids[i] != 0:
                    adjacency[node_id].append(false_node_ids[i])

        # BFS to compute distances from each node to nearest leaf
        distances = {}
        for node_id in node_ids:
            if node_id not in distances:
                distances[node_id] = self._bfs_distance_to_leaf(node_id, adjacency, modes, node_ids)

        return distances

    def _bfs_distance_to_leaf(self, start_node: int, adjacency: Dict, modes: List, node_ids: List) -> Union[int, float]:
        """Compute distance from start_node to nearest leaf using BFS."""
        if start_node not in adjacency:
            return 0  # This is a leaf node

        visited = set()
        queue = [(start_node, 0)]

        while queue:
            current_node, distance = queue.pop(0)

            if current_node in visited:
                continue

            visited.add(current_node)

            # Find the index of current_node in node_ids
            try:
                idx = node_ids.index(current_node)
                if modes[idx] == b'LEAF' or modes[idx] == 'LEAF':
                    return distance
            except ValueError:
                continue

            # Add neighbors to queue
            if current_node in adjacency:
                for neighbor in adjacency[current_node]:
                    if neighbor != 0:  # 0 indicates no child
                        queue.append((neighbor, distance + 1))

        return float('inf')  # No path to leaf found

    def _traverse_tree(self, X: np.ndarray, tree_info: Dict, tree_idx: int) -> List[Tuple[int, str, float]]:
        """
        Traverse a single tree based on input data to find all visited nodes along the path.

        Args:
            X: Input data (single sample)
            tree_info: Dictionary containing tree information
            tree_idx: Index of the tree in the ensemble

        Returns:
            List of tuples (node_id, operation, threshold) for all nodes along the path
        """
        node_ids = tree_info['node_ids']
        feature_ids = tree_info['feature_ids']
        values = tree_info['values']
        modes = tree_info['modes']
        true_node_ids = tree_info['true_node_ids']
        false_node_ids = tree_info['false_node_ids']
        tree_ids = tree_info['tree_ids']

        # Find nodes belonging to this specific tree
        tree_mask = [i for i, tid in enumerate(tree_ids) if tid == tree_idx]

        if not tree_mask:
            return []

        # Create a mapping from node_id to index for this tree
        node_to_idx = {node_ids[i]: i for i in tree_mask}

        # Find the root node - it should be the one that's not referenced as a child
        all_child_nodes = set()
        for i in tree_mask:
            if true_node_ids[i] != 0:
                all_child_nodes.add(true_node_ids[i])
            if false_node_ids[i] != 0:
                all_child_nodes.add(false_node_ids[i])

        # Root node is the one that's not a child of any other node
        root_candidates = [node_ids[i] for i in tree_mask if node_ids[i] not in all_child_nodes]

        if not root_candidates:
            # Fallback: use first node in tree
            root_node_id = node_ids[tree_mask[0]]
        else:
            root_node_id = root_candidates[0]

        current_node_id = root_node_id
        visited_nodes = []

        while True:
            if current_node_id not in node_to_idx:
                break

            idx = node_to_idx[current_node_id]
            mode = modes[idx]
            feature_id = feature_ids[idx] if feature_ids else 0
            value = values[idx] if values else 0.0

            # Add this node to visited nodes (including its operation and threshold)
            visited_nodes.append((current_node_id, mode, value))

            # If we've reached a leaf, stop traversing
            if mode == b'LEAF' or mode == 'LEAF':
                break

            # If it's a branch node, traverse based on feature value
            if mode == b'BRANCH_LEQ' or mode == 'BRANCH_LEQ':
                # Get the feature value for this sample
                if feature_id < X.shape[0]:
                    feature_value = X[feature_id]

                    # Decide which path to take
                    if feature_value <= value:
                        next_node_id = true_node_ids[idx]
                    else:
                        next_node_id = false_node_ids[idx]

                    # Move to next node
                    if next_node_id != 0:  # 0 indicates no child
                        current_node_id = next_node_id
                    else:
                        break
                else:
                    # Feature ID out of bounds, stop
                    break
            else:
                # Unknown mode, stop
                break

        return visited_nodes

    def _hash_node(self, node_id: int, operation: str, threshold: float) -> int:
        """
        Hash a node using murmurhash3 based on {node_id} {operation} {threshold}.

        Args:
            node_id: Node identifier
            operation: Operation type (BRANCH_LEQ, LEAF, etc.)
            threshold: Threshold value for the split

        Returns:
            Hash value
        """
        node_string = f"{node_id} {operation} {threshold}"
        return mmh3.hash(node_string)

    def _compute_node_weight(self, distance: Union[int, float]) -> float:
        """
        Compute weight based on inverse distance to leaf node.

        Args:
            distance: Distance to leaf node

        Returns:
            Weight value using 1/d(x,y)^p formula
        """
        if distance == 0:
            return 1.0
        if distance == float('inf'):
            return 0.0  # Infinite distance gets zero weight
        return 1.0 / (distance ** self.distance_power)

    def _create_hash_vector_from_nodes(self, visited_nodes: List[Tuple[int, str, float]], tree_info: Dict, distances: Dict[int, Union[int, float]]) -> np.ndarray:
        """
        Create hash vector from all visited nodes along the path.

        Args:
            visited_nodes: List of tuples (node_id, operation, threshold) for visited nodes
            tree_info: Dictionary containing tree information
            distances: Dictionary mapping node_id to distance to leaf

        Returns:
            Hash vector of dimension hash_dim
        """
        hash_vector = np.zeros(self.hash_dim)

        for node_id, operation, threshold in visited_nodes:
            # Hash the node using its ID, operation, and threshold
            node_hash = self._hash_node(node_id, operation, threshold)

            # Compute weight based on distance to leaf
            distance = distances.get(node_id, 0)
            weight = self._compute_node_weight(distance)

            # Add weighted hash to vector
            hash_idx = abs(node_hash) % self.hash_dim
            hash_vector[hash_idx] += weight

        return hash_vector

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform input data to hash vector representation by traversing trees.

        Args:
            X: Input data matrix of shape (n_samples, n_features)

        Returns:
            Hash vector representation of shape (n_samples, hash_dim)
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        n_samples = X.shape[0]
        result = np.zeros((n_samples, self.hash_dim))

        for sample_idx in range(n_samples):
            sample = X[sample_idx]

            # Aggregate hash vectors from all trees
            sample_vector = np.zeros(self.hash_dim)

            for tree_idx, tree_info in enumerate(self.tree_ensembles):
                # Find unique tree IDs in this tree ensemble
                unique_tree_ids = list(set(tree_info['tree_ids']))

                for tree_id in unique_tree_ids:
                    # Traverse this specific tree to find visited nodes along the path
                    visited_nodes = self._traverse_tree(sample, tree_info, tree_id)

                    # Create hash vector from visited nodes
                    tree_vector = self._create_hash_vector_from_nodes(
                        visited_nodes, tree_info, self.tree_distances[tree_idx]
                    )

                    # Add to sample vector
                    sample_vector += tree_vector

            result[sample_idx] = sample_vector

        return result

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Alias for transform method to maintain sklearn-like interface.

        Args:
            X: Input data matrix

        Returns:
            Hash vector representation
        """
        return self.transform(X)


def create_tree_ensemble_hash(
    onnx_model: Union[ModelProto, Tuple[ModelProto, Any]],
    X: np.ndarray,
    hash_dim: int = 128,
    distance_power: float = 0.5
) -> np.ndarray:
    """
    Convenience function to create tree ensemble hash vector.

    Args:
        onnx_model: ONNX model containing tree ensemble operators
        X: Input data matrix
        hash_dim: Dimension for murmurhash3 output
        distance_power: Power parameter for distance weighting (0 < p < 1)

    Returns:
        Hash vector representation of the tree ensemble
    """
    hasher = TreeEnsembleHash(onnx_model, hash_dim=hash_dim, distance_power=distance_power)
    return hasher.transform(X)
```

## Using Local LLMs as an Arbiter/Classifier

_June 2025_

Using LLMs for doing arbitration or classification is nothing new. Generally I
found that "smaller" local LLMs struggled or led to overly optimistic results. I
think 'smaller' models are finally performant enough (to some extent at least)
where coming up with binary outcomes in a structured manner is "good enough". In
general as of writing things still aren't good enough. This is reflected in
things like the
[Goose Blogpost](https://block.github.io/goose/blog/2025/03/14/goose-ollama/)
where 32gb of RAM is still generally recommended. Here, I use the very good
`gemma-3-12b` model to accomplish this -- I generally found models smaller/older
than this failed to properly perform structured outputs in a meaningful manner.

The below example is based on
[Tricube Tales](https://www.drivethrurpg.com/en/product/294202/tricube-tales)
which is a TTRPG released under CC BY 3.0 license.

````py
import os
from pathlib import Path
from pprint import pprint
from typing import Literal

from llama_cpp import Llama
from llama_cpp.llama_tokenizer import LlamaHFTokenizer
from pydantic import BaseModel, model_validator

model_id = str(
    (
        Path(os.environ.get("HOME", "~")) / "dev/google_gemma-3-12b-it-qat-Q4_0.gguf"
    ).absolute()
)

prompt = """
You are "The Arbiter", an impartial rules adjudicator for the RPG Tricube Tales.
Your sole job is to determine which dice modifiers apply to a proposed action. Do not narrate outcomes or roll dice.

Your judgments follow this logic:

- Award +1 die if the action clearly aligns with the character's trait.
  - The action must directly reflect the strengths of that trait:
    - Brawny: feats of strength, endurance, or physical toughness
    - Agile: speed, balance, dexterity, or evasive finesse
    - Crafty: cleverness, deception, planning, or technical skill

- Apply -1 die if the character's quirk meaningfully impedes the action.
- Apply -1 die if the action is inherently challenging for most people in the setting, or the character lacks relevant expertise for the task

These modifiers are cumulative. The resulting diceModifier may be: +1, 0, -1, or -2.

Return your judgment using the following strict JSON schema:

```json
{
  "trait_bonus":   true | false,
  "quirk_penalty": true | false,
  "difficult":     true | false,
  "rationale":     "concise human-readable explanation (30-70 words)"
}
"""


class Arbiter(BaseModel):
    trait_bonus: bool
    quirk_penalty: bool
    difficult: bool
    rationale: str


class DiceModifierArbiter(Arbiter):
    dice_modifier: int | None = None

    @model_validator(mode="after")
    def validate_dice_modifier(self):
        modifier = 0
        if self.trait_bonus:
            modifier += 1
        if self.quirk_penalty:
            modifier -= 1
        if self.difficult:
            modifier -= 1
        self.dice_modifier = modifier
        return self


def create_chat_completion_with_schema(
    messages: list[dict], model: Llama, schema: type[BaseModel], **kwargs
) -> type[BaseModel]:
    output = model.create_chat_completion(
        messages=messages,
        response_format={
            "type": "json_object",
            "schema": schema.model_json_schema(),
        },
        **kwargs,
    )
    return schema.model_validate_json(output["choices"][0]["message"]["content"])


def determine_dice_modifier(context: str) -> DiceModifierArbiter:
    llama_model = Llama(
        model_path=model_id,
        tokenizer=LlamaHFTokenizer.from_pretrained("gemma-3-12b"),
        n_ctx=2048,
    )
    result = create_chat_completion_with_schema(
        [
            {"role": "system", "content": prompt},
            {"role": "user", "content": context},
        ],
        llama_model,
        Arbiter,
    )  # type: ignore
    return DiceModifierArbiter.model_validate(result.model_dump())


if __name__ == "__main__":
    result = []
    result.append(
        determine_dice_modifier("A brawny barbarian swings his sword at a goblin.")
    )
    result.append(
        determine_dice_modifier("A brawny barbarian draws his bow at a goblin.")
    )
    result.append(
        determine_dice_modifier("A brawny barbarian attempts to bargin with a vendor")
    )
    result.append(
        determine_dice_modifier(
            "Would a greedy, agile, thief successfully flee from a situation or get distracted with a treasure chest?"
        )
    )
    pprint(result)
````

Output:

```py
 DiceModifierArbiter(trait_bonus=True, quirk_penalty=False, difficult=False, rationale="Swinging a sword is a direct application of brawny strength. There's no indication of a quirk impeding the action, and while combat is challenging, it's not inherently difficult for most warriors.", dice_modifier=1)

 DiceModifierArbiter(trait_bonus=False, quirk_penalty=False, difficult=False, rationale="Drawing a bow is not inherently a feat of strength, endurance, or toughness. While barbarians might be proficient, it doesn't automatically grant a bonus. There's no indication of a quirk impeding the action, and drawing a bow isn't exceptionally difficult.", dice_modifier=0)

 DiceModifierArbiter(trait_bonus=False, quirk_penalty=False, difficult=True, rationale="While a barbarian might have some persuasive ability, bargaining is not a direct application of Brawny. It is also inherently difficult for most people, requiring social finesse and negotiation skills, which are not typically a barbarian's forte.", dice_modifier=-1)

 DiceModifierArbiter(trait_bonus=True, quirk_penalty=True, difficult=True, rationale="The thief's agility grants a bonus for fleeing. However, their greed introduces a penalty as they'd be tempted to grab treasure, hindering their escape. Fleeing a dangerous situation is generally difficult, warranting a further penalty.", dice_modifier=-1)
```

Using `Qwen3-8B` model also works relative well with this output:

```py
 DiceModifierArbiter(trait_bonus=True, quirk_penalty=False, difficult=False, rationale='Swinging a sword is a strength-based action, aligning with Brawny. No quirk or difficulty mentioned.', dice_modifier=1)

 DiceModifierArbiter(trait_bonus=False, quirk_penalty=False, difficult=False, rationale='Drawing a bow is not a strength-based action. Brawny traits relate to melee, not ranged combat.', dice_modifier=0)

 DiceModifierArbiter(trait_bonus=False, quirk_penalty=False, difficult=True, rationale='Brawny characters are physically strong but not necessarily skilled in negotiation. Bargaining is an inherently difficult task without relevant expertise.', dice_modifier=-1)

 DiceModifierArbiter(trait_bonus=False, quirk_penalty=True, difficult=False, rationale="The thief's greed (quirk) would likely distract them from fleeing, outweighing their agility.", dice_modifier=-1)]
```

## stdlib Numpy for when you don't need all of Numpy

_May 2025_

Sometimes you just want the convenience of numpy-like notation for slicing, and
setting values.

```py
from copy import deepcopy
import json
import re


class Array:
    def __init__(self, data):
        # Accepts list, list of lists, or another Array
        if isinstance(data, Array):
            self.data = self._deepcopy(data.data)
        else:
            self.data = self._deepcopy(data)
        self._update_shape()

    def _deepcopy(self, data):
        if isinstance(data, list):
            if data and isinstance(data[0], list):
                return [row[:] for row in data]
            else:
                return data[:]
        return data

    def _update_shape(self):
        if isinstance(self.data, list):
            if self.data and isinstance(self.data[0], list):
                self.shape = (len(self.data), len(self.data[0]))
            else:
                self.shape = (len(self.data),)
        else:
            self.shape = ()

    def __getitem__(self, key):
        if len(self.shape) == 2:
            if isinstance(key, tuple):
                row_key, col_key = key
                rows = self._process_slice(row_key, axis=0)
                cols = self._process_slice(col_key, axis=1)
                result = [[self.data[i][j] for j in cols] for i in rows]
                if len(result) == 1 and len(result[0]) == 1:
                    return result[0][0]
                if len(result) == 1:
                    return Array(result[0])
                if len(result[0]) == 1:
                    return Array([row[0] for row in result])
                return Array(result)
            else:
                # Single row
                rows = self._process_slice(key, axis=0)
                result = [self.data[i][:] for i in rows]
                if len(result) == 1:
                    return Array(result[0])
                return Array(result)
        elif len(self.shape) == 1:
            if isinstance(key, int):
                idx = key
                if idx < 0:
                    idx += self.shape[0]
                if idx < 0 or idx >= self.shape[0]:
                    raise IndexError("index out of range")
                return self.data[idx]
            elif isinstance(key, slice) or isinstance(key, list):
                idxs = self._process_slice(key, axis=0)
                result = [self.data[i] for i in idxs]
                return Array(result)
            else:
                raise TypeError(f"Invalid index type: {type(key)}")
        else:
            raise IndexError("Array is empty or has invalid shape.")

    def __setitem__(self, key, value):
        if len(self.shape) == 2:
            if isinstance(key, tuple):
                row_key, col_key = key
                rows = self._process_slice(row_key, axis=0)
                cols = self._process_slice(col_key, axis=1)
                if isinstance(value, Array):
                    value = value.data
                # Broadcasting for scalar assignment
                if not isinstance(value, list) or (
                    value and not isinstance(value[0], list)
                ):
                    value = [[value for _ in cols] for _ in rows]
                elif value and not isinstance(value[0], list):
                    value = [value] * len(rows)
                for i, row in enumerate(rows):
                    for j, col in enumerate(cols):
                        self.data[row][col] = value[i][j]
            else:
                rows = self._process_slice(key, axis=0)
                if isinstance(value, Array):
                    value = value.data
                if value and not isinstance(value[0], list):
                    value = [value] * len(rows)
                for i, row in enumerate(rows):
                    self.data[row] = value[i]
        elif len(self.shape) == 1:
            if isinstance(key, int):
                idx = key
                if idx < 0:
                    idx += self.shape[0]
                if idx < 0 or idx >= self.shape[0]:
                    raise IndexError("index out of range")
                self.data[idx] = value
            elif isinstance(key, slice) or isinstance(key, list):
                idxs = self._process_slice(key, axis=0)
                if isinstance(value, Array):
                    value = value.data
                # Broadcasting for scalar assignment
                if not isinstance(value, list):
                    value = [value] * len(idxs)
                if len(value) != len(idxs):
                    raise ValueError("could not broadcast input array to shape")
                for i, idx in enumerate(idxs):
                    self.data[idx] = value[i]
            else:
                raise TypeError(f"Invalid index type: {type(key)}")
        else:
            raise IndexError("Array is empty or has invalid shape.")

    def _process_slice(self, key, axis):
        n = self.shape[axis]
        if isinstance(key, int):
            if key < 0:
                key += n
            return [key]
        elif isinstance(key, slice):
            return list(range(*key.indices(n)))
        elif isinstance(key, list):
            return [(k + n if k < 0 else k) for k in key]
        else:
            raise TypeError(f"Invalid index type: {type(key)}")
    def __repr__(self):
        data_copy = deepcopy(self.data)
        max_string_length = 0
        is_numeric_data = True

        def process_value(val):
            nonlocal is_numeric_data
            str_val = json.dumps(val)
            if not isinstance(val, (int, float)):
                is_numeric_data = False
            return str_val

        def format_value(str_val):
            if is_numeric_data:
                return str_val.rjust(max_string_length).replace(" ", "_")
            return str_val.ljust(max_string_length)

        if len(self.shape) == 2:
            # Process 2D array values
            for i, row in enumerate(data_copy):
                for j, col in enumerate(row):
                    data_copy[i][j] = process_value(col)
                    max_string_length = max(max_string_length, len(data_copy[i][j]))

            # Format values
            for i, row in enumerate(data_copy):
                for j, col in enumerate(row):
                    data_copy[i][j] = format_value(data_copy[i][j])

            # Format output
            output = json.dumps(data_copy, indent=2)
            output = output.replace('"', "")
            output = re.sub(r"\n\s*(?=[^\s\[]|])", " ", output)
            if is_numeric_data:
                output = output.replace("_", " ")
            output = re.sub(r"\]\s*\]", "]\n]", output)

        else:  # 1D array
            # Process 1D array values
            for i, val in enumerate(data_copy):
                data_copy[i] = process_value(val)
                max_string_length = max(max_string_length, len(data_copy[i]))

            # Format values
            for i, val in enumerate(data_copy):
                data_copy[i] = format_value(data_copy[i])

            # Format output
            output = json.dumps(data_copy)
            output = output.replace('"', "")
            if is_numeric_data:
                output = output.replace("_", " ")

        return output


def array(data):
    return Array(data)


def zeros(shape):
    if isinstance(shape, int):
        return Array([0 for _ in range(shape)])
    elif len(shape) == 1:
        return Array([0 for _ in range(shape[0])])
    elif len(shape) == 2:
        rows, cols = shape
        return Array([[0 for _ in range(cols)] for _ in range(rows)])
    else:
        raise ValueError("Only 1D and 2D arrays are supported.")


def ones(shape):
    if isinstance(shape, int):
        return Array([1 for _ in range(shape)])
    elif len(shape) == 1:
        return Array([1 for _ in range(shape[0])])
    elif len(shape) == 2:
        rows, cols = shape
        return Array([[1 for _ in range(cols)] for _ in range(rows)])
    else:
        raise ValueError("Only 1D and 2D arrays are supported.")
```

## Prompting ChatGPT to Generate Prompts

_March 2025_

Okay, so this isn't that "new" though sometimes its hard to find a "winner".
This is one that I like for writing code reviews.

**Prompt**

```
Write me a chatgpt prompt for reviewing code PRs specifically off git diff. that does the following:

- Summarizes the changes
- Code quality and adherence to best practices
- Potential bugs or edge cases
- Performance optimizations
- Readability and maintainability
- Any security concerns

Suggest improvements and explain your reasoning for each suggestion. Keep each element short and succint

At the end of the review it should also give a rating on the estimated risk of merge
```

The generated output:

```
Here’s a solid ChatGPT prompt for reviewing a PR using a git diff:

Prompt:

You are a highly skilled code reviewer analyzing a git diff. Your task is to review the changes and provide a structured response covering:
	1.	Summary of Changes: Briefly describe what was modified.
	2.	Code Quality & Best Practices: Note any deviations from coding standards.
	3.	Potential Bugs & Edge Cases: Highlight any potential issues.
	4.	Performance Optimizations: Suggest improvements for efficiency.
	5.	Readability & Maintainability: Evaluate how easy it is to understand and extend.
	6.	Security Concerns: Identify vulnerabilities or security risks.
	7.	Suggested Improvements: Provide specific recommendations and reasoning.

At the end, provide an Estimated Merge Risk rating (Low, Medium, High) based on potential issues and complexity. Keep feedback concise but actionable.

Input: Provide the git diff for analysis.
```

What is interesting is how near (or far?) it is from curated offerings. For
example this is the one from
[continue.dev](https://github.com/continuedev/prompt-file-examples/blob/main/code-review.prompt)

    <system>
    You will be acting as a senior software engineer performing a code review for a colleague.
    </system>

    You will follow the guidelines for giving a great code review outlined below:
    {{{ url "https://google.github.io/eng-practices/review/reviewer/looking-for.html" }}}

    ---

    Here is the proposed code changes you will be reviewing:
    {{{ diff }}}

    ---


    Do not include a greeting. Immediately begin reviewing the changes.

    For each file, decide if you need to provide any feedback on the changes.
    If so, outline the feedback using one or two sentences.
    If a code change is required, then mention the original code, and
    then propose a code change to fix it.
    Do not add any other text after the suggestion.
    If you have no feedback on a file, do not add a comment for that file.
    Lastly, provide a one to two summary of your feedback at the end.

    Here are some examples.

    <example>
    ### filename.js
    The name of this variable is unclear.

    Original:
    ```js
    const x = getAllUsers();
    ```

    Suggestion:
    ```js
    const allUsers = getAllUsers();
    ```
    </example>

    <example>
    ### filename.js
    This code is overly complex.

    Original:
    ```py
    class AgeCalculator:
        def __init__(self, birth_year):
            self.birth_year = birth_year

        def calculate_age(self, current_year):
            age = current_year - self.birth_year
            return self._validate_and_format_age(age)

        def _validate_and_format_age(self, age):
            if age < 0:
                raise ValueError("Invalid age calculated")
            return f"User is {age} years old"

    def get_user_age(birth_year, current_year):
        calculator = AgeCalculator(birth_year)
        return calculator.calculate_age(current_year)
    ```

    Suggestion:
    ```python
    def get_user_age(birth_year, current_year):
        return current_year - birth_year
    ```
    </example>

    <example>
    ### Summary
    Overall, these changes appear to be minor improvements to the
    project structure and code cleanliness.
    </example>


    Here is the additional input from the code author:

    <input>
    {{ input }}
    </input>


    Think through your feedback step by step before replying.

## IsoForest with k-NN

_February 2025_

Following on the general theme of using random-ness to infer or generate
predictions, I've always been very curious about isolation forests, more
specifically given the innate measure of "similarity" or "distance" can we use
this in the supervised learning scenario via k-NN?

Below is some code which implements this completely using stdlib Python with no
dependencies on `numpy`. Written with heavy assistance from LLMs.

```python
import random
import math
from collections import Counter

# ----------------------
# 1) Isolation Forest
# ----------------------


class IsolationTree:
    """
    A single isolation tree.
    """

    def __init__(self, max_depth, seed=None):
        self.max_depth = max_depth
        # We'll store a seed to replicate the random picks
        # If None, we use the global random generator
        self.seed = seed if seed is not None else random.randint(0, 2**31 - 1)

        self.split_feature = None
        self.split_value = None
        self.left = None
        self.right = None
        self.size = 0  # number of samples in this node (for leaf nodes)

    def fit(self, data, depth=0, rng=None):
        """
        Recursively build the isolation tree.

        :param data: List of data points (each a list of features).
        :param depth: Current depth of the tree.
        :param rng: A random.Random instance. If None, we build one using self.seed.
        """
        if rng is None:
            rng = random.Random(self.seed)  # create a local RNG

        n = len(data)
        self.size = n

        # Base case: stop if max depth reached or not enough samples
        if depth >= self.max_depth or n <= 1:
            return

        # Number of features
        num_features = len(data[0])

        # Randomly pick a feature to split on
        self.split_feature = rng.randint(0, num_features - 1)

        # Get min and max of that feature
        feature_values = [point[self.split_feature] for point in data]
        min_val = min(feature_values)
        max_val = max(feature_values)

        # If all points are the same on this feature, stop splitting
        if min_val == max_val:
            self.split_feature = None
            return

        # Randomly pick a split value
        split_val = rng.uniform(min_val, max_val)
        self.split_value = split_val

        # Partition data into left and right
        left_data = []
        right_data = []
        for point in data:
            if point[self.split_feature] < split_val:
                left_data.append(point)
            else:
                right_data.append(point)

        # Create child nodes
        self.left = IsolationTree(self.max_depth)
        self.right = IsolationTree(self.max_depth)

        # Recursively fit child nodes (reuse the same rng for consistency)
        self.left.fit(left_data, depth + 1, rng)
        self.right.fit(right_data, depth + 1, rng)

    def path_length(self, x, depth=0):
        """
        Compute the path length of a sample x in this isolation tree.
        Includes the c_factor for leaf nodes.
        """
        # If leaf node or no further split
        if self.split_feature is None or self.left is None or self.right is None:
            return depth + c_factor(self.size)

        # Check which branch x goes to
        if x[self.split_feature] < self.split_value:
            return self.left.path_length(x, depth + 1)
        else:
            return self.right.path_length(x, depth + 1)

    def shared_depth(self, x, y, depth=0):
        """
        Compute how many levels x and y share in the same branch
        before diverging (or reaching a leaf).
        """
        # If we are at a leaf node or no further split
        if self.split_feature is None or self.left is None or self.right is None:
            return depth

        x_left = x[self.split_feature] < self.split_value
        y_left = y[self.split_feature] < self.split_value

        if x_left != y_left:
            return depth

        if x_left:
            return self.left.shared_depth(x, y, depth + 1)
        else:
            return self.right.shared_depth(x, y, depth + 1)


def harmonic_number(n):
    """
    Return the nth harmonic number H_n = 1 + 1/2 + 1/3 + ... + 1/n
    """
    return sum(1.0 / i for i in range(1, n + 1))


def c_factor(n):
    """
    Normalization factor commonly used in isolation forest scoring:
        c(n) = 2 * H_{n-1} - 2*(n-1)/n
    For large n, c(n) ~ 2 * ln(n-1) + gamma - 2*(n-1)/n
    """
    if n <= 1:
        return 0
    return 2.0 * harmonic_number(n - 1) - 2.0 * (n - 1) / n


class IsolationForest:
    """
    An Isolation Forest: collection of IsolationTrees.
    """

    def __init__(self, n_estimators=10, max_samples=256, max_depth=8, random_seed=None):
        """
        :param n_estimators: Number of isolation trees.
        :param max_samples: Subsample size for each tree.
        :param max_depth: Maximum depth of each tree.
        :param random_seed: Optional integer seed for reproducibility.
        """
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_depth = max_depth
        self.trees = []

        # For partial_fit, we want a random generator we can use each time
        # to pick which trees to replace, etc.
        if random_seed is not None:
            self._rng = random.Random(random_seed)
        else:
            self._rng = random.Random()

    def fit(self, data):
        """
        Train the isolation forest on the given dataset.

        :param data: List of data points (each a list of features).
        """
        self.trees = []
        n = len(data)

        for _ in range(self.n_estimators):
            # Subsample the data
            if n > self.max_samples:
                subset = random.sample(data, self.max_samples)
            else:
                subset = data[:]

            tree = IsolationTree(self.max_depth)
            tree.fit(subset)
            self.trees.append(tree)

    def path_length(self, x):
        """
        Compute the average path length of x across all trees.

        :param x: A single data point (list of features).
        :return: Average path length as a float.
        """
        total_path = 0.0
        for tree in self.trees:
            total_path += tree.path_length(x)
        return total_path / float(len(self.trees))

    def anomaly_score(self, x):
        """
        Compute the anomaly (outlier) score for a single sample x.

        A common formula is:
            score(x) = 2^(- E(path_length) / c_factor(max_samples))
        """
        avg_path = self.path_length(x)
        # Classical isolation score:
        return math.pow(2, -avg_path / c_factor(self.max_samples))

    def predict(self, data, threshold=0.5):
        """
        Classify data points as -1 (outlier) or 1 (inlier) based on a threshold.
        """
        preds = []
        for x in data:
            score = self.anomaly_score(x)
            preds.append(-1 if score > threshold else 1)
        return preds

    def similarity_score(self, x, y):
        """
        Compute a similarity score between two points x and y, based on
        shared depth across all trees (normalized by max_depth).

        The higher the score, the more 'similar' the points are
        in the sense of how the forest partitions the space.
        """
        total_shared_depth = 0
        for tree in self.trees:
            total_shared_depth += tree.shared_depth(x, y, depth=0)

        avg_shared_depth = total_shared_depth / float(len(self.trees))
        # Normalize by the maximum possible depth
        similarity = avg_shared_depth / float(self.max_depth)
        return similarity

    def partial_fit(self, data, n_replace=1):
        """
        *Heuristic* partial fit that:
          1) Takes the new minibatch 'data'
          2) Randomly picks 'n_replace' trees to remove
          3) Builds 'n_replace' new trees on the minibatch (up to max_samples)
             reusing seeds from the removed trees if desired
        """
        if not self.trees:
            # If forest was empty, just do a fresh fit with n_estimators
            self.fit(data)
            return

        # Bound n_replace by the current number of trees
        n_replace = min(n_replace, len(self.trees))

        # 1) Randomly pick which trees to remove
        remove_indices = self._rng.sample(range(len(self.trees)), n_replace)
        remove_indices = set(remove_indices)

        # 2) For each tree to remove, gather its seed to reuse
        removed_seeds = []
        kept_trees = []
        for i, tree in enumerate(self.trees):
            if i in remove_indices:
                removed_seeds.append(tree.seed)
            else:
                kept_trees.append(tree)

        # 3) Build n_replace new trees on the minibatch
        #    If minibatch larger than max_samples, sample
        n_data = len(data)
        if n_data > self.max_samples:
            subset = self._rng.sample(data, self.max_samples)
        else:
            subset = data[:]

        new_trees = []
        for seed in removed_seeds:
            # We create a new tree reusing the old tree's seed
            new_tree = IsolationTree(self.max_depth, seed=seed)
            new_tree.fit(subset)
            new_trees.append(new_tree)

        # 4) Update self.trees
        self.trees = kept_trees + new_trees

    def __len__(self):
        """
        Number of trees in the forest.
        """
        return len(self.trees)


# -------------------------------------
# 2) IsoKNNClassifier (Meta-Learner)
# -------------------------------------


class IsoKNNClassifier:
    """
    A classifier that uses Isolation Forest to define a similarity metric,
    then applies a k-Nearest Neighbors approach based on highest similarity.
    """

    def __init__(self, k=3, iso_forest_params=None):
        """
        :param k: Number of neighbors to consider.
        :param iso_forest_params: Dictionary of parameters for IsolationForest.
        """
        self.k = k
        if iso_forest_params is None:
            iso_forest_params = {}
        self.iso_forest = IsolationForest(**iso_forest_params)
        self.X_train = []
        self.y_train = []

    def fit(self, X, y):
        """
        Fit the IsoKNN model: store training data and build the Isolation Forest.

        :param X: List of feature vectors.
        :param y: List of corresponding labels (same order).
        """
        # Make sure X and y are in a usable structure
        self.X_train = X
        self.y_train = y

        # Train the Isolation Forest on X
        self.iso_forest.fit(X)

    def predict(self, X_test):
        """
        Predict labels for a list of test points using the IsoForest-based kNN.

        :param X_test: List of feature vectors for testing.
        :return: List of predicted labels.
        """
        predictions = []
        for x in X_test:
            # 1. Compute similarity to each training point
            similarities = []
            for i, x_train in enumerate(self.X_train):
                sim = self.iso_forest.similarity_score(x, x_train)
                similarities.append((sim, self.y_train[i]))

            # 2. Sort by similarity in descending order
            similarities.sort(key=lambda tup: tup[0], reverse=True)

            # 3. Get top-k neighbors
            top_k = similarities[: self.k]

            # 4. Majority vote among neighbors
            labels = [label for (_, label) in top_k]
            label_count = Counter(labels)
            # Pick the label with the highest count; break ties arbitrarily
            majority_label = label_count.most_common(1)[0][0]
            predictions.append(majority_label)

        return predictions


# ----------------
# Example Usage
# ----------------


def main():
    # Synthetic training data
    X_train = [
        [0.1, 0.2],  # label 0
        [0.2, 0.1],  # label 0
        [0.15, 0.18],  # label 0
        [5.0, 6.0],  # label 1
        [5.2, 5.9],  # label 1
        [10.0, 10.0],  # label 2
    ]
    y_train = [0, 0, 0, 1, 1, 2]

    # Define test points
    X_test = [
        [0.15, 0.17],  # Close to first cluster => likely label 0
        [5.1, 6.1],  # Close to second cluster => likely label 1
        [10.1, 9.9],  # Close to the outlier => likely label 2
        [4.9, 5.8],  # Possibly near cluster (5.0, 6.0) => label 1
    ]

    # Create and fit IsoKNNClassifier
    # We'll use an Isolation Forest with a few trees for demonstration.
    iso_knn = IsoKNNClassifier(
        k=2,
        iso_forest_params={
            "n_estimators": 5,
            "max_samples": 4,
            "max_depth": 5,
            "random_seed": 42,
        },
    )
    iso_knn.fit(X_train, y_train)

    # Predict
    preds = iso_knn.predict(X_test)
    for x, label in zip(X_test, preds):
        print(f"Test point {x} => predicted label: {label}")


def demo_partial_fit():
    """
    Minimal demonstration of partial_fit usage.
    """
    # Original data
    data_initial = [
        [0.1, 0.2],
        [0.2, 0.1],
        [0.15, 0.18],
        [5.0, 6.0],
        [5.2, 5.9],
        [4.9, 6.2],
    ]

    # New data (minibatch) to incorporate
    data_minibatch = [[10.0, 10.0], [10.1, 9.9]]

    # Build an initial forest
    iso_f = IsolationForest(n_estimators=4, max_depth=5, max_samples=3, random_seed=42)
    iso_f.fit(data_initial)
    print("Initial number of trees =", len(iso_f))

    # Check anomaly scores before partial_fit
    outlier_score_before = iso_f.anomaly_score([10.0, 10.0])

    # partial_fit with new data, replacing 2 trees
    iso_f.partial_fit(data_minibatch, n_replace=2)
    print("Number of trees after partial_fit =", len(iso_f))

    # Check anomaly scores after partial_fit
    outlier_score_after = iso_f.anomaly_score([10.0, 10.0])

    print(f"Outlier score (before) = {outlier_score_before:.3f}")
    print(f"Outlier score (after)  = {outlier_score_after:.3f}")


if __name__ == "__main__":
    main()
    print("== Partial fit demo ==")
    demo_partial_fit()
```

Commentary from the future: this approach is very similar to Mondrian Trees, in
which the random partitions are induced by a Mondrian Process. I should include
a "fast approximation" to Mondrian processes and expand it to multi-dimensions
using standard lib in the future.

## Brownian Bridges

_February 2025_

I got LLM via prompting to recreate how it may have generated the simple kriging
diagram on the wikipedia page. I intentionally got it to do it via brownian
bridges rather than the "typical" guassian process way simply because I was
interested in brownian bridges. I may have made changes for ease of
implementation (I was thinking about re-implementing it using stdlib or in
another language as a close approximation), though the "idea" remains. This
technique maybe useful if one wants to do something better than sampling over a
grid search.

To extend to multiple dimensions, we can make use of tree-parzen like approach
to sample per region (chosen heuristically) and presume univariate relationships
(which was the approach taken in the original paper, though later approaches
used multivariate KDEs). For the purposes of a simplistic implementation,
simplifying assumptions will suffice. For example, it could be that the
tree-structure restricts sampling to a fixed number of points per a region to
determine whether or not a split is necessary.

```python
# %%
import numpy as np
from numpy.random import default_rng
from numpy.random import default_rng
import matplotlib.pyplot as plt

def brownian_realisation(x_obs, y_obs, x_pred, sigma=1.0, mean=0.0, rng=None):
    """
    Generate one realization of the Gaussian process (Brownian motion
    with variance scale sigma^2) at the locations x_pred,
    conditioned on the data (x_obs, y_obs).

    Parameters
    ----------
    x_obs : array-like
        Observed locations (1D).
    y_obs : array-like
        Observed values corresponding to x_obs.
    x_pred: array-like
        Locations where we want a realization.
    sigma : float, optional
        Standard deviation parameter for Brownian motion. Cov = sigma^2 * min(t1, t2).
    mean : float or array-like, optional
        Global mean (scalar or an array the same length as x_obs/x_pred).
    rng : np.random.Generator or None
        Random generator for reproducibility. If None, a new default generator is used.

    Returns
    -------
    y_pred_samp : ndarray
        One random sample (realization) of the GP at x_pred.
    """
    if rng is None:
        rng = default_rng()

    # Convert to arrays
    x_obs = np.asarray(x_obs, dtype=float)
    y_obs = np.asarray(y_obs, dtype=float)
    x_pred = np.asarray(x_pred, dtype=float)

    # Sort so everything is in ascending order (not strictly required, but good practice)
    # (If your x_obs, x_pred come sorted, this step can be skipped or adapted.)
    idx_obs_sort = np.argsort(x_obs)
    x_obs = x_obs[idx_obs_sort]
    y_obs = y_obs[idx_obs_sort]

    # offset by the initial point as it is assumed to be 0
    y_obs_initial = y_obs[0]
    y_obs = y_obs - y_obs_initial

    idx_pred_sort = np.argsort(x_pred)
    x_pred_sorted = x_pred[idx_pred_sort]

    # Build entire (obs+pred) set of x's
    x_all = np.concatenate([x_obs, x_pred_sorted])

    # Build full covariance
    K_all = brownian_covariance(x_all, sigma=sigma)

    # Partition covariance
    n_obs = len(x_obs)
    n_pred = len(x_pred_sorted)

    K_oo = K_all[:n_obs, :n_obs]               # Cov among observed points
    K_op = K_all[:n_obs, n_obs:]               # Cov between obs and pred
    K_po = K_all[n_obs:, :n_obs]               # Cov between pred and obs
    K_pp = K_all[n_obs:, n_obs:]               # Cov among pred points

    # Mean vectors
    # We'll assume 'mean' is a scalar. If it's not, adapt below for shape
    m_o = mean * np.ones(n_obs)
    m_p = mean * np.ones(n_pred)

    # Compute the conditional mean
    # y_obs - m_o is the 'centered' observed data
    K_oo_inv = np.linalg.pinv(K_oo)
    cond_mean = m_p + K_po @ K_oo_inv @ (y_obs - m_o)

    # Compute the conditional covariance
    cond_cov = K_pp - K_po @ K_oo_inv @ K_op

    # Draw one realization from N(cond_mean, cond_cov)
    y_pred_sample = rng.multivariate_normal(mean=cond_mean, cov=cond_cov)

    # Reorder to match the original order of x_pred
    y_pred_samp_original_order = np.empty_like(y_pred_sample)
    y_pred_samp_original_order[idx_pred_sort] = y_pred_sample

    return y_pred_samp_original_order + y_obs_initial


def brownian_bridge_realisation(
    x_obs,
    y_obs,
    x_pred,
    sigma=1.0,
    mean=0.0,
    rng=None
):
    """
    Generate a 1D Brownian motion sample that is 'pinned' to observed data
    at certain locations (times) WITHOUT doing the full covariance-matrix inversion.

    This uses the Brownian-bridge property on each segment between consecutive
    observed points. In 1D, this produces the exact conditional sample:

       Y(x_obs[i]) = y_obs[i]    for all observed i,

    and at any intermediate x_pred in that segment, Y is drawn from a
    Brownian bridge. This avoids the NxN matrix inversion from the usual
    'brownian_realisation' approach.

    Parameters
    ----------
    x_obs : array-like
        The observed (pinned) locations (1D), shape (n_obs,).
    y_obs : array-like
        The observed values at x_obs, shape (n_obs,).
    x_pred: array-like
        The locations (times) where we want to simulate the Brownian motion.
    sigma : float, optional
        Brownian motion scale parameter. Covariance is sigma^2 * min(t1, t2).
        Equivalently, increments have variance sigma^2 * (t_{i+1} - t_i).
    mean : float, optional
        Global constant mean for the process. If not 0, we effectively
        do Brownian bridging on (y_obs - mean) and then add 'mean' at the end.
    rng : np.random.Generator or None
        NumPy random generator for reproducibility. If None, use default_rng().

    Returns
    -------
    y_pred : ndarray
        A single realization of the Brownian motion at x_pred, shaped like x_pred.
    """

    if rng is None:
        rng = default_rng()

    # Convert inputs to arrays
    x_obs = np.asarray(x_obs, dtype=float)
    y_obs = np.asarray(y_obs, dtype=float)
    x_pred = np.asarray(x_pred, dtype=float)

    # 1) Sort the observed data by x_obs
    sort_obs_idx = np.argsort(x_obs)
    x_obs_sorted = x_obs[sort_obs_idx]
    y_obs_sorted = y_obs[sort_obs_idx]

    # 2) Sort x_pred
    sort_pred_idx = np.argsort(x_pred)
    x_pred_sorted = x_pred[sort_pred_idx]

    # 3) Merge (x_obs_sorted) and (x_pred_sorted) into a single sorted array x_all
    #    We'll keep all unique points but preserve order.
    x_all = np.unique(np.concatenate([x_obs_sorted, x_pred_sorted]))

    # We'll create an array to hold the entire path: Y(x_all)
    # For pinned points (observed), we fix Y. For the rest, we fill by bridging.
    y_all = np.empty_like(x_all, dtype=float)

    # A helper dictionary for pinned values: pinned_dict[x_obs[i]] = y_obs[i]
    # We'll store 'centered' pinned values by subtracting the global mean
    pinned_dict = {}
    for xo, yo in zip(x_obs_sorted, y_obs_sorted):
        pinned_dict[xo] = yo - mean

    # Mark which of x_all are pinned vs. not
    pinned_mask = np.array([ (x in pinned_dict) for x in x_all ])

    # Fill in pinned points in y_all
    y_all[:] = np.nan
    for i, xval in enumerate(x_all):
        if pinned_mask[i]:
            y_all[i] = pinned_dict[xval]

    # Indices of pinned points
    pinned_indices = np.where(pinned_mask)[0]

    # If there are no pinned points, or only one pinned point,
    # you might define boundary conditions or do an unconditional random walk.
    # For a minimal demonstration, let's handle segments only if we have >= 2 pins.
    if len(pinned_indices) == 0:
        # No data at all: unconditionally simulate from 0 up to max(x_all)
        # For a pure Brownian motion with mean=0, we can do increments in ascending x_all.
        # Then add 'mean' at the end.
        y_all[0] = 0.0
        for i in range(1, len(x_all)):
            dt = x_all[i] - x_all[i-1]  # time step
            # Increments ~ Normal(0, sigma^2 * dt)
            incr = rng.normal(loc=0.0, scale=sigma*np.sqrt(dt))
            incr = rng.uniform(low=-sigma*np.sqrt(dt), high=sigma*np.sqrt(dt))
            y_all[i] = y_all[i-1] + incr
        y_all += mean

    else:
        # We'll do piecewise bridging from one pinned index to the next
        # If pinned_indices doesn't include the first or last index,
        # we might do an unconditional portion before the first pin or after the last pin.

        # Handle any region before the first pinned index (unconditional from x_all[0] to that pinned point)
        first_pin = pinned_indices[0]
        if first_pin > 0:
            # From i=0 up to i=first_pin
            # We'll do a "bridge" from (x_all[0], random start) to (x_all[first_pin], pinned_value)
            # If you want a known start, set y_all[0] = 0 or something. We'll just do it unconditionally
            # until we reach the pinned point, so let's pick y_all[0] = pinned_value as a hack, or 0.
            # We'll do an unconditional path and then shift so that at i=first_pin we get pinned.
            #
            # Simpler approach:
            #   y_all[0] = pinned_value_of_the_first_pin to avoid a big jump,
            # or
            #   start from 0.0 and see where we end up, then shift everything.
            # We'll do something straightforward for demonstration:
            y_all[0] = y_all[first_pin]  # pin the start to the same y so variance is only from increments

            for i in range(1, first_pin+1):
                dt = x_all[i] - x_all[i-1]
                incr = rng.normal(loc=0.0, scale=sigma*np.sqrt(dt))
                incr = rng.uniform(low=-sigma*np.sqrt(dt), high=sigma*np.sqrt(dt))
                y_all[i] = y_all[i-1] + incr

            # Now forcibly set the pinned index to pinned value
            y_all[first_pin] = pinned_dict[x_all[first_pin]]

        # Now do bridging on each segment between consecutive pinned points
        for seg_start, seg_end in zip(pinned_indices, pinned_indices[1:]):
            # We have x_left, y_left and x_right, y_right
            x_left = x_all[seg_start]
            y_left = y_all[seg_start]   # pinned value
            x_right = x_all[seg_end]
            y_right = y_all[seg_end]    # pinned value

            # The interior points are seg_start+1 .. seg_end-1
            # We'll fill them in a forward pass using the Brownian bridge increments
            for i in range(seg_start+1, seg_end):
                t_i   = x_all[i-1]
                y_i   = y_all[i-1]
                t_next = x_all[i]

                # Markov property for Brownian bridge:
                #  Y(t_next) | Y(t_i)=y_i, Y(x_right)=y_right is normal with:
                #     mean = y_i + ((t_next - t_i)/(x_right - t_i)) * (y_right - y_i)
                #     var  = sigma^2 * (t_next - t_i) * (x_right - t_next) / (x_right - t_i)
                #
                dt = (t_next - t_i)
                denom = (x_right - t_i)

                # Mean increment (drift toward y_right)
                mean_inc = (dt/denom) * (y_right - y_i)

                # Variance of that increment
                var_inc = sigma**2 * dt * (x_right - t_next) / denom
                sd_inc = np.sqrt(var_inc)

                # Sample from Normal( mean_inc, sd_inc^2 )
                inc = rng.normal(loc=mean_inc, scale=sd_inc)
                inc = rng.uniform(low=mean_inc-sd_inc, high=mean_inc + sd_inc)

                y_all[i] = y_i + inc

            # do we forcibly set end point to pinned?
            # y_all[seg_end] = y_right

        # Handle any region after the last pinned index
        last_pin = pinned_indices[-1]
        if last_pin < len(x_all) - 1:
            # unconditional from that pinned point to the end
            for i in range(last_pin+1, len(x_all)):
                dt = x_all[i] - x_all[i-1]
                incr = rng.normal(loc=0.0, scale=sigma*np.sqrt(dt))
                incr = rng.uniform(low=-sigma*np.sqrt(dt), high=sigma*np.sqrt(dt))
                y_all[i] = y_all[i-1] + incr

        # Finally, add 'mean' back in, since we subtracted it from y_obs
        y_all += mean

    # 4) We only need to return values at x_pred
    #    So we'll make a dict from x_all -> y_all
    val_dict = {x: y for (x, y) in zip(x_all, y_all)}

    # 5) Extract the results in the original order of x_pred
    y_pred_out = np.array([val_dict[x] for x in x_pred_sorted])

    # Re-map to the original (unsorted) x_pred order
    y_pred = np.empty_like(y_pred_out)
    y_pred[sort_pred_idx] = y_pred_out

    return y_pred


# %%
def simple_kriging(x_obs, y_obs, x_pred, n_reals=1000, sigma=1.0, mean=0.0, rng=None, use_random=False):
    """
    Perform simple kriging under Brownian motion assumption by Monte Carlo.

    1. Draw multiple conditional realizations from the GP.
    2. Compute empirical mean and credible intervals from those realizations.

    Parameters
    ----------
    x_obs : array-like
        Observed locations.
    y_obs : array-like
        Observed values.
    x_pred: array-like
        Prediction locations.
    n_reals : int
        Number of Monte Carlo realizations to draw.
    sigma : float
        Standard deviation parameter for Brownian motion.
    mean : float
        Global mean assumption for "simple" kriging.
    rng : np.random.Generator or None
        Random generator for reproducibility.

    Returns
    -------
    pred_mean : ndarray
        The empirical mean of the GP at x_pred over all realizations.
    pred_lower : ndarray
        The lower bound of the 95% credible interval (2.5th percentile).
    pred_upper : ndarray
        The upper bound of the 95% credible interval (97.5th percentile).
    all_realizations : ndarray
        All realizations of shape (n_reals, len(x_pred)).
    """
    if rng is None:
        rng = default_rng()

    x_pred = np.asarray(x_pred, dtype=float)
    all_realizations = []

    # Draw n_reals times
    for _ in range(n_reals):
        sample = brownian_bridge_realisation(
            x_obs, y_obs, x_pred, sigma=sigma, mean=mean, rng=rng
        )
        all_realizations.append(sample)

    all_realizations = np.array(all_realizations)  # shape (n_reals, n_pred)

    # Empirical stats across realizations
    pred_mean = np.mean(all_realizations, axis=0)
    pred_lower = np.percentile(all_realizations, 2.5, axis=0)
    pred_upper = np.percentile(all_realizations, 97.5, axis=0)

    return pred_mean, pred_lower, pred_upper, all_realizations


# %%
# Observed data
x_obs = np.array([0.0, 2.0, 3.0, 7.0, 8.0])
y_obs = np.array([1.4, 1.2, 0.9, 2.5, 2.2])

# Points where we want to predict
x_pred = np.linspace(0, 8, 50)  # 50 points from 0 to 8

# Perform simple kriging via Brownian motion realisations
pred_mean, pred_lower, pred_upper, reals = simple_kriging(
    x_obs, y_obs, x_pred,
    n_reals=2000,
    sigma=1.0,      # if you believe the process has variance sigma^2
    mean=0.0        # simple kriging with known mean 0
)

# Print some results
# print("Prediction Mean:", pred_mean)
# print("95% Credible Interval Lower:", pred_lower)
# print("95% Credible Interval Upper:", pred_upper)

# Plotting (optional) -- requires matplotlib
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
# Observed data
plt.plot(x_obs, y_obs, 'ko', label='Observations')
# Mean
plt.plot(x_pred, pred_mean, 'b-', label='Posterior mean')

# print some realisations
plt.plot(x_pred, reals[0, :], '-', label='A single realization')

# 95% Credible Interval
plt.fill_between(x_pred, pred_lower, pred_upper, color='blue', alpha=0.2, label='95% CI')
plt.legend()
plt.title("Simple Kriging under Brownian Motion Assumption")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
```

![output](/assets/2025/brownian-bridges-simple-kriging.png)
