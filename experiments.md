# Experiments

A collection of experiments, code dump etc, that are more for the vibe than for the writing (i.e. these may be heavily
written by AI, but are still "cool").

# 2026

## Full Text Search

Although semantic search is the typical go-to full text search generally doesn't get as much attention due to the
perceived difficulty to setup. [Here](https://github.com/NoRaincheck/fulltextsearch/tree/main) is a `duckdb` variant
that should help with that

```py
"""
A full-text search client using DuckDB for local RAG (Retrieval Augmented Generation) applications.

This module provides a simple interface for creating collections of documents,
inserting documents with metadata, and performing full-text search operations
using DuckDB's full-text search capabilities.
"""

# from duckdb_extensions import import_extension

# import_extension("fts", force_install=True)

import duckdb
import orjson
from liquid import Template

type Id = str
type Content = str
type Metadata = dict[str, str]

create_collection = Template("""
CREATE TABLE {{collection}} (
    id VARCHAR PRIMARY KEY,
    content TEXT,
    metadata JSON
);
""")

insert_into_collection = Template("""
    INSERT into {{collection}} VALUES (?, ?, ?)
""")

insert_or_replace_into_collection = Template("""
    INSERT OR REPLACE into {{collection}} VALUES (?, ?, ?)
""")

count_collection = Template("""
    SELECT COUNT(*) FROM {{collection}}
""")

create_fts_index = Template("""
    PRAGMA create_fts_index('{{collection}}', 'id', 'content', overwrite=1)
""")

search_with_metadata_filter = Template("""
    SELECT id, content, metadata, score
    FROM (
        SELECT id, content, metadata, fts_main_{{collection}}.match_bm25(id, ?) AS score
        FROM {{collection}}
    )
    WHERE score IS NOT NULL
    AND {{where_clause}}
    ORDER BY score DESC
    {{limit_clause}}
""")

search_without_metadata_filter = Template("""
    SELECT id, content, metadata, score
    FROM (
        SELECT id, content, metadata, fts_main_{{collection}}.match_bm25(id, ?) AS score
        FROM {{collection}}
    )
    WHERE score IS NOT NULL
    ORDER BY score DESC
    {{limit_clause}}
""")


class Client:
    """
    A client for performing full-text search operations using DuckDB.

    This client allows you to create collections (tables) of documents,
    insert documents with content and metadata, and perform full-text searches
    on the content.

    Attributes:
        con: The DuckDB connection object used for database operations.
    """

    def __init__(self, db: str = ":memory:"):
        """
        Initialize the Client with a DuckDB connection.

        Args:
            db: Path to the DuckDB database file. Use ':memory:' for an
                in-memory database (default). For persistent storage, provide
                a file path.
        """
        self.con = duckdb.connect(db)
        self.con.execute("INSTALL fts")
        self.con.execute("LOAD fts")

    def create_collection(self, collection: str):
        """
        Create a new collection (table) for storing documents.

        Args:
            collection: The name of the collection to create. This will be used
                        as the table name in DuckDB.

        Note:
            The collection will have the following schema:
            - id: VARCHAR PRIMARY KEY (document identifier)
            - content: TEXT (document content for full-text search)
            - metadata: BLOB (serialized JSON metadata)
        """
        self.con.execute(create_collection.render(collection=collection))

    def insert(
        self, collection: str, data: list[tuple[Id, Content, Metadata]], replace=False
    ):
        # Convert metadata to JSON string for storage in JSON column
        serialized_data: list[tuple[Id, Content, str]] = [
            (id, content, orjson.dumps(metadata, default=str).decode("utf-8"))
            for id, content, metadata in data
        ]
        insert_template = (
            insert_or_replace_into_collection if replace else insert_into_collection
        )
        self.con.executemany(
            insert_template.render(collection=collection), serialized_data
        )

        # Create or update FTS index to include new data (only content since metadata is JSON)
        # Only create FTS index if there's data in the table
        count_sql = count_collection.render(collection=collection)
        count = self.con.execute(count_sql).fetchone()[0]
        if count > 0:
            try:
                # Always recreate the FTS index after inserting data to ensure it's up to date
                fts_sql = create_fts_index.render(collection=collection)
                self.con.execute(fts_sql)
            except Exception as e:
                pass  # FTS index creation failed

    def collection_exists(self, collection: str) -> bool:
        """
        Check if a collection exists in the database

        Args:
            collection: The name of the collection to check

        Returns:
            True if the collection exists, False otherwise
        """
        result = self.con.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'main'
            AND table_name = ?
        """,
            [collection],
        ).fetchone()

        return result is not None

    def search(
        self,
        collection: str,
        query: str,
        limit: int = None,
        metadata_filter: dict | None = None,
    ):
        """
        Search using FTS in DuckDB

        Args:
            collection: The name of the collection to search in
            query: The search query string
            limit: Maximum number of results to return
            metadata_filter: Optional dictionary to filter results by metadata

        Returns:
            List of tuples containing (id, content, metadata, score) for matching documents
        """
        # Check if there's data in the collection before attempting FTS
        count_sql = count_collection.render(collection=collection)
        count = self.con.execute(count_sql).fetchone()[0]

        if count == 0:
            # If no data in collection, return empty list
            return []

        # Create FTS index if it doesn't exist - only for content field since metadata is BLOB
        fts_sql = create_fts_index.render(collection=collection)
        self.con.execute(fts_sql)

        # Build the query using json dot notation
        if metadata_filter:
            # Build WHERE clause for metadata filtering using JSONPath
            where_conditions = []
            params = []

            for key, value in metadata_filter.items():
                # Handle JSONPath dot notation by converting to DuckDB JSON access
                # For example: "nested.field" becomes metadata->'$.nested.field'
                # DuckDB supports the -> operator for JSON extraction
                json_path = f"$.{key}"

                # Extract the value from JSON and compare with the filter value
                if isinstance(value, bool):
                    # For boolean values, extract as string and compare
                    where_conditions.append(
                        f"json_extract_string(metadata, '{json_path}') = ?"
                    )
                    # Convert boolean to string representation for comparison
                    params.append(str(value).lower())
                elif isinstance(value, (int, float)):
                    # For numeric values, we need to extract and cast appropriately
                    where_conditions.append(
                        f"CAST(json_extract_string(metadata, '{json_path}') AS DOUBLE) = ?"
                    )
                    params.append(value)
                elif isinstance(value, str):
                    where_conditions.append(
                        f"json_extract_string(metadata, '{json_path}') = ?"
                    )
                    params.append(value)
                else:
                    # For other types, convert to string for comparison
                    where_conditions.append(
                        f"json_extract_string(metadata, '{json_path}') = ?"
                    )
                    params.append(str(value))

            # Construct the full query with FTS and metadata filtering
            # Use a subquery or CTE to avoid calling match_bm25 twice
            where_clause = " AND ".join(where_conditions)
            limit_clause = f"LIMIT {limit}" if limit else ""

            search_query = search_with_metadata_filter.render(
                collection=collection,
                where_clause=where_clause,
                limit_clause=limit_clause
            )

            results = self.con.execute(search_query, [query] + params).fetchall()
        else:
            # Original query without metadata filtering
            limit_clause = f"LIMIT {limit}" if limit else ""

            search_query = search_without_metadata_filter.render(
                collection=collection,
                limit_clause=limit_clause
            )

            results = self.con.execute(search_query, [query]).fetchall()

        # Deserialize the metadata from JSON string to Python object
        deserialized_results = []
        for id, content, metadata, score in results:
            # If metadata is a string (JSON string), parse it to Python object
            if isinstance(metadata, str):
                metadata = orjson.loads(metadata.encode("utf-8"))
            deserialized_results.append((id, content, metadata, score))

        return deserialized_results
```

# 2025

## Lightweight SHAP Implementation

_December 2025_

When performing SHAP, sometimes the requirements to install are overly onerous. This is a lightweight implementation of
Permutation SHAP including helper functions matching the style of `scikit-learn` interface.

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

I've been thinking about how to encode decision trees (on and off). This is a vibe-coded approach that tries to encode
the path of every instance based on what leaf of the decision tree(s) the instance lands on. The rough idea is that it
also gets 'weight' from every parent node (weighted based on distance to leaf) and is encoded by feature hashing, which
allows it to be aggregated by all trees. It somewhat works. I think this kind of approach may be useful for scenarios
where online learning with just a linear head is used, especially when feature engineering (from the trees) is done in
an online manner, for example via Mondrian Trees.

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

Using LLMs for doing arbitration or classification is nothing new. Generally I found that "smaller" local LLMs struggled
or led to overly optimistic results. I think 'smaller' models are finally performant enough (to some extent at least)
where coming up with binary outcomes in a structured manner is "good enough". In general as of writing things still
aren't good enough. This is reflected in things like the
[Goose Blogpost](https://block.github.io/goose/blog/2025/03/14/goose-ollama/) where 32gb of RAM is still generally
recommended. Here, I use the very good `gemma-3-12b` model to accomplish this -- I generally found models smaller/older
than this failed to properly perform structured outputs in a meaningful manner.

The below example is based on [Tricube Tales](https://www.drivethrurpg.com/en/product/294202/tricube-tales) which is a
TTRPG released under CC BY 3.0 license.

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

Sometimes you just want the convenience of numpy-like notation for slicing, and setting values.

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

Okay, so this isn't that "new" though sometimes its hard to find a "winner". This is one that I like for writing code
reviews.

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

What is interesting is how near (or far?) it is from curated offerings. For example this is the one from
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

Following on the general theme of using random-ness to infer or generate predictions, I've always been very curious
about isolation forests, more specifically given the innate measure of "similarity" or "distance" can we use this in the
supervised learning scenario via k-NN?

Below is some code which implements this completely using stdlib Python with no dependencies on `numpy`. Written with
heavy assistance from LLMs.

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

Commentary from the future: this approach is very similar to Mondrian Trees, in which the random partitions are induced
by a Mondrian Process. I should include a "fast approximation" to Mondrian processes and expand it to multi-dimensions
using standard lib in the future.

## Brownian Bridges

_February 2025_

I got LLM via prompting to recreate how it may have generated the simple kriging diagram on the wikipedia page. I
intentionally got it to do it via brownian bridges rather than the "typical" guassian process way simply because I was
interested in brownian bridges. I may have made changes for ease of implementation (I was thinking about re-implementing
it using stdlib or in another language as a close approximation), though the "idea" remains. This technique maybe useful
if one wants to do something better than sampling over a grid search.

To extend to multiple dimensions, we can make use of tree-parzen like approach to sample per region (chosen
heuristically) and presume univariate relationships (which was the approach taken in the original paper, though later
approaches used multivariate KDEs). For the purposes of a simplistic implementation, simplifying assumptions will
suffice. For example, it could be that the tree-structure restricts sampling to a fixed number of points per a region to
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

![output](data:image/png;base64,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c+fSrl07xo4da/L877//nvz8fCZPnkxJSQkffvghAwcO5PDhwwQGBla739mzZ/O///2Pe+65hwkTJpCens7HH3/MTTfdxIEDB6oU6JdTWFjIsGHD2Lt3L8uXL+e2226rcr3y8nKGDBlCnz59eP/996udqtbr9YwYMYI9e/YwadIkWrVqxW+//cbDDz981bEY2LZtGytXruTJJ5/E3d2djz76iLvuuoukpCR8fX0B8eMxdOhQgoODeeONN9DpdLz55pv4+/vXeD+14Z133kGtVjN9+nRyc3N57733eOCBB9i9e7dxnW+++YYnnniCXr16MW3aNE6fPs3tt9+Oj48P4eHhxvX0ej23334727Zt4/HHH6d169YcPnyY+fPnc/LkSVavXm2y73/++YdffvmFKVOm4Ofnd9WEKJ1OR0ZGBiAEx/Hjx43CuXfv3pXWv/vuu2nevDlvv/228cd/woQJfPfdd4wePZrnnnuO3bt3M2fOHI4fP86qVasA6Nu3L/Pnz+fo0aPG787WrVtRq9Vs3bqVp556yrgM4Kabbqr1e1oVH374IbfffjsPPPAAZWVl/PTTT9x999388ccflS7EanIsXY3777+fH3/8kXfeeQeVSkVGRgbr16/nhx9+qFIYGyxBgwYNYtKkScTGxvL555+zd+9etm/fjr29Pa+88gq5ubmcO3eO+fPnA1zRe7tx40aGDRtGkyZNmDlzJsXFxXz88cf07t2b/fv3Vzom7rnnHqKjo5kzZw779+/n66+/JiAggHffffeqr/fChQts2rSJ7777zvj658+fzyeffIKDg4PJ65wzZw4TJkygR48e5OXlsW/fPvbv32+M/t91110cPXqUqVOnEhUVRVpaGhs2bCApKemaE/vq+3z7xBNPcOHCBTZs2MAPP/xw1fEcPXqUvn374uHhwQsvvIC9vT0LFy6kf//+/Pvvv9xwww2Vxu/t7c3rr79OYmIiCxYsYMqUKfz888/X9H5ILABFIjEz69evVzQajaLRaJSePXsqL7zwgrJu3TqlrKys0rqRkZHKww8/bLy/aNEiBVCGDBmi6PV64/KePXsqKpVKmThxonFZeXm5EhYWpvTr18+4LCEhQQEUZ2dn5dy5c8blu3fvVgDlmWeeMS57/fXXlYpfmcTEREWj0SizZ882GePhw4cVOzu7SssvxzD2yMhIxd7eXlm9enW16z788MMKoLz00ktVPhYZGWm8v2LFCgVQFixYYFym0+mUgQMHKoCyaNGial+ToigKoDg4OCjx8fHGZTExMQqgfPzxx8ZlI0aMUFxcXJTz588bl8XFxSl2dnaVtlkVl3+WBvr162fyGW3atEkBlNatWyulpaXG5R9++KECKIcPH1YURVHKysqUgIAApVOnTibrffnllwpgss0ffvhBUavVytatW032/cUXXyiAsn37dpP3Q61WK0ePHr3qazKMH6h0a926tXL69GmTdQ3v//3332+y/ODBgwqgTJgwwWT59OnTFUD5559/FEVRlLS0NAVQPvvsM0VRFCUnJ0dRq9XK3XffrQQGBhqf99RTTyk+Pj7G70hN31NFqXx8KYqiFBUVmdwvKytT2rVrpwwcONBkeU2PpaowfDfnzp2rHDlyRAGMn9enn36quLm5KYWFhcrDDz+suLq6Gp+XlpamODg4KLfccoui0+mMyz/55BMFUL799lvjsltvvbXSa6u474rflU6dOikBAQFKZmamyWtRq9XK2LFjjcsMn+kjjzxiss077rhD8fX1veJrNvD+++8rzs7OSl5enqIoinLy5EkFUFatWmWyXseOHZVbb7212u1kZ2cb38MrASivv/56peXmPN9Onjy52vPI5eMdNWqU4uDgoJw6dcq47MKFC4q7u7ty0003VRr/oEGDTMb/zDPPKBqNRsnJyan6DZJYPNJmIDE7gwcPZufOndx+++3ExMTw3nvvMWTIEEJDQ/n9999rtI1HH33UZBrqhhtuQFEUHn30UeMyjUZDt27dOH36dKXnjxo1itDQUOP9Hj16cMMNN7BmzZpq97ly5Ur0ej333HMPGRkZxltQUBDNmzdn06ZNNRp7amoqTk5OJpHD6pg0adJV11m7di329vY89thjxmVqtZrJkyfXaDwAgwYNomnTpsb7HTp0wMPDw/je6XQ6Nm7cyKhRowgJCTGu16xZsxolrV0L48ePN4lK9e3bF8A4pn379pGWlsbEiRNN1hs3bhyenp4m2/r1119p3bo1rVq1MvnsBg4cCFDps+vXrx9t2rSp8VijoqLYsGEDGzZs4K+//mLBggXk5uYybNiwKqczJ06caHLfcNw9++yzJsufe+45AP78808A/P39adWqFVu2bAFg+/btaDQann/+eVJTU4mLiwNEZLZPnz6V7B9X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+6/JOF566SXeeecdjh8/TocOHSqN88MPP2TevHm8//77HDp0iCFDhnD77bebiG3Ddp5++mmOHz/OkCFDqn3db731FmPHjuXgwYO0atWKMWPG8MQTTzBjxgz27duHoigm1pStW7cyduxYnn76aY4dO8bChQtZvHgxs2fPNq6jVqv56KOPOHr0KN999x3//PMPL7zwgsl+i4qKeP/99/nhhx/YsmULSUlJTJ8+/aqfk0QisU7y84U39sgRcc4ODLT+zlvWiJ2diNCePi1Kn1lWKyhJQyDt6DaIoij8/fffrFu3jqlTpxIXF8fvv//O9u3b6dWrFwBLliwhPDyc1atXc/fdd5OUlMRdd91F+/btAWjSpIlxez4+PgAEBATgdTFttLS0lLfffpuNGzcaE/CaNGnCtm3bWLhwIf369TM+/8033zSJuF/O+++/z4svvsh9990HwLvvvsumTZtYsGABn376qXG9adOmceedd1719Y8fP5577rkHgBdffJGePXvyv//9zyiAn376acaPH29c/4033uCll17i4YcfNr6Ot956ixdeeIHXX3/duG8DUVFRzJo1i4kTJ/LZZ58Zl2u1Wr744guaNm0KwJQpU3jzzTevOl6JRGJdKIoot3X8uBC0oaEyucvcODqKcmcnTohGFOHh5h6RpCGRX7+r4OIiCjWba9+14Y8//sDNzQ2tVoter2fMmDHMnDmTv//+Gzs7O2644Qbjur6+vrRs2ZLjx48D8NRTTzFp0iTWr1/PoEGDuOuuu6qMohqIj4+nqKiokkgtKyujc+fOJsu6detW7Xby8vK4cOECvXv3Nlneu3dvYmJiarydilQct6HJhkGkG5aVlJSQl5eHh4cHMTExbN++3SQSq9PpKCkpoaioCBcXFzZu3MicOXM4ceIEeXl5lJeXmzwO4OLiYhSyAMHBwaSlpdVozBKJxDooK4P4eBEBdHAQSUgyGmsZuLkJ28fRo0LcBgSYe0SShkKK2augUoGrq7lHUTMGDBjA559/joODAyEhIdjVIlQwYcIEhgwZwp9//sn69euZM2cO8+bNY+rUqVWub/CL/vnnn4SGhpo8dnldX9c6egNrup2KiWGqi78yVS3TX2yGXlBQwBtvvFFl1NfJyYnExERuu+02Jk2axOzZs/Hx8WHbtm08+uijlJWVGcXs5QlpKpUKRc53SSQ2Q06O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)
