# vonmises
Von-mises fisher distribution for k-means clustering based attention layer.
The objective of this project is to achieve nlogn complexity in token length for attention layers.

## Library Overview

The `lib.py` file implements core functionality for efficient attention approximation:

### Clustering
- `cluster_spherical`: Hierarchical clustering on unit sphere with weighted balanced k-means
- `cluster`: Simplified hierarchical clustering without weights

### Von Mises-Fisher Distribution
- `vmf_concentration`: Fast approximation of concentration parameter (κ) from centroid
- `vmf_concentration_exact`: Newton's method for exact κ calculation

### Attention Approximation
- `log_expected_query_mass`: Saddle-point approximation for expected attention
- `simple_log_expected_query_mass`: First-order approximation (faster but less accurate)

### Attention Implementations
- Standard: `approx_qkv_attention`, `batched_approx_qkv_attention`
- Simplified: `simple_approx_qkv_attention`, `batched_simple_approx_qkv_attention`
- Top-k selection: `topk_approx_qkv_attention`, `batched_topk_approx_qkv_attention`
- Stochastic sampling: `stochastic_approx_qkv_attention`, `batched_stochastic_approx_qkv_attention`
- Adaptive weighting: `adaptive_stochastic_approx_qkv_attention`, `batched_adaptive_stochastic_approx_qkv_attention`

All implementations use JAX for GPU acceleration with JIT compilation and vectorization.

## Detailed Function Documentation

### Clustering Functions

#### `cluster_spherical(xs, levels=3, vs=None)`
Performs hierarchical balanced clustering of vectors on a unit sphere. This function:
- Recursively splits data into 2^levels clusters using balanced k-means
- Normalizes vectors to unit sphere but uses original magnitudes as weights
- Creates a binary tree of cluster centers with log-weights for each node
- Optionally clusters additional vectors (`vs`) using the same partitioning
- Returns cluster centers, log-weights, and clustered data
- Used for approximating attention by organizing keys into a hierarchical structure

#### `cluster(xs, levels=3)`
Simplified version of hierarchical clustering without weights. This function:
- Performs recursive balanced k-means clustering into 2^levels clusters
- Creates a binary tree of cluster centers
- Returns cluster centers and clustered data points
- Maintains equal number of points per cluster

### Von Mises-Fisher Distribution Functions

#### `vmf_concentration(centroid)`
Estimates the concentration parameter (κ) for a von Mises-Fisher distribution:
- Uses dimension-specific formulas with correction factors
- Handles various regimes of centroid magnitude (R) with specialized approximations
- Optimized for transformer dimensions (~64)
- The concentration parameter determines how tightly clustered points are around the mean direction
- Higher κ indicates tighter clustering

#### `vmf_concentration_exact(centroid, max_iter=20, tol=1e-6)`
Computes the exact concentration parameter using numerical methods:
- Employs Newton's method to iteratively solve the implicit equation
- More accurate but computationally more expensive than the approximation
- Uses different formulas for small, medium, and large κ values
- Returns exact κ value with high precision

### Attention Approximation Functions

#### `simple_log_expected_query_mass(query, centroid)`
First-order approximation for the expected attention weight:
- Simply returns the dot product between query and centroid
- Fast but less accurate than saddle-point approximation
- Based on first-order term in Taylor expansion of exponential

#### `log_expected_query_mass(query, centroid)`
Computes the log expected attention weight using saddle-point approximation:
- Estimates concentration κ from the centroid
- Computes qappa = norm(query + mu * kappa)
- Uses different approximations for near-uniform vs concentrated cases
- Includes correction term for improved accuracy
- Returns negative log of the expected attention mass

### Attention Implementation Functions

#### `approx_qkv_attention(query, centers, log_weights, clustered_keys, clustered_values, weighted_means=None, beam_width=4, pruning_threshold=-10.0)`
Standard approximate attention algorithm:
- Calculates scores for all leaf clusters
- Selects top-k clusters based on expected attention mass
- Computes exact attention for selected clusters
- Optionally approximates other clusters using weighted means
- Returns approximate attention output vector

#### `batched_approx_qkv_attention(queries, centers, log_weights, clustered_keys, clustered_values, weighted_means=None, beam_width=4, pruning_threshold=-10.0)`
Batch version of the standard approximate attention:
- Processes each query in the batch individually
- Supports progress tracking with tqdm if available
- Returns batch of approximate attention outputs

#### `simple_approx_qkv_attention(query, centers, log_weights, clustered_keys, clustered_values, weighted_means)`
Simplified approximate attention using only the best cluster:
- JAX-compatible with no Python control flow
- Finds the best leaf node based on expected attention mass
- Computes exact attention for the best cluster
- Approximates all other nodes using weighted means
- Fully JIT-compilable for maximum performance

#### `batched_simple_approx_qkv_attention(queries, centers, log_weights, clustered_keys, clustered_values, weighted_means)`
Vectorized version of simple approximate attention:
- Uses JAX's vmap to process queries in parallel
- Fully JIT-compilable for batch processing
- Maintains the simplicity and speed of the single-query version

#### `topk_approx_qkv_attention(query, centers, log_weights, clustered_keys, clustered_values, weighted_means, k=2, threshold=-10.0)`
Extended approximation using top-k clusters:
- Finds k best clusters based on expected attention mass
- Computes exact attention for these k clusters
- Uses residual approximation for remaining clusters
- Includes JIT compilation with static k parameter
- Balances accuracy and speed by focusing computation on important clusters

#### `batched_topk_approx_qkv_attention(queries, centers, log_weights, clustered_keys, clustered_values, weighted_means, k=2, threshold=-10.0)`
Vectorized version of top-k approximate attention:
- Uses JAX's vmap to process queries in parallel
- Maintains the same k parameter across all queries
- Fully JIT-compilable for efficient batch processing

#### `stochastic_approx_qkv_attention(query, centers, log_weights, clustered_keys, clustered_values, weighted_means, temperature=1.0, key=None)`
Stochastic attention approximation:
- Deterministically selects the best cluster
- Stochastically samples a second cluster based on temperature-controlled softmax
- Balances exploitation (best cluster) and exploration (stochastic sampling)
- Useful for improving gradient estimates during training
- Temperature parameter controls randomness in selection

#### `batched_stochastic_approx_qkv_attention(queries, centers, log_weights, clustered_keys, clustered_values, weighted_means, temperature=1.0, key=None)`
Vectorized version of stochastic approximate attention:
- Creates unique random keys for each query in the batch
- Uses JAX's vmap to process queries in parallel
- Maintains the stochastic properties across the batch

#### `adaptive_stochastic_approx_qkv_attention(query, centers, log_weights, clustered_keys, clustered_values, weighted_means, temperature=1.0, key=None)`
Enhanced stochastic attention with adaptive residual weighting:
- Selects best cluster deterministically and second cluster stochastically
- Estimates attention mass captured by selected clusters
- Applies adaptive residual weighting based on mass ratio
- Reduces bias by down-weighting residual when selected clusters capture more mass
- Improves accuracy while maintaining stochastic benefits

#### `batched_adaptive_stochastic_approx_qkv_attention(queries, centers, log_weights, clustered_keys, clustered_values, weighted_means, temperature=1.0, key=None)`
Vectorized version of adaptive stochastic approximate attention:
- Creates unique random keys for each query in the batch
- Uses JAX's vmap to process queries in parallel
- Maintains the adaptive stochastic properties across the batch
