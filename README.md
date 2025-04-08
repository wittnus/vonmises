# vonmises
Von-mises fisher distribution for k-means clustering based attention layer.
The objective of this project is to achieve nlogn complexity in token length for attention layers.

## Core Ideas

### Standard Attention Complexity

In transformer models, the standard attention mechanism computes:

```
Attention(Q, K, V) = softmax(QK^T / √d) · V
```

For sequence length n and embedding dimension d:
- Computing QK^T requires O(n²d) operations
- Applying softmax and multiplying by V requires O(n²d) operations
- Total complexity: O(n²d)

This quadratic complexity in sequence length becomes prohibitively expensive as n grows, limiting the context length of transformer models.

### Clustering-Based Attention Complexity

This project implements attention approximation using von Mises-Fisher clustering:

1. **Preprocessing** (done once for all queries):
   - Organize keys into a hierarchical binary tree with k = 2^levels clusters
   - Complexity: O(nd log k)

2. **Query Processing** (per query):
   - Compute expected attention mass for each cluster: O(kd)
   - Select top-b clusters (beam width): O(k log b)
   - Compute exact attention for selected clusters: O(bd * n/k)
   - Approximate remaining clusters: O(kd)
   - Total per-query complexity: O(kd + n/k * bd) = O((k + nb/k)d)

When k ≈ √n (optimal clustering):
- Per-query complexity: O(√n·d)
- For all n queries: O(n^1.5·d)

This is a significant improvement over the standard O(n²d) complexity, especially for large sequence lengths.

### Achieving O(n log n) Complexity via Hierarchical Beam Search

The hierarchical cluster structure can be further exploited to achieve O(n log n) complexity through hierarchical beam search:

1. **Hierarchical Traversal** instead of flat cluster scoring:
   - Start at the root of the binary tree
   - At each level, evaluate only the most promising b nodes (beam width)
   - Traverse down the tree, maintaining only the top-b candidates at each level
   - With log k levels and b nodes per level: O(b log k · d)

2. **Complexity Analysis**:
   - Preprocessing: O(nd log k) (unchanged)
   - Per-query traversal: O(b log k · d)
   - Exact computation for selected leaf clusters: O(b · n/k · d)
   - When k = n/log n and b = O(log n):
     - Per-query: O(log n · log(n/log n) · d + log n · log n · d) = O(log²n · d)
     - For all n queries: O(n log²n · d)

This approach achieves near-O(n log n) complexity while maintaining accuracy by adaptively focusing computation on the most relevant parts of the hierarchical tree. With careful tuning of the beam width and pruning strategies, the complexity can be further reduced to O(n log n · d).

The hierarchical structure enables logarithmic-time decision making at each step of the traversal, resulting in significant computational savings compared to flat clustering approaches.

Additionally, the preprocessing step (O(nd log k)) only needs to be performed once for a set of key-value pairs, making this approach particularly efficient for scenarios with many queries against the same key-value set.

The von Mises-Fisher distribution provides a probabilistic framework for clustering directional data on the unit hypersphere, which is ideal for key vectors in attention mechanisms.

### Cumulant Generating Functions and Log Expected Query Mass

The log expected attention mass for a query q over a cluster of keys k drawn from a distribution P(k) is directly related to the cumulant generating function (CGF):

```
Log expected mass = log(E[exp(q·k)]) = K(q)
```

Where K(q) is the CGF of the key distribution evaluated at point q. This relationship enables efficient approximation of attention scores using statistical properties of clustered keys:

#### Von Mises-Fisher Distribution
For keys distributed according to vMF with mean direction μ and concentration κ:
- The CGF can be expressed using modified Bessel functions
- For dimension d, K(q) = log(I_{d/2}(||κμ+q||)/I_{d/2}(κ))
- For high dimensions and large κ, saddle-point approximation gives:
  - K(q) ≈ (d/2-1)log(κ/||κμ+q||) + (||κμ+q||-κ) + (1/2)log(κ/||κμ+q||)
- In the simple first-order approximation: K(q) ≈ R·(q·μ) where R = ||μ||

#### Approximating Clusters with Spherical Shells
When a cluster is better approximated as a spherical shell (keys distributed on a sphere of radius R centered at μ):
- For sphere centered at origin: K(q) = log(I_{d/2-1}(R·||q||)) - (d/2-1)log(R·||q||) + constant
- For sphere centered at μ, the CGF becomes a function of the offset: K(q) = log(E[exp(q·(μ+Rz))]) where z is uniform on unit sphere
- To first order: K(q) ≈ q·μ
- For a uniform spherical shell, the covariance is isotropic: Σ = (R²/d)·I
- To second order: K(q) ≈ q·μ + (R²/2d)·||q||²
- This captures both the mean direction effect (q·μ) and the isotropic variance (R²/2d)·||q||²
- In high dimensions, the variance term becomes negligible compared to the mean term

Note: For a vMF distribution (which is different from a uniform spherical shell), the covariance has the anisotropic structure Σ = (1-R²)/d·(I-μμᵀ/||μ||²), where R = ||μ|| is related to the concentration parameter.

#### Approximating Clusters with Balls
When a cluster is better approximated as a ball (keys uniformly distributed in a ball of radius R centered at μ):
- For ball centered at origin: K(q) = log(d·I_{d/2}(R·||q||)·(2/(R·||q||))^{d/2}) where I_v is the modified Bessel function
- For ball centered at μ: K(q) = log(E[exp(q·(μ+Rz))]) where z is uniform in unit ball
- To first order: K(q) ≈ q·μ
- To second order: K(q) ≈ q·μ + R²||q||²/2(d+2)
- The variance term R²/2(d+2) is smaller than for spherical shells due to lower average distance from center
- As d increases, both approximations converge to K(q) ≈ q·μ, with the variance terms becoming negligible

These approximations provide theoretical justification for the attention mass estimation functions in the library. The relationship between cluster geometry (R, μ) and the resulting cumulant function explains why the simple first-order approximation (q·μ·R) is often effective, especially in high-dimensional spaces where concentration effects dominate.

### Joint Distribution of Keys and Values: Error Analysis

The accuracy of attention approximation depends not only on modeling the key distribution but also on the joint distribution of keys (k) and values (v). The expected error when approximating the output contribution of a cluster can be analyzed through the joint cumulant generating function (CGF):

```
K(q,t) = log E[exp(q·k + t·v)]
```

Where q is the query vector and t is an auxiliary variable for analyzing value contributions. The true attention output for a query is related to the partial derivative of this joint CGF:

```
E[v·exp(q·k)]/E[exp(q·k)] = ∇_t K(q,0)|_{t=0}
```

#### Approximation Error Analysis

When approximating a cluster's contribution using only its average value μ_v:

1. **First-Order Approximation (Using μ_v)**
   - For jointly Gaussian distributions, the error is exactly: Σ_vk·q
   - Where Σ_vk is the cross-covariance matrix between keys and values
   - Error magnitude: ||Σ_vk·q||
   - Perfect only when keys and values are independent (Σ_vk = 0)

2. **Second-Order Approximation (Using μ_v + Σ_vk·q)**
   - Captures linear correlation between keys and values
   - Exact for jointly Gaussian distributions
   - Equivalently: using conditional expectation E[v|q·k]

3. **Higher-Order Errors**
   - Arise from non-Gaussian aspects of joint distribution
   - Involve third and higher mixed cumulants between keys and values
   - Represent nonlinear dependencies not captured by covariance structure

The practical implication is that clustering strategies should consider not just key similarity but also the correlation structure between keys and values within clusters. Even with perfect estimation of attention weights, output errors can be significant if the key-value correlation structure is not properly accounted for.

### Outer Product Clustering: A Unified Approach

An alternative approach that directly addresses the joint key-value distribution challenge is to cluster the outer products kv^T rather than clustering keys and values separately:

#### Theoretical Framework

1. **Direct Optimization of Attention Contribution**
   - For any query q, its contribution from a key-value pair is (q·k)v
   - This can be rewritten as q^T·(kv^T)·e, where e is a standard basis vector
   - The outer product kv^T (d×d matrix) fully captures the attention contribution pattern

2. **Clustering in Outer Product Space**
   - Represent each token by its kv^T outer product
   - Define distance metric: ||k₁v₁^T - k₂v₂^T||_F (Frobenius norm)
   - Cluster these outer products directly
   - Use cluster centroids as representative kv^T matrices

3. **Error Analysis**
   - First-order approximation using cluster centroids becomes exact if outer products are identical within clusters
   - No separate modeling of key-value correlations needed
   - Error depends solely on the variance of outer products within clusters

#### Complexity Considerations

This approach introduces significant computational trade-offs:

1. **Storage Requirements**
   - Standard approach: O(nd) for storing n keys and values of dimension d
   - Outer product approach: O(nd²) for storing n outer products (d×d matrices)
   - Can be mitigated using low-rank approximations: O(ndr) where r << d

2. **Computational Complexity**
   - Distance calculations: O(d²) instead of O(d)
   - Preprocessing: O(nd² log k) instead of O(nd log k)
   - Query processing: potentially more efficient as it directly optimizes for what matters

3. **Practical Optimizations**
   - Use structured low-rank approximations of outer products
   - Hierarchical clustering with SVD-based dimensionality reduction
   - Progressive refinement focusing on high-error regions

This unified approach elegantly addresses the joint distribution modeling challenge by directly optimizing the structures that contribute to the final attention output. While computationally more intensive in preprocessing, it could potentially achieve higher accuracy with fewer clusters by focusing on the exact patterns that determine attention output quality.

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
