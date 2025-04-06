import jax
from jax import numpy as jnp
from chex import Array
from typing import Callable, Optional, Tuple


def cluster_spherical(xs: Array, levels: int = 3) -> Tuple[Array, Array]:
    """Cluster vectors on a sphere using their normalized directions.
    
    Recursively cluster the data into 2^levels clusters by splitting in
    two each time with balanced k-means. The vectors are normalized to the
    unit sphere, but their original magnitudes are used as weights when
    computing cluster centers. Points (not weights) are evenly distributed.
    
    Args:
        xs: Input vectors with shape (n_samples, n_features)
        levels: Number of recursive splitting levels, resulting in 2^levels clusters
        
    Returns:
        Tuple containing:
            - centers: Array of cluster centers, organized in binary tree order
            - clustered_data: Original data organized by cluster
    """
    n_clusters = 2**levels
    n_samples = xs.shape[0]
    n_per_cluster = n_samples // n_clusters
    
    # Compute magnitudes (weights) and normalize vectors
    magnitudes = jnp.sqrt(jnp.sum(xs**2, axis=1, keepdims=True))
    
    # Handle zero vectors by replacing with small values
    safe_magnitudes = jnp.maximum(magnitudes, 1e-10)
    normalized_xs = xs / safe_magnitudes
    
    # Store exponential of magnitudes as weights
    weights = jnp.exp(magnitudes.squeeze())
    
    # Initialize centers array
    all_centers = jnp.zeros((2*n_clusters-1, xs.shape[1]))
    
    # Define weighted k-means function using JAX
    def kmeans_balanced(data, data_weights, key):
        n_data = data.shape[0]
        # Random initialization of centroids
        key, subkey = jax.random.split(key)
        idx = jax.random.choice(subkey, n_data, shape=(2,), replace=False)
        centroids = data[idx]
        
        # K-means iterations
        def step(_, centroids):
            # Compute distances to centroids
            dists = jnp.sum((data[:, None, :] - centroids[None, :, :])**2, axis=-1)
            # Calculate distance differences
            dist_diff = dists[:, 0] - dists[:, 1]
            # Sort indices by distance difference
            sorted_idx = jnp.argsort(dist_diff)
            # Balanced assignment: half to each cluster (by count, not weight)
            half = n_data // 2
            assign_0 = sorted_idx[:half]
            assign_1 = sorted_idx[half:]
            
            # Update centroids using weighted means
            weights_0 = data_weights[assign_0]
            weights_1 = data_weights[assign_1]
            
            # Compute weighted means
            sum_weights_0 = jnp.sum(weights_0) + 1e-10  # Avoid division by zero
            sum_weights_1 = jnp.sum(weights_1) + 1e-10
            
            weighted_sum_0 = jnp.sum(data[assign_0] * weights_0[:, None], axis=0)
            weighted_sum_1 = jnp.sum(data[assign_1] * weights_1[:, None], axis=0)
            
            new_centroid_0 = weighted_sum_0 / sum_weights_0
            new_centroid_1 = weighted_sum_1 / sum_weights_1
            
            # Do not normalize - this behaves like implicit vonmises-fisher clustering
            #new_centroid_0 = new_centroid_0 / jnp.maximum(jnp.sqrt(jnp.sum(new_centroid_0**2)), 1e-10)
            #new_centroid_1 = new_centroid_1 / jnp.maximum(jnp.sqrt(jnp.sum(new_centroid_1**2)), 1e-10)
            
            return jnp.stack([new_centroid_0, new_centroid_1])
        
        # Run 10 iterations of k-means (typically sufficient for convergence)
        centroids = jax.lax.fori_loop(0, 10, step, centroids)
        
        # Final assignments
        dists = jnp.sum((data[:, None, :] - centroids[None, :, :])**2, axis=-1)
        dist_diff = dists[:, 0] - dists[:, 1]
        sorted_idx = jnp.argsort(dist_diff)
        half = n_data // 2
        
        # Return centroids and indices for both clusters
        return centroids, sorted_idx[:half], sorted_idx[half:]
    
    # Recursive clustering function with non-local state
    def recursive_cluster(data, data_weights, data_original, indices, node_idx, depth, key, centers):
        key, subkey = jax.random.split(key)
        
        if depth == levels:
            # Leaf node: compute weighted center and return data
            # For the center, use weighted mean with original magnitudes
            sum_weights = jnp.sum(data_weights) + 1e-10
            weighted_sum = jnp.sum(data * data_weights[:, None], axis=0)
            center = weighted_sum / sum_weights
            
            # Do not normalize - keep center inside the sphere
            
            centers = centers.at[node_idx].set(center)
            # Return the original data points (not normalized)
            return centers, data_original, indices
        
        # Split using balanced k-means on normalized data
        centroids, left_idx, right_idx = kmeans_balanced(data, data_weights, subkey)
        
        # Update internal node center
        # Use average of centroids, do not normalize
        center = (centroids[0] + centroids[1]) / 2
        centers = centers.at[node_idx].set(center)
        
        # Process left subtree
        left_key, right_key = jax.random.split(subkey)
        centers, left_data, left_indices = recursive_cluster(
            data[left_idx], data_weights[left_idx], data_original[left_idx], 
            indices[left_idx], 2*node_idx+1, depth+1, left_key, centers)
        
        # Process right subtree
        centers, right_data, right_indices = recursive_cluster(
            data[right_idx], data_weights[right_idx], data_original[right_idx],
            indices[right_idx], 2*node_idx+2, depth+1, right_key, centers)
        
        return centers, jnp.vstack([left_data, right_data]), jnp.concatenate([left_indices, right_indices])
    
    # Create array of original indices to track point order
    original_indices = jnp.arange(n_samples)
    
    # Start recursive clustering from root (node 0)
    key = jax.random.PRNGKey(0)
    all_centers, clustered_xs, sorted_indices = recursive_cluster(
        normalized_xs, weights, xs, original_indices, 0, 0, key, all_centers)
    
    # Reshape clustered data to desired output format with even distribution
    clustered_xs = clustered_xs.reshape((n_clusters, n_per_cluster, -1))
    
    return all_centers, clustered_xs


def cluster(xs: Array, levels: int = 3) -> Tuple[Array, Array]:
    """recursively cluster the data into 2^levels clusters by splitting in
    two each time with balanced k-means. return the cluster centers, organized
    such that the last half of the centers are the lowest level clusters, and
    so on up recursively. return also the original data points, organized
    in a 2^levels x n array, where n is the number of points per cluster.
    The first dimension is ordered according to the lowest level clusters.
    It is assumed that the number of points is a power of two.
    """
    n_clusters = 2**levels
    n_samples = xs.shape[0]
    n_per_cluster = n_samples // n_clusters
    
    # Initialize centers array
    all_centers = jnp.zeros((2*n_clusters-1, xs.shape[1]))
    
    # Define k-means function using JAX
    def kmeans_balanced(data, key):
        n_data = data.shape[0]
        # Random initialization of centroids
        key, subkey = jax.random.split(key)
        idx = jax.random.choice(subkey, n_data, shape=(2,), replace=False)
        centroids = data[idx]
        
        # K-means iterations
        def step(_, centroids):
            # Compute distances to centroids
            dists = jnp.sum((data[:, None, :] - centroids[None, :, :])**2, axis=-1)
            # Calculate distance differences
            dist_diff = dists[:, 0] - dists[:, 1]
            # Sort indices by distance difference
            sorted_idx = jnp.argsort(dist_diff)
            # Balanced assignment: half to each cluster
            half = n_data // 2
            assign_0 = sorted_idx[:half]
            assign_1 = sorted_idx[half:]
            # Update centroids
            new_centroid_0 = jnp.mean(data[assign_0], axis=0)
            new_centroid_1 = jnp.mean(data[assign_1], axis=0)
            return jnp.stack([new_centroid_0, new_centroid_1])
        
        # Run 10 iterations of k-means (typically sufficient for convergence)
        centroids = jax.lax.fori_loop(0, 10, step, centroids)
        
        # Final assignments
        dists = jnp.sum((data[:, None, :] - centroids[None, :, :])**2, axis=-1)
        dist_diff = dists[:, 0] - dists[:, 1]
        sorted_idx = jnp.argsort(dist_diff)
        half = n_data // 2
        
        # Return centroids and indices for both clusters
        return centroids, sorted_idx[:half], sorted_idx[half:]
    
    # Recursive clustering function with non-local state
    def recursive_cluster(data, node_idx, depth, key, centers):
        key, subkey = jax.random.split(key)
        
        if depth == levels:
            # Leaf node: store center and return data
            center = jnp.mean(data, axis=0)
            centers = centers.at[node_idx].set(center)
            return centers, data
        
        # Split using balanced k-means
        centroids, left_idx, right_idx = kmeans_balanced(data, subkey)
        
        # Update internal node center
        centers = centers.at[node_idx].set(jnp.mean(centroids, axis=0))
        
        # Process left subtree
        left_key, right_key = jax.random.split(subkey)
        centers, left_data = recursive_cluster(
            data[left_idx], 2*node_idx+1, depth+1, left_key, centers)
        
        # Process right subtree
        centers, right_data = recursive_cluster(
            data[right_idx], 2*node_idx+2, depth+1, right_key, centers)
        
        return centers, jnp.vstack([left_data, right_data])
    
    # Start recursive clustering from root (node 0)
    key = jax.random.PRNGKey(0)
    all_centers, clustered_xs = recursive_cluster(xs, 0, 0, key, all_centers)
    
    # Reshape clustered data to desired output format
    clustered_xs = clustered_xs.reshape((n_clusters, n_per_cluster, -1))
    
    return all_centers, clustered_xs

def vmf_concentration(centroid: Array) -> Array:
    """Compute the concentration parameter (kappa) for a von Mises-Fisher distribution
    given the unnormalized centroid.
    """
    R = jnp.linalg.norm(centroid)
    d = centroid.shape[0]
    kappa = R * (d - R**2) / (1 - R**2)
    return kappa

def log_expected_query_mass(query, centroid):
    """Compute the log expected attention weight (using exponential of dot product)
    where the key is vmf distributed with given centroid.
    
    Uses saddle-point approximation with correction term, accurate for moderate to large
    values of kappa and dimension.
    """
    kappa = vmf_concentration(centroid)
    mu = centroid / jnp.linalg.norm(centroid)
    d = query.shape[0]
    v = d/2 - 1  # Order of the Bessel function
    qappa = jnp.linalg.norm(query + mu * kappa)
    
    # For very small kappa (near-uniform distribution), use accurate approximation
    def near_uniform_case():
        # For small kappa, I_v(kappa) ≈ (kappa/2)^v / Gamma(v+1)
        lnZk = v * jnp.log(kappa/2) - jax.lax.lgamma(v+1) - kappa
        lnZq = v * jnp.log(qappa/2) - jax.lax.lgamma(v+1) - qappa
        return lnZk - lnZq
    
    # For moderate to large kappa, use saddle-point approximation with correction
    def saddle_point_case():
        # Main saddle-point approximation term
        main_term = v * jnp.log(kappa/qappa) + (qappa - kappa)
        # Correction term from the next order in the asymptotic expansion
        correction_term = 0.5 * jnp.log(kappa/qappa)
        return main_term + correction_term
    
    # Calculate a balanced cutoff that works well across dimensions
    # This formula provides good accuracy from d=3 to d=512+
    cutoff = jnp.maximum(v + 10, v * 1.2)
    
    # Switch between approximations based on the minimum of kappa and qappa
    return jax.lax.cond(
        jnp.minimum(kappa, qappa) < cutoff,
        near_uniform_case,
        saddle_point_case
    )

def approx_qkv_attention(queries: Array, centroids: Array, keys: Array, values: Array):
    """Given queries and centroids from hierarchical clustering, compute the approximate
    attention weights and output values. keys and values are organized as in the output
    of cluster_spherical.
    """
    raise NotImplementedError("This function is not implemented yet.")
