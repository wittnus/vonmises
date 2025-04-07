import jax
import jax.numpy as jnp
from jax.random import PRNGKey
from chex import Array
import numpy as np
from functools import partial
from typing import Tuple, List, Dict
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
from lib import vmf_concentration, vmf_concentration_exact, log_expected_query_mass, cluster_spherical


def generate_unit_vectors(key: PRNGKey, n_samples: int = 2**12, dim: int = 64) -> Array:
    """Generate random unit vectors uniformly distributed on the unit sphere in R^dim.
    
    Args:
        key: JAX random key
        n_samples: Number of vectors to generate (default: 2^12 = 4096)
        dim: Dimension of the vectors (default: 64)
        
    Returns:
        Array of shape (n_samples, dim) containing unit vectors
    """
    # Generate normally distributed vectors
    data = jax.random.normal(key, (n_samples, dim))
    
    # Normalize to unit length
    norms = jnp.sqrt(jnp.sum(data**2, axis=1, keepdims=True))
    unit_vectors = data / norms
    
    return unit_vectors


def balanced_kmeans_scipy(data: np.ndarray, k: int, max_iters: int = 20, seed: int = 42) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Implementation of balanced k-means using scipy/numpy.
    Each cluster will have exactly the same number of points.
    
    Args:
        data: Input data of shape (n_samples, n_features)
        k: Number of clusters
        max_iters: Maximum number of iterations
        seed: Random seed for centroid initialization
        
    Returns:
        centroids: Array of cluster centroids
        clusters: List of arrays with indices of data points in each cluster
    """
    n_samples, n_features = data.shape
    
    # Check if the number of samples is divisible by k
    if n_samples % k != 0:
        raise ValueError(f"Number of samples ({n_samples}) must be divisible by k ({k})")
    
    points_per_cluster = n_samples // k
    
    # Initialize centroids randomly from the data
    np.random.seed(seed)
    indices = np.random.choice(n_samples, k, replace=False)
    centroids = data[indices].copy()
    
    # Run iterations
    for iteration in range(max_iters):
        # Calculate distances between each point and each centroid
        # Using sklearn's euclidean_distances for efficiency
        distances = euclidean_distances(data, centroids)
        
        # Sort data points by their distance to each centroid
        dist_diff = np.zeros((n_samples, k))
        for i in range(k):
            # For each centroid, compute how much closer/farther it is to each point
            # compared to all other centroids
            other_centroids_mask = np.ones(k, dtype=bool)
            other_centroids_mask[i] = False
            min_other_dist = np.min(distances[:, other_centroids_mask], axis=1)
            dist_diff[:, i] = distances[:, i] - min_other_dist
        
        # Initialize clusters
        clusters = [[] for _ in range(k)]
        
        # Assign points to clusters in a balanced way
        # Sort indices by their preference for each cluster
        all_indices = np.arange(n_samples)
        unassigned = np.ones(n_samples, dtype=bool)
        
        for i in range(k):
            # Get unassigned points
            remaining_indices = all_indices[unassigned]
            remaining_diffs = dist_diff[unassigned, i]
            
            # If this is the last centroid, assign all remaining points
            if i == k - 1:
                clusters[i] = remaining_indices
                break
                
            # Otherwise, sort by preference and take the top points_per_cluster
            sorted_indices = remaining_indices[np.argsort(remaining_diffs)]
            clusters[i] = sorted_indices[:points_per_cluster]
            
            # Mark these points as assigned
            for idx in clusters[i]:
                unassigned[idx] = False
                
        # Update centroids
        new_centroids = np.zeros((k, n_features))
        for i in range(k):
            cluster_points = data[clusters[i]]
            new_centroids[i] = np.mean(cluster_points, axis=0)
            
            # Normalize centroid to unit length (since we're working with unit vectors)
            norm = np.linalg.norm(new_centroids[i])
            if norm > 1e-10:  # Avoid division by zero
                new_centroids[i] /= norm
        
        # Check for convergence
        if np.allclose(centroids, new_centroids, atol=1e-6):
            print(f"Converged after {iteration+1} iterations.")
            break
            
        centroids = new_centroids
        
    # Convert clusters to numpy arrays
    clusters = [np.array(c) for c in clusters]
    
    return centroids, clusters


def analyze_cluster_fit(data: np.ndarray, centroids: np.ndarray, clusters: List[np.ndarray]) -> Dict:
    """
    Analyze how well the clusters fit vMF distributions.
    
    Args:
        data: Original data points
        centroids: Cluster centroids
        clusters: Lists of point indices for each cluster
        
    Returns:
        Dictionary with analysis metrics
    """
    n_clusters = len(centroids)
    results = {
        'cosine_mean': [],
        'cosine_std': [],
        'cosine_min': [],
        'cosine_max': [],
        'magnitude': [],
        'kappa_approx': [],
        'kappa_exact': [],
        'kappa_ratio': []
    }
    
    # For each cluster
    for i in range(n_clusters):
        # Get cluster points
        cluster_points = data[clusters[i]]
        centroid = centroids[i]
        
        # Calculate cosine similarity between points and centroid
        cosine_sims = np.dot(cluster_points, centroid)
        
        # Calculate unnormalized centroid (mean of points)
        unnormalized_centroid = np.mean(cluster_points, axis=0)
        magnitude = np.linalg.norm(unnormalized_centroid)
        
        # Calculate kappa values
        kappa_approx = vmf_concentration(jnp.array(unnormalized_centroid)).item()
        kappa_exact = vmf_concentration_exact(jnp.array(unnormalized_centroid)).item()
        
        # Store results
        results['cosine_mean'].append(np.mean(cosine_sims))
        results['cosine_std'].append(np.std(cosine_sims))
        results['cosine_min'].append(np.min(cosine_sims))
        results['cosine_max'].append(np.max(cosine_sims))
        results['magnitude'].append(magnitude)
        results['kappa_approx'].append(kappa_approx)
        results['kappa_exact'].append(kappa_exact)
        results['kappa_ratio'].append(kappa_approx / kappa_exact if kappa_exact > 0 else 1.0)
    
    return results


def run_kmeans_test():
    """Run test of k-means clustering and analyze the results."""
    print("Generating unit vectors...")
    key = PRNGKey(42)
    
    # Generate 2^12 unit vectors in R^64
    n_samples = 2**12  # 4096
    dim = 64
    unit_vectors = generate_unit_vectors(key, n_samples, dim)
    
    # Convert to numpy for sklearn
    np_vectors = np.array(unit_vectors)
    
    print(f"Running balanced k-means to cluster {n_samples} vectors into {2**6} clusters...")
    centroids, clusters = balanced_kmeans_jax(np_vectors, k=2**6, max_iters=50)
    
    print("Analyzing cluster fit...")
    results = analyze_cluster_fit(np_vectors, centroids, clusters)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Avg cosine similarity: {np.mean(results['cosine_mean']):.4f}")
    print(f"Avg cosine std dev: {np.mean(results['cosine_std']):.4f}")
    print(f"Avg magnitude of unnormalized centroid: {np.mean(results['magnitude']):.4f}")
    print(f"Avg kappa (approx): {np.mean(results['kappa_approx']):.4f}")
    print(f"Avg kappa (exact): {np.mean(results['kappa_exact']):.4f}")
    print(f"Avg kappa ratio (approx/exact): {np.mean(results['kappa_ratio']):.4f}")
    
    # Plot distribution of metrics
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.hist(results['magnitude'], bins=20)
    plt.title('Centroid Magnitude Distribution')
    plt.xlabel('Magnitude')
    plt.ylabel('Count')
    
    plt.subplot(2, 2, 2)
    plt.hist(results['cosine_mean'], bins=20)
    plt.title('Mean Cosine Similarity Distribution')
    plt.xlabel('Mean Cosine Similarity')
    plt.ylabel('Count')
    
    plt.subplot(2, 2, 3)
    plt.hist(results['kappa_approx'], bins=20, alpha=0.7, label='Approx')
    plt.hist(results['kappa_exact'], bins=20, alpha=0.7, label='Exact')
    plt.title('Kappa Distribution')
    plt.xlabel('Kappa')
    plt.ylabel('Count')
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.scatter(results['magnitude'], results['kappa_exact'])
    plt.title('Magnitude vs Kappa')
    plt.xlabel('Centroid Magnitude')
    plt.ylabel('Kappa (Exact)')
    
    plt.tight_layout()
    plt.savefig('kmeans_vmf_fit.png', dpi=300)
    print("Plot saved as 'kmeans_vmf_fit.png'")
    
    return np_vectors, centroids, clusters, results


def test_query_accuracy():
    """Test accuracy of log_expected_query_mass on k-means clusters."""
    # First run the clustering to get data, centroids, and clusters
    np_vectors, centroids, clusters, cluster_results = run_kmeans_test()
    
    # Generate some test queries
    key = PRNGKey(43)  # Different seed than clustering
    n_queries = 10
    test_queries = generate_unit_vectors(key, n_queries, dim=64)
    np_queries = np.array(test_queries)
    
    print(f"\nTesting log_expected_query_mass with {n_queries} queries...")
    
    # Store results
    query_results = {
        'exact_mass': [],
        'approx_mass': [],
        'relative_error': [],
        'centroid_magnitude': [],
        'kappa_exact': [],
        'cosine_similarity': [],
        'query_idx': [],
        'cluster_idx': []
    }
    
    # Test each query against each cluster
    for q_idx in range(n_queries):
        query = np_queries[q_idx]
        query_jax = jnp.array(query)
        
        for c_idx in range(len(centroids)):
            # Get cluster info
            cluster_points = np_vectors[clusters[c_idx]]
            unnormalized_centroid = np.mean(cluster_points, axis=0)
            magnitude = np.linalg.norm(unnormalized_centroid)
            kappa_exact = jax.jit(vmf_concentration_exact)(jnp.array(unnormalized_centroid)).item()
            
            # Compute cosine similarity between query and centroid
            normalized_centroid = centroids[c_idx]
            cosine_sim = np.dot(query, normalized_centroid)
            
            # Calculate exact expected mass (average of exp(dot product))
            exact_scores = np.exp(np.dot(cluster_points, query))
            exact_mass = np.mean(exact_scores)
            
            # Calculate approximate mass using log_expected_query_mass
            approx_mass = np.exp(jax.jit(log_expected_query_mass)(
                query_jax, jnp.array(unnormalized_centroid)).item())
            
            # Calculate relative error
            relative_error = abs(approx_mass - exact_mass) / exact_mass
            
            # Store results
            query_results['exact_mass'].append(exact_mass)
            query_results['approx_mass'].append(approx_mass)
            query_results['relative_error'].append(relative_error)
            query_results['centroid_magnitude'].append(magnitude)
            query_results['kappa_exact'].append(kappa_exact)
            query_results['cosine_similarity'].append(cosine_sim)
            query_results['query_idx'].append(q_idx)
            query_results['cluster_idx'].append(c_idx)
    
    # Convert lists to numpy arrays for easier analysis
    for key in query_results:
        query_results[key] = np.array(query_results[key])
    
    # Print summary statistics
    print("\nQuery Accuracy Summary:")
    print(f"Mean relative error: {np.mean(query_results['relative_error']):.4f}")
    print(f"Median relative error: {np.median(query_results['relative_error']):.4f}")
    print(f"Max relative error: {np.max(query_results['relative_error']):.4f}")
    
    # Plot results
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    plt.scatter(query_results['cosine_similarity'], query_results['relative_error'], alpha=0.5)
    plt.title('Error vs Cosine Similarity')
    plt.xlabel('Cosine Similarity (Query·Centroid)')
    plt.ylabel('Relative Error')
    plt.yscale('log')
    
    plt.subplot(2, 2, 2)
    plt.scatter(query_results['kappa_exact'], query_results['relative_error'], alpha=0.5)
    plt.title('Error vs Kappa')
    plt.xlabel('Kappa (Exact)')
    plt.ylabel('Relative Error')
    plt.yscale('log')
    plt.xscale('log')
    
    plt.subplot(2, 2, 3)
    plt.scatter(query_results['centroid_magnitude'], query_results['relative_error'], alpha=0.5)
    plt.title('Error vs Centroid Magnitude')
    plt.xlabel('Centroid Magnitude')
    plt.ylabel('Relative Error')
    plt.yscale('log')
    
    plt.subplot(2, 2, 4)
    plt.scatter(query_results['exact_mass'], query_results['approx_mass'], alpha=0.5)
    plt.plot([min(query_results['exact_mass']), max(query_results['exact_mass'])], 
             [min(query_results['exact_mass']), max(query_results['exact_mass'])], 'r--')
    plt.title('Exact vs Approximate Mass')
    plt.xlabel('Exact Mass')
    plt.ylabel('Approximate Mass')
    
    plt.tight_layout()
    plt.savefig('query_mass_accuracy.png', dpi=300)
    print("Plot saved as 'query_mass_accuracy.png'")
    
    # Create a detailed comparison for a single query across all clusters
    sample_query_idx = 0  # Use the first query for detailed analysis
    
    # Filter results for the sample query
    mask = query_results['query_idx'] == sample_query_idx
    sample_exact_mass = query_results['exact_mass'][mask]
    sample_approx_mass = query_results['approx_mass'][mask]
    sample_cluster_idx = query_results['cluster_idx'][mask]
    
    # Sort by exact mass
    sort_indices = np.argsort(sample_exact_mass)
    sorted_exact_mass = sample_exact_mass[sort_indices]
    sorted_approx_mass = sample_approx_mass[sort_indices]
    sorted_cluster_idx = sample_cluster_idx[sort_indices]
    
    # Calculate relative error for sorted values
    sorted_relative_error = np.abs(sorted_approx_mass - sorted_exact_mass) / sorted_exact_mass
    
    # Plot comparison for single query across all clusters
    plt.figure(figsize=(14, 10))
    
    # Bar chart comparing exact and approximate mass
    plt.subplot(2, 1, 1)
    x = np.arange(len(sorted_cluster_idx))
    width = 0.35
    
    # Set up a logarithmic scale for better visibility
    plt.yscale('log')
    plt.bar(x - width/2, sorted_exact_mass, width, label='Exact Mass')
    plt.bar(x + width/2, sorted_approx_mass, width, label='Approximate Mass')
    
    plt.title(f'Comparison of Exact vs Approximate Mass for Query {sample_query_idx}')
    plt.xlabel('Cluster Index (sorted by Exact Mass)')
    plt.ylabel('Mass (log scale)')
    plt.legend()
    
    # Add cluster indices as x-tick labels
    plt.xticks(x, sorted_cluster_idx)
    
    # Plot the relative error for each cluster
    plt.subplot(2, 1, 2)
    plt.plot(x, sorted_relative_error, 'ro-')
    plt.axhline(y=0.1, color='g', linestyle='--', label='10% Error')
    plt.title(f'Relative Error for Query {sample_query_idx} (sorted by Exact Mass)')
    plt.xlabel('Cluster Index (sorted by Exact Mass)')
    plt.ylabel('Relative Error')
    plt.xticks(x, sorted_cluster_idx)
    plt.yscale('log')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('single_query_comparison.png', dpi=300)
    print("Single query comparison plot saved as 'single_query_comparison.png'")
    
    return query_results


def balanced_kmeans_jax(data: np.ndarray, k: int, max_iters: int = 20, seed: int = 42) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Wrapper for JAX implementation of balanced k-means using cluster_spherical.
    Each cluster will have exactly the same number of points.
    
    Args:
        data: Input data of shape (n_samples, n_features)
        k: Number of clusters
        max_iters: Maximum number of iterations (not used, included for interface compatibility)
        seed: Random seed for centroid initialization
        
    Returns:
        centroids: Array of cluster centroids
        clusters: List of arrays with indices of data points in each cluster
    """
    n_samples, n_features = data.shape
    
    # Check if the number of samples is divisible by k
    if n_samples % k != 0:
        raise ValueError(f"Number of samples ({n_samples}) must be divisible by k ({k})")
    
    # Calculate levels needed for k clusters (k = 2^levels)
    levels = int(np.log2(k))
    if 2**levels != k:
        raise ValueError(f"k ({k}) must be a power of 2")
    
    # Set random seed
    key = jax.random.PRNGKey(seed)
    
    # Convert data to JAX array
    jax_data = jnp.array(data)
    
    # Call cluster_spherical
    centers, log_weights, clustered_data = cluster_spherical(jax_data, levels=levels)
    
    # Convert results back to numpy
    np_centers = np.array(centers)
    np_clustered_data = np.array(clustered_data)
    
    # Extract just the leaf node centers
    leaf_start_idx = len(np_centers) - k
    leaf_centers = np_centers[leaf_start_idx:]
    
    # Create list of cluster indices
    clusters = []
    points_per_cluster = n_samples // k
    
    # Get original indices by comparing clustered data to original data
    for i in range(k):
        # Flatten the cluster data to make comparison easier
        cluster_points = np_clustered_data[i].reshape(points_per_cluster, -1)
        
        # Find indices of these points in the original data
        # This is inefficient but matches the interface of balanced_kmeans_scipy
        indices = []
        for point in cluster_points:
            # Find the index of this point in the original data
            # Using approximate equality due to potential floating point differences
            matches = np.all(np.isclose(data, point), axis=1)
            if np.any(matches):
                idx = np.where(matches)[0][0]
                indices.append(idx)
        
        clusters.append(np.array(indices))
    
    return leaf_centers, clusters


if __name__ == "__main__":
    test_query_accuracy()
