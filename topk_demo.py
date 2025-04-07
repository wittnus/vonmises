import numpy as np
import jax
import jax.numpy as jnp
from jax.random import PRNGKey
import matplotlib.pyplot as plt
import time
from lib import (
    cluster_spherical, 
    simple_approx_qkv_attention, 
    batched_simple_approx_qkv_attention,
    topk_approx_qkv_attention, 
    batched_topk_approx_qkv_attention,
    #log_expected_query_mass,
)
from lib import simple_log_expected_query_mass as log_expected_query_mass

def generate_random_data(key: PRNGKey, dim: int = 64, n_samples: int = 4096, n_queries: int = 100):
    """Generate random data for attention experiments.
    
    Args:
        key: JAX random key
        dim: Dimension of key/query vectors
        n_samples: Number of key/value pairs
        n_queries: Number of query vectors
        
    Returns:
        Tuple of (queries, keys, values)
    """
    key1, key2, key3 = jax.random.split(key, 3)
    
    # Generate keys with varying magnitudes
    keys = jax.random.normal(key1, (n_samples, dim))
    keys = keys / jnp.sqrt(jnp.sum(keys**2, axis=1, keepdims=True))  # Normalize
    
    # Generate random values
    values = jax.random.normal(key2, (n_samples, dim))  # Values same dimension as keys
    
    # Generate test queries
    queries = jax.random.normal(key3, (n_queries, dim))
    queries = queries / jnp.sqrt(jnp.sum(queries**2, axis=1, keepdims=True))  # Normalize

    #key4, _ = jax.random.split(key3)
    #gamma_variates = jax.random.gamma(key4, 1.0, (n_queries,))
    #queries = queries * gamma_variates[:, None]  # Scale queries by gamma variates
    
    return 10.*queries, keys, values

def compute_exact_attention(queries, keys, values):
    """Compute exact attention for comparison."""
    results = []
    
    for query in queries:
        scores = jnp.exp(jnp.dot(keys, query))
        weighted_sum = jnp.sum(scores[:, None] * values, axis=0)
        normalization = jnp.sum(scores)
        result = weighted_sum / normalization
        results.append(result)
    
    return jnp.stack(results)

def compute_relative_error(exact, approx):
    """Compute relative error between exact and approximate results."""
    # More numerically stable implementation
    norm_diff = jnp.sqrt(jnp.sum((exact - approx)**2))
    norm_exact = jnp.sqrt(jnp.sum(exact**2))
    
    # Avoid division by zero
    safe_norm_exact = jnp.maximum(norm_exact, 1e-8)
    rel_error = norm_diff / safe_norm_exact
    
    return rel_error

def evaluate_topk_accuracy(queries, keys, values, max_k=32, levels=6):
    """Evaluate the accuracy of topk_approx_qkv_attention for different k values.
    
    Args:
        queries: Batch of query vectors [n_queries, dim]
        keys: Key vectors [n_keys, dim]
        values: Value vectors [n_keys, dim_v]
        max_k: Maximum k value to test
        levels: Number of hierarchical clustering levels
        
    Returns:
        Dictionary of results
    """
    # Use fewer queries for testing to speed things up
    test_queries = queries[:20] if len(queries) > 20 else queries
    print(f"Clustering {keys.shape[0]} keys with {levels} levels...")
    # Perform hierarchical clustering
    centers, log_weights, weighted_means, clustered_keys, clustered_values = cluster_spherical(
        keys, levels=levels, vs=values
    )
    
    # Number of leaf clusters
    n_clusters = clustered_keys.shape[0]
    print(f"Created {n_clusters} leaf clusters with {clustered_keys.shape[1]} keys per cluster")
    
    # Compute exact attention (baseline)
    print("Computing exact attention (baseline)...")
    exact_results = compute_exact_attention(test_queries, keys, values)
    
    # Evaluate for different k values
    k_values = [0] + [2**i for i in range(0, int(np.log2(max_k)) + 1)]
    k_values = [k for k in k_values if k <= max_k]  # Ensure we don't exceed max_k
    
    # Add n_clusters to test the extreme case
    if n_clusters not in k_values:
        k_values.append(n_clusters)
    
    errors = []
    times = []
    k_result_pairs = []  # Store (k, results) pairs for debugging
    
    print("\nEvaluating different k values:")
    for k in k_values:
        print(f"Testing k={k}...")
        
        # Run with warmup for accurate timing
        _ = batched_topk_approx_qkv_attention(
            test_queries[:2], centers, log_weights, clustered_keys, clustered_values, 
            weighted_means, k=k
        )
        
        # Measure execution time
        start_time = time.time()
        topk_results = batched_topk_approx_qkv_attention(
            test_queries, centers, log_weights, clustered_keys, clustered_values, 
            weighted_means, k=k
        )
        
        # Store results for debugging
        k_result_pairs.append((k, topk_results))
        execution_time = time.time() - start_time
        
        # Calculate errors
        batch_errors = []
        for i in range(len(test_queries)):
            error = compute_relative_error(exact_results[i], topk_results[i])
            batch_errors.append(error)
        
        mean_error = jnp.mean(jnp.array(batch_errors))
        errors.append(mean_error.item())
        times.append(execution_time)
        
        print(f"  k={k}: mean error={mean_error:.6f}, time={execution_time:.4f}s")
        
        # For k=64 (all clusters), print detailed error analysis
        if k == n_clusters:
            print(f"  Detailed analysis for k={k} (should be exact):")
            min_err = jnp.min(jnp.array(batch_errors))
            max_err = jnp.max(jnp.array(batch_errors))
            median_err = jnp.median(jnp.array(batch_errors))
            print(f"    Min error: {min_err:.8f}")
            print(f"    Max error: {max_err:.8f}")
            print(f"    Median error: {median_err:.8f}")
            
            # Check if there are numerical issues
            print("    Checking for numerical issues...")
            if max_err > 0.01:  # If error is still significant
                # Look at some individual errors
                worst_idx = jnp.argmax(jnp.array(batch_errors))
                print(f"    Worst query (idx={worst_idx}):")
                print(f"      Error: {batch_errors[worst_idx]:.8f}")
                
                # Look at norms
                exact_norm = jnp.linalg.norm(exact_results[worst_idx])
                approx_norm = jnp.linalg.norm(topk_results[worst_idx])
                print(f"      Exact result norm: {exact_norm:.8f}")
                print(f"      Approx result norm: {approx_norm:.8f}")
                print(f"      Norm ratio: {approx_norm/exact_norm:.8f}")
                
                # Check residual calculation
                print("\n    Debugging residual calculation:")
                # For the worst query, manually compute what's happening
                query = test_queries[worst_idx]
                
                # 1. Manual calculation of all the attention weights for exact method
                print("    Computing exact attention weights...")
                exact_scores = jnp.exp(jnp.dot(keys, query))
                exact_total = jnp.sum(exact_scores)
                
                # 2. Calculate the weights for the top clusters in topk method
                print("    Computing weights for top clusters...")
                # First get the scores for all nodes
                def score_node(centroid, log_weight):
                    return log_expected_query_mass(query, centroid) + log_weight
                
                # Get leaf node scores
                leaf_start_idx = centers.shape[0] - clustered_keys.shape[0]
                leaf_indices = jnp.arange(clustered_keys.shape[0])
                leaf_nodes = leaf_start_idx + leaf_indices
                
                # Calculate scores manually
                leaf_scores = []
                for i in leaf_indices:
                    node_idx = leaf_start_idx + i
                    score = score_node(centers[node_idx], log_weights[node_idx])
                    leaf_scores.append(score)
                
                # Find top-k leaf indices
                leaf_scores = jnp.array(leaf_scores)
                topk_leaves = jnp.argsort(-leaf_scores)[:k]
                
                # Get total attention weight in top clusters
                exact_weight_in_topk = 0.0
                for i in topk_leaves:
                    # Get keys for this cluster
                    cluster_keys = clustered_keys[i]
                    # Calculate dot products
                    cluster_scores = jnp.exp(jnp.dot(cluster_keys, query))
                    # Sum weights
                    exact_weight_in_topk += jnp.sum(cluster_scores)
                
                # 3. Report findings
                print(f"    Total exact attention weight: {exact_total:.8f}")
                print(f"    Weight in top-{k} clusters: {exact_weight_in_topk:.8f}")
                print(f"    Weight coverage: {exact_weight_in_topk/exact_total:.4%}")
                
                # If not all weight is covered, this explains why error persists
                if exact_weight_in_topk < 0.99 * exact_total:
                    print("    ISSUE FOUND: Not all attention weight is covered by top clusters!")
                    print("    This explains why error doesn't approach zero as k increases.")
    
    # Compare with a direct calculation for k=64 (process exact clusters with no residual)
    if n_clusters in k_values:
        print("\nTesting custom direct calculation (k=64 with no residual)...")
        k = n_clusters
        
        # Run with a custom implementation for k=64
        def exact_attention_with_clustered_keys(query, clustered_keys, clustered_values):
            """Compute attention using all clusters directly with no residual."""
            total_weighted_sum = jnp.zeros_like(clustered_values[0, 0])
            total_norm = 0.0
            
            # Process each cluster
            for i in range(clustered_keys.shape[0]):
                # Get keys and values for this cluster
                keys = clustered_keys[i]
                values = clustered_values[i]
                
                # Compute attention scores for this cluster
                scores = jnp.exp(jnp.dot(query, keys.T))
                
                # Update weighted sum and normalization
                cluster_weighted_sum = jnp.sum(scores[:, None] * values, axis=0)
                cluster_norm = jnp.sum(scores)
                
                total_weighted_sum += cluster_weighted_sum
                total_norm += cluster_norm
            
            # Normalize
            safe_norm = jnp.maximum(total_norm, 1e-10)
            return total_weighted_sum / safe_norm
        
        # Calculate for all test queries
        custom_results = []
        for query in test_queries:
            result = exact_attention_with_clustered_keys(query, clustered_keys, clustered_values)
            custom_results.append(result)
        custom_results = jnp.stack(custom_results)
        
        # Calculate errors
        custom_batch_errors = []
        for i in range(len(test_queries)):
            error = compute_relative_error(exact_results[i], custom_results[i])
            custom_batch_errors.append(error)
        
        custom_mean_error = jnp.mean(jnp.array(custom_batch_errors))
        print(f"  Custom k=64 approach: mean error={custom_mean_error:.6f}")
        
        # Compare with our actual k=64 results
        last_k_idx = k_values.index(n_clusters)
        print(f"  Regular k=64 approach: mean error={errors[last_k_idx]:.6f}")
        
        # Check if there's a difference between the two approaches
        avg_diff = jnp.mean(jnp.abs(custom_results - k_result_pairs[last_k_idx][1]))
        print(f"  Average difference between approaches: {avg_diff:.8f}")
    
    # Compare with simple_approx (which is essentially k=1 but with different implementation)
    print("\nComparing with simple_approx_qkv_attention:")
    
    # Warmup
    _ = batched_simple_approx_qkv_attention(
        test_queries[:2], centers, log_weights, clustered_keys, clustered_values, weighted_means
    )
    
    # Measure execution time
    start_time = time.time()
    simple_results = batched_simple_approx_qkv_attention(
        test_queries, centers, log_weights, clustered_keys, clustered_values, weighted_means
    )
    simple_time = time.time() - start_time
    
    # Calculate errors
    simple_batch_errors = []
    for i in range(len(test_queries)):
        error = compute_relative_error(exact_results[i], simple_results[i])
        simple_batch_errors.append(error)
    
    simple_mean_error = jnp.mean(jnp.array(simple_batch_errors))
    print(f"  simple_approx: mean error={simple_mean_error:.6f}, time={simple_time:.4f}s")
    
    # Compare error distributions between k=1 and k=n
    if n_clusters in k_values:
        k1_idx = k_values.index(1)
        kn_idx = k_values.index(n_clusters)
        
        k1_errors = batch_errors  # These are from the most recent k loop (which would be k=n_clusters)
        kn_errors = []  # Need to recompute these
        
        # Get results for k=1 and k=n
        k1_results = k_result_pairs[k1_idx][1]
        kn_results = k_result_pairs[kn_idx][1]
        
        # Check if they're the same or different
        avg_diff = jnp.mean(jnp.abs(k1_results - kn_results))
        print(f"\nDifference analysis between k=1 and k={n_clusters}:")
        print(f"  Average absolute difference: {avg_diff:.8f}")
        
        if avg_diff < 1e-6:
            print("  WARNING: Results are nearly identical! This suggests the algorithm might not be")
            print("  properly using all k clusters when k > 1.")
            
        # Compare some specific element by element
        if len(test_queries) > 0:
            q_idx = 0  # First query
            print(f"\nComparing first element of output for query {q_idx}:")
            print(f"  Exact: {exact_results[q_idx][0]:.8f}")
            print(f"  k=1:   {k1_results[q_idx][0]:.8f}")
            print(f"  k={n_clusters}: {kn_results[q_idx][0]:.8f}")
            
            # If values have high dimension, check a few more elements
            if exact_results[q_idx].shape[0] > 5:
                print("\nSampling a few more elements for query 0:")
                for i in range(1, min(5, exact_results[q_idx].shape[0])):
                    print(f"  Element {i}:")
                    print(f"    Exact: {exact_results[q_idx][i]:.8f}")
                    print(f"    k=1:   {k1_results[q_idx][i]:.8f}")
                    print(f"    k={n_clusters}: {kn_results[q_idx][i]:.8f}")
    
    # Analyze probability mass distribution for a sample query
    sample_query_idx = 0  # Use the first query as a representative example
    if len(test_queries) > 0:
        print("\nAnalyzing probability mass distribution for sample query...")
        query = test_queries[sample_query_idx]
        
        # 1. Calculate exact attention weights for all keys
        exact_scores = jnp.exp(jnp.dot(keys, query))
        total_exact_mass = jnp.sum(exact_scores)
        
        # 2. Calculate the actual attention mass in each cluster
        exact_cluster_masses = []
        cluster_indices = []
        
        for i in range(n_clusters):
            cluster_keys = clustered_keys[i]
            cluster_scores = jnp.exp(jnp.dot(cluster_keys, query))
            cluster_mass = jnp.sum(cluster_scores)
            cluster_indices.append(i)
            exact_cluster_masses.append(cluster_mass / total_exact_mass)  # Normalize
        
        # 3. Calculate estimated attention mass using log_expected_query_mass
        # First calculate the leaf start index
        leaf_start_idx = centers.shape[0] - clustered_keys.shape[0]
        
        estimated_cluster_masses = []
        for i in range(n_clusters):
            leaf_idx = leaf_start_idx + i
            centroid = centers[leaf_idx]
            # Get log estimated mass for this cluster
            log_mass = log_expected_query_mass(query, centroid) + log_weights[leaf_idx]
            estimated_mass = jnp.exp(log_mass)
            estimated_cluster_masses.append(estimated_mass)
        
        # Normalize the estimated masses to sum to 1
        total_estimated_mass = sum(estimated_cluster_masses)
        estimated_cluster_masses = [m / total_estimated_mass for m in estimated_cluster_masses]
        
        # 4. Sort clusters by exact probability mass (descending)
        sorted_indices = jnp.argsort(-jnp.array(exact_cluster_masses))
        sorted_exact_masses = [exact_cluster_masses[i] for i in sorted_indices]
        sorted_estimated_masses = [estimated_cluster_masses[i] for i in sorted_indices]
        sorted_cluster_indices = [cluster_indices[i] for i in sorted_indices]
        
        print(f"  Sample query index: {sample_query_idx}")
        print(f"  Total probability mass: {total_exact_mass:.4f}")
        print(f"  Mass in top-1 cluster: {sorted_exact_masses[0]:.4f} ({sorted_exact_masses[0]*100:.2f}%)")
        print(f"  Mass in top-4 clusters: {sum(sorted_exact_masses[:4]):.4f} ({sum(sorted_exact_masses[:4])*100:.2f}%)")
        print(f"  Mass in top-8 clusters: {sum(sorted_exact_masses[:8]):.4f} ({sum(sorted_exact_masses[:8])*100:.2f}%)")
        
        # Store the probability mass data for plotting
        prob_mass_data = {
            "query_idx": sample_query_idx,
            "exact_masses": sorted_exact_masses,
            "estimated_masses": sorted_estimated_masses,
            "cluster_indices": sorted_cluster_indices,
            "total_exact_mass": total_exact_mass
        }
    else:
        prob_mass_data = None
    
    # Return results in a dictionary
    return {
        "k_values": k_values,
        "errors": errors,
        "times": times,
        "simple_error": simple_mean_error.item(),
        "simple_time": simple_time,
        "exact_results": exact_results,
        "topk_results": topk_results,
        "simple_results": simple_results,
        "prob_mass_data": prob_mass_data
    }

def plot_results(results):
    """Plot the results of evaluating topk accuracy.
    
    Args:
        results: Dictionary of results from evaluate_topk_accuracy
    """
    k_values = results["k_values"]
    errors = results["errors"]
    times = results["times"]
    simple_error = results["simple_error"]
    simple_time = results["simple_time"]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot error vs k
    ax1.plot(k_values, errors, 'o-', linewidth=2, label='topk_approx')
    ax1.axhline(y=simple_error, color='r', linestyle='--', label='simple_approx')
    ax1.set_xlabel('k (number of clusters)')
    ax1.set_ylabel('Mean Relative Error')
    ax1.set_title('Error vs. k')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True)
    ax1.legend()
    
    # Plot time vs k
    ax2.plot(k_values, times, 'o-', linewidth=2, label='topk_approx')
    ax2.axhline(y=simple_time, color='r', linestyle='--', label='simple_approx')
    ax2.set_xlabel('k (number of clusters)')
    ax2.set_ylabel('Execution Time (seconds)')
    ax2.set_title('Time vs. k')
    ax2.set_xscale('log')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('topk_evaluation.png', dpi=300)
    plt.show()
    
    # Create a third plot showing the error-time tradeoff
    plt.figure(figsize=(10, 6))
    plt.plot(times, errors, 'o-', linewidth=2)
    plt.plot(simple_time, simple_error, 'rs', markersize=10, label='simple_approx')
    
    # Add k-value annotations
    for i, k in enumerate(k_values):
        plt.annotate(f'k={k}', (times[i], errors[i]), 
                    xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('Execution Time (seconds)')
    plt.ylabel('Mean Relative Error')
    plt.title('Error-Time Tradeoff')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('topk_tradeoff.png', dpi=300)
    plt.show()
    
    # Plot cumulative probability mass if available
    if "prob_mass_data" in results:
        plot_cumulative_probability_mass(results["prob_mass_data"])

def plot_cumulative_probability_mass(prob_mass_data):
    """Plot cumulative probability mass across clusters.
    
    Args:
        prob_mass_data: Dictionary containing probability mass data
    """
    query_idx = prob_mass_data["query_idx"]
    exact_masses = prob_mass_data["exact_masses"]
    estimated_masses = prob_mass_data["estimated_masses"]
    cluster_indices = prob_mass_data["cluster_indices"]
    
    # Calculate cumulative sums
    exact_cumulative = np.cumsum(exact_masses)
    estimated_cumulative = np.cumsum(estimated_masses)
    
    # Create x-axis values (0 to num_clusters - 1)
    x = np.arange(len(exact_masses))
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot probability mass per cluster
    ax1.bar(x, exact_masses, alpha=0.7, label='Exact Prob. Mass')
    ax1.bar(x, estimated_masses, alpha=0.3, label='Estimated Prob. Mass')
    ax1.set_xlabel('Cluster Rank (sorted by exact probability mass)')
    ax1.set_ylabel('Probability Mass')
    ax1.set_title(f'Probability Mass per Cluster (Query #{query_idx})')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add threshold lines for top-k selection with different k values
    for k in [1, 2, 4, 8, 16, 32]:
        if k < len(exact_masses):
            ax1.axvline(x=k-0.5, color='r', linestyle='--', alpha=0.5)
            ax1.annotate(f'k={k}', xy=(k-0.5, max(exact_masses)*0.9), 
                       xytext=(5, 0), textcoords='offset points', rotation=90)
    
    # Plot cumulative probability mass
    ax2.plot(x, exact_cumulative, 'o-', linewidth=2, label='Exact Cumulative Mass')
    ax2.plot(x, estimated_cumulative, 's-', linewidth=2, label='Estimated Cumulative Mass')
    ax2.set_xlabel('Cluster Rank (sorted by exact probability mass)')
    ax2.set_ylabel('Cumulative Probability Mass')
    ax2.set_title(f'Cumulative Probability Mass (Query #{query_idx})')
    ax2.grid(True)
    ax2.legend()
    
    # Add threshold lines for top-k selection with different k values
    for k in [1, 2, 4, 8, 16, 32]:
        if k < len(exact_masses):
            ax2.axvline(x=k-0.5, color='r', linestyle='--', alpha=0.5)
            ax2.annotate(f'k={k}', xy=(k-0.5, 0.8), 
                       xytext=(5, 0), textcoords='offset points', rotation=90)
    
    # Add horizontal line at y=1.0
    ax2.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('probability_mass_analysis.png', dpi=300)
    plt.show()

def main():
    # Set random seed for reproducibility
    key = PRNGKey(42)
    
    # Parameters
    dim = 64  # Vector dimension (2^6)
    seq_length = 4096  # Sequence length (2^12)
    n_queries = 100  # Number of queries to test
    levels = 6  # 2^6 = 64 clusters
    max_k = 32  # Maximum k value to test
    
    print(f"Generating random data: dim={dim}, seq_length={seq_length}, n_queries={n_queries}")
    queries, keys, values = generate_random_data(key, dim=dim, n_samples=seq_length, n_queries=n_queries)
    
    # Evaluate accuracy for different k values
    results = evaluate_topk_accuracy(queries, keys, values, max_k=max_k, levels=levels)
    
    # Plot the results
    plot_results(results)
    
    # Print summary
    print("\nSummary:")
    print(f"  Total keys: {keys.shape[0]}")
    print(f"  Vector dimension: {dim}")
    print(f"  Cluster levels: {levels}")
    print(f"  Leaf clusters: {2**levels}")
    print(f"  Keys per cluster: {keys.shape[0] // (2**levels)}")
    
    # Calculate the optimal k value (best error-time tradeoff)
    k_values = results["k_values"]
    errors = results["errors"]
    times = results["times"]
    
    # Find the "elbow point" in the error curve
    error_reductions = [errors[i] - errors[i+1] for i in range(len(errors)-1)]
    elbow_idx = np.argmax(error_reductions)
    optimal_k = k_values[elbow_idx + 1]
    
    print(f"\nRecommended k value: {optimal_k}")
    print(f"  Error at k={optimal_k}: {errors[elbow_idx + 1]:.6f}")
    print(f"  Time at k={optimal_k}: {times[elbow_idx + 1]:.4f}s")
    
    print("\nDone! Results saved to 'topk_evaluation.png' and 'topk_tradeoff.png'")

if __name__ == "__main__":
    main()
