import numpy as np
import jax
import jax.numpy as jnp
from jax.random import PRNGKey
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import chi2
import math
from lib import (
    cluster, cluster_spherical, approx_qkv_attention, batched_approx_qkv_attention,
    simple_approx_qkv_attention, batched_simple_approx_qkv_attention,
    stochastic_approx_qkv_attention, batched_stochastic_approx_qkv_attention,
    adaptive_stochastic_approx_qkv_attention, batched_adaptive_stochastic_approx_qkv_attention,
    log_expected_query_mass
)

def generate_sphere_data(key: PRNGKey, n_samples: int = 256) -> jnp.ndarray:
    """Generate unit vectors uniformly distributed on a 3D sphere."""
    # First generate normally distributed vectors
    data = jax.random.normal(key, (n_samples, 3))
    
    # Normalize to unit length
    norms = jnp.sqrt(jnp.sum(data**2, axis=1, keepdims=True))
    unit_vectors = data / norms
    
    return unit_vectors

def generate_weighted_sphere_data(key: PRNGKey, n_samples: int = 256) -> jnp.ndarray:
    """Generate vectors with varying lengths but directions uniformly distributed on a 3D sphere."""
    # Get two separate keys
    key1, key2 = jax.random.split(key)
    
    # First generate uniformly distributed unit vectors
    unit_vectors = generate_sphere_data(key1, n_samples)
    
    # Generate random magnitudes following a gamma distribution
    magnitudes = jax.random.gamma(key2, 2.0, (n_samples, 1)) + 0.5
    
    # Scale the unit vectors by the magnitudes
    vectors = unit_vectors * magnitudes
    
    return vectors


def estimate_vmf_parameters_from_centroid(centroid):
    """
    Estimate von Mises-Fisher distribution parameters directly from an unnormalized centroid.
    
    Args:
        centroid: Unnormalized centroid vector 
        
    Returns:
        mu: Mean direction (unit vector)
        kappa: Concentration parameter
        axis1, axis2: Principal axes of uncertainty on the tangent plane
        sigma1, sigma2: Standard deviations along principal axes (isotropic)
    """
    # Get magnitude of the centroid
    R = np.linalg.norm(centroid)
    
    # Normalized mean direction
    mu = centroid / R if R > 0 else np.array([0, 0, 1.0])  # Default to north pole if zero
    
    # Compute kappa directly from R (smaller R means less concentrated)
    if R > 0.9999:  # Very concentrated
        kappa = 1000  # Approximately a very high concentration
    elif R < 0.1:  # Very dispersed
        kappa = 1.0  # Low concentration
    else:
        # Approximate kappa as a function of R
        # For von Mises-Fisher in 3D, R and kappa are related by 
        # R = coth(kappa) - 1/kappa
        # A simpler approximation: kappa ≈ R/(1-R^2)
        kappa = R / (1 - R*R) * 3  # The factor 3 is empirical
    
    # Calculate tangent space for uncertainty visualization
    # Find two vectors orthogonal to mu to define tangent plane
    if abs(mu[0]) > abs(mu[1]):
        v1 = np.array([-mu[2], 0, mu[0]])
    else:
        v1 = np.array([0, -mu[2], mu[1]])
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.cross(mu, v1)
    v2 = v2 / np.linalg.norm(v2)
    
    # The theoretical standard deviation is approximately 1/sqrt(kappa)
    sigma = 1.0 / np.sqrt(kappa) if kappa > 0 else 0.1
    
    # Use isotropic uncertainty (same in all directions)
    axis1 = v1
    axis2 = v2
    sigma1 = sigma2 = sigma
    
    return mu, kappa, axis1, axis2, sigma1, sigma2

def plot_vmf_uncertainty(ax, mu, axis1, axis2, sigma1, sigma2, confidence=0.68, samples=100, color='red', alpha=0.3, dimension='3d'):
    """
    Plot uncertainty ellipse for a von Mises-Fisher distribution.
    
    Args:
        ax: Matplotlib axis to plot on
        mu: Mean direction (unit vector)
        axis1, axis2: Principal axes of uncertainty on tangent plane
        sigma1, sigma2: Standard deviations along principal axes
        confidence: Confidence level (default 0.68 = 1 sigma)
        samples: Number of points to use for ellipse
        color: Color of ellipse
        alpha: Transparency of ellipse
        dimension: '3d' for 3D axis, '2d' for 2D projection
    """
    # Calculate chi-square value for desired confidence level (2 degrees of freedom)
    chi2_val = chi2.ppf(confidence, 2)
    
    # Generate points on a circle
    theta = np.linspace(0, 2 * np.pi, samples)
    circle_x = np.cos(theta)
    circle_y = np.sin(theta)
    
    # Scale by stddev and chi-square value
    scaled_x = circle_x * sigma1 * np.sqrt(chi2_val)
    scaled_y = circle_y * sigma2 * np.sqrt(chi2_val)
    
    # Convert to 3D points on the sphere
    sphere_points = np.zeros((samples, 3))
    for i in range(samples):
        # Point in tangent space
        tangent_point = scaled_x[i] * axis1 + scaled_y[i] * axis2
        
        # Project to sphere using exponential map (approximation)
        # For small distances, we can approximate exp_map(v) ≈ cos(|v|)·mu + sin(|v|)·v/|v|
        v_norm = np.linalg.norm(tangent_point)
        if v_norm < 1e-10:
            sphere_points[i] = mu
        else:
            sphere_points[i] = np.cos(v_norm) * mu + np.sin(v_norm) * tangent_point / v_norm
        
        # Ensure unit norm
        sphere_points[i] = sphere_points[i] / np.linalg.norm(sphere_points[i])
    
    if dimension == '3d':
        # Plot on 3D axis
        ax.plot(sphere_points[:, 0], sphere_points[:, 1], sphere_points[:, 2], 
                color=color, alpha=alpha)
    elif dimension == '2d-xy':
        # Project to XY plane
        ax.plot(sphere_points[:, 0], sphere_points[:, 1], color=color, alpha=alpha)
    elif dimension == '2d-lambert-north':
        # Lambert azimuthal projection (north pole)
        mask = sphere_points[:, 2] >= 0
        if np.any(mask):
            r = np.sqrt(2 / (1 + sphere_points[mask, 2]))
            proj_x = sphere_points[mask, 0] * r
            proj_y = sphere_points[mask, 1] * r
            ax.plot(proj_x, proj_y, color=color, alpha=alpha)
    elif dimension == '2d-lambert-south':
        # Lambert azimuthal projection (south pole)
        mask = sphere_points[:, 2] <= 0
        if np.any(mask):
            r = np.sqrt(2 / (1 - sphere_points[mask, 2]))
            proj_x = sphere_points[mask, 0] * r
            proj_y = sphere_points[mask, 1] * r
            ax.plot(proj_x, proj_y, color=color, alpha=alpha)
    elif dimension == '2d-mollweide':
        # Mollweide projection
        lon = np.arctan2(sphere_points[:, 1], sphere_points[:, 0])
        lat = np.arcsin(sphere_points[:, 2])
        
        # Iterative solution for auxiliary angle
        theta = lat.copy()
        for i in range(5):
            theta = theta - (2*theta + np.sin(2*theta) - np.pi*np.sin(lat)) / (2 + 2*np.cos(2*theta))
        
        x_moll = 2 * np.sqrt(2) / np.pi * lon * np.cos(theta)
        y_moll = np.sqrt(2) * np.sin(theta)
        
        ax.plot(x_moll, y_moll, color=color, alpha=alpha)


def plot_sphere_clusters(centers: np.ndarray, clustered_data: np.ndarray, weights=None, title_suffix="") -> None:
    """Plot clustered data on the 2-sphere with multiple projections.
    
    Args:
        centers: Array of cluster centers
        clustered_data: Data points organized by cluster
        weights: Optional array of weights/magnitudes for each point. If provided, will be used to set alpha values.
        title_suffix: Optional suffix to add to plot titles
    """
    n_clusters = clustered_data.shape[0]
    n_points_per_cluster = clustered_data.shape[1]
    
    # Flatten clustered data for plotting
    flat_data = clustered_data.reshape(-1, clustered_data.shape[-1])
    
    # Create labels array for coloring
    labels = np.repeat(np.arange(n_clusters), n_points_per_cluster)
    
    # Get only leaf cluster centers and normalize for visualization
    n_centers = centers.shape[0]
    leaf_nodes_start = n_centers - n_clusters
    leaf_centers = centers[leaf_nodes_start:]
    
    # Calculate center magnitudes for visualization
    center_magnitudes = np.sqrt(np.sum(leaf_centers**2, axis=1, keepdims=True))
    # Normalize centers to unit sphere for visualization
    normalized_leaf_centers = leaf_centers / np.maximum(center_magnitudes, 1e-10)
    
    # If weights are provided, compute alpha values based on them
    alphas = None
    if weights is not None:
        # Normalize weights so max alpha is exactly 0.9 and min is 0.1
        min_weight = np.min(weights)
        max_weight = np.max(weights)
        if max_weight > min_weight:
            alphas = 0.1 + 0.8 * (weights - min_weight) / (max_weight - min_weight)
            # Ensure max is exactly 0.9
            alphas = 0.9 * alphas / np.max(alphas)
        else:
            alphas = np.full_like(weights, 0.7)
    else:
        # Default alpha value if no weights
        alphas = np.full(flat_data.shape[0], 0.7)
    
    # Create a figure with a grid layout
    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1])
    
    # 3D plot of sphere with alpha values
    ax1 = fig.add_subplot(gs[0, 0:2], projection='3d')
    
    # Plot each point with its own alpha
    for i in range(len(flat_data)):
        ax1.scatter(flat_data[i, 0], flat_data[i, 1], flat_data[i, 2], 
                  c=[plt.cm.gist_rainbow(labels[i] / n_clusters)], 
                  alpha=alphas[i], s=30)
    
    # Plot normalized cluster centers
    ax1.scatter(normalized_leaf_centers[:, 0], normalized_leaf_centers[:, 1], normalized_leaf_centers[:, 2], 
              c='red', marker='x', s=100, linewidths=2)
    
    # Estimate vMF parameters and plot uncertainty ellipses
    for i in range(n_clusters):
        cluster_center = leaf_centers[i]
        # Convert unnormalized center to normalized direction
        center_dir = normalized_leaf_centers[i]
        
        # Estimate vMF parameters directly from the unnormalized centroid
        mu, kappa, axis1, axis2, sigma1, sigma2 = estimate_vmf_parameters_from_centroid(cluster_center)
        
        # For visualization, use the normalized center as the mean direction
        # but keep the uncertainty estimates from the data
        plot_vmf_uncertainty(ax1, center_dir, axis1, axis2, sigma1, sigma2, 
                            color=plt.cm.gist_rainbow(i / n_clusters), 
                            dimension='3d')
    
    # Draw a wireframe sphere
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x = 0.99 * np.outer(np.cos(u), np.sin(v))
    y = 0.99 * np.outer(np.sin(u), np.sin(v))
    z = 0.99 * np.outer(np.ones(np.size(u)), np.cos(v))
    ax1.plot_surface(x, y, z, color='lightgray', alpha=0.1)
    
    ax1.set_title(f'3D Sphere View{title_suffix}')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_box_aspect([1, 1, 1])  # Equal aspect ratio
    
    # Orthographic projection (XY plane)
    ax2 = fig.add_subplot(gs[0, 2])
    
    # Plot each point with its own alpha
    for i in range(len(flat_data)):
        ax2.scatter(flat_data[i, 0], flat_data[i, 1], 
                  c=[plt.cm.gist_rainbow(labels[i] / n_clusters)], 
                  alpha=alphas[i], s=30)
    
    ax2.scatter(normalized_leaf_centers[:, 0], normalized_leaf_centers[:, 1], c='red', marker='x', s=100, linewidths=2)
    
    # Add uncertainty ellipses for XY projection
    for i in range(n_clusters):
        cluster_points = clustered_data[i]
        center_dir = normalized_leaf_centers[i]
        
        # If weights are provided, use them for the cluster
        if weights is not None:
            cluster_weights = weights[n_points_per_cluster * i:n_points_per_cluster * (i + 1)]
        else:
            cluster_weights = None
        
        # Estimate vMF parameters directly from the unnormalized centroid
        mu, kappa, axis1, axis2, sigma1, sigma2 = estimate_vmf_parameters_from_centroid(cluster_center)
        
        # Plot uncertainty ellipse
        plot_vmf_uncertainty(ax2, center_dir, axis1, axis2, sigma1, sigma2, 
                           color=plt.cm.gist_rainbow(i / n_clusters), 
                           dimension='2d-xy')
    ax2.set_title(f'Orthographic Projection (XY plane){title_suffix}')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.add_patch(plt.Circle((0, 0), 1, fill=False, linestyle='--', color='gray'))
    ax2.set_aspect('equal')
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    # Lambert azimuthal equal-area projection (from positive Z direction)
    ax3 = fig.add_subplot(gs[1, 0])
    # Project only points with z≥0 (visible from north pole)
    mask = flat_data[:, 2] >= 0
    r = np.sqrt(2 / (1 + flat_data[mask, 2]))
    proj_x = flat_data[mask, 0] * r
    proj_y = flat_data[mask, 1] * r
    
    # Plot each point with its own alpha
    for i in range(len(proj_x)):
        masked_idx = np.where(mask)[0][i]
        ax3.scatter(proj_x[i], proj_y[i], 
                  c=[plt.cm.gist_rainbow(labels[masked_idx] / n_clusters)], 
                  alpha=alphas[masked_idx], s=30)
    
    # Project normalized centers
    centers_mask = normalized_leaf_centers[:, 2] >= 0
    centers_r = np.sqrt(2 / (1 + normalized_leaf_centers[centers_mask, 2]))
    centers_x = normalized_leaf_centers[centers_mask, 0] * centers_r
    centers_y = normalized_leaf_centers[centers_mask, 1] * centers_r
    ax3.scatter(centers_x, centers_y, c='red', marker='x', s=100, linewidths=2)
    
    # Add uncertainty ellipses for Lambert north projection
    for i in range(n_clusters):
        if normalized_leaf_centers[i, 2] >= 0:  # Only for northern hemisphere
            cluster_center = leaf_centers[i]
            center_dir = normalized_leaf_centers[i]
            
            # Estimate vMF parameters directly from the unnormalized centroid
            mu, kappa, axis1, axis2, sigma1, sigma2 = estimate_vmf_parameters_from_centroid(cluster_center)
            
            # Plot uncertainty ellipse
            plot_vmf_uncertainty(ax3, center_dir, axis1, axis2, sigma1, sigma2, 
                               color=plt.cm.gist_rainbow(i / n_clusters), 
                               dimension='2d-lambert-north')
    
    ax3.set_title(f'Lambert Equal-Area Projection (from North){title_suffix}')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.add_patch(plt.Circle((0, 0), np.sqrt(2), fill=False, linestyle='--', color='gray'))
    ax3.set_aspect('equal')
    ax3.grid(True, linestyle='--', alpha=0.5)
    
    # Lambert azimuthal projection (from negative Z direction)
    ax4 = fig.add_subplot(gs[1, 1])
    # Project only points with z≤0 (visible from south pole)
    mask = flat_data[:, 2] <= 0
    r = np.sqrt(2 / (1 - flat_data[mask, 2]))
    proj_x = flat_data[mask, 0] * r
    proj_y = flat_data[mask, 1] * r
    
    # Plot each point with its own alpha
    for i in range(len(proj_x)):
        masked_idx = np.where(mask)[0][i]
        ax4.scatter(proj_x[i], proj_y[i], 
                  c=[plt.cm.gist_rainbow(labels[masked_idx] / n_clusters)], 
                  alpha=alphas[masked_idx], s=30)
    
    # Project normalized centers
    centers_mask = normalized_leaf_centers[:, 2] <= 0
    centers_r = np.sqrt(2 / (1 - normalized_leaf_centers[centers_mask, 2]))
    centers_x = normalized_leaf_centers[centers_mask, 0] * centers_r
    centers_y = normalized_leaf_centers[centers_mask, 1] * centers_r
    ax4.scatter(centers_x, centers_y, c='red', marker='x', s=100, linewidths=2)
    
    # Add uncertainty ellipses for Lambert south projection
    for i in range(n_clusters):
        if normalized_leaf_centers[i, 2] <= 0:  # Only for southern hemisphere
            cluster_center = leaf_centers[i]
            center_dir = normalized_leaf_centers[i]
            
            # Estimate vMF parameters directly from the unnormalized centroid
            mu, kappa, axis1, axis2, sigma1, sigma2 = estimate_vmf_parameters_from_centroid(cluster_center)
            
            # Plot uncertainty ellipse
            plot_vmf_uncertainty(ax4, center_dir, axis1, axis2, sigma1, sigma2, 
                               color=plt.cm.gist_rainbow(i / n_clusters), 
                               dimension='2d-lambert-south')
    
    ax4.set_title(f'Lambert Equal-Area Projection (from South){title_suffix}')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.add_patch(plt.Circle((0, 0), np.sqrt(2), fill=False, linestyle='--', color='gray'))
    ax4.set_aspect('equal')
    ax4.grid(True, linestyle='--', alpha=0.5)
    
    # Mollweide projection
    ax5 = fig.add_subplot(gs[1, 2])
    # Convert 3D coordinates to longitude and latitude
    lon = np.arctan2(flat_data[:, 1], flat_data[:, 0])
    lat = np.arcsin(flat_data[:, 2])
    
    # Mollweide projection formula
    theta = lat.copy()
    # Iterative solution for the auxiliary angle
    for i in range(5):  # Few iterations is usually enough
        theta = theta - (2*theta + np.sin(2*theta) - np.pi*np.sin(lat)) / (2 + 2*np.cos(2*theta))
    
    x_moll = 2 * np.sqrt(2) / np.pi * lon * np.cos(theta)
    y_moll = np.sqrt(2) * np.sin(theta)
    
    # Plot each point with its own alpha
    for i in range(len(x_moll)):
        ax5.scatter(x_moll[i], y_moll[i], 
                  c=[plt.cm.gist_rainbow(labels[i] / n_clusters)], 
                  alpha=alphas[i], s=30)
    
    # Project normalized centers to Mollweide
    lon_centers = np.arctan2(normalized_leaf_centers[:, 1], normalized_leaf_centers[:, 0])
    lat_centers = np.arcsin(normalized_leaf_centers[:, 2])
    
    theta_centers = lat_centers.copy()
    for i in range(5):
        theta_centers = theta_centers - (2*theta_centers + np.sin(2*theta_centers) - np.pi*np.sin(lat_centers)) / (2 + 2*np.cos(2*theta_centers))
    
    x_moll_centers = 2 * np.sqrt(2) / np.pi * lon_centers * np.cos(theta_centers)
    y_moll_centers = np.sqrt(2) * np.sin(theta_centers)
    
    ax5.scatter(x_moll_centers, y_moll_centers, c='red', marker='x', s=100, linewidths=2)
    
    # Add uncertainty ellipses for Mollweide projection
    for i in range(n_clusters):
        cluster_center = leaf_centers[i]
        center_dir = normalized_leaf_centers[i]
        
        # Estimate vMF parameters directly from the unnormalized centroid
        mu, kappa, axis1, axis2, sigma1, sigma2 = estimate_vmf_parameters_from_centroid(cluster_center)
        
        # Plot uncertainty ellipse
        plot_vmf_uncertainty(ax5, center_dir, axis1, axis2, sigma1, sigma2, 
                           color=plt.cm.gist_rainbow(i / n_clusters), 
                           dimension='2d-mollweide')
    
    # Draw Mollweide grid
    ax5.set_title(f'Mollweide Projection{title_suffix}')
    ax5.set_aspect('equal')
    ax5.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # Different filename based on whether weights are used
    if weights is not None:
        plt.savefig('weighted_hierarchical_clustering_sphere_result.png', dpi=300)
    else:
        plt.savefig('hierarchical_clustering_sphere_result.png', dpi=300)
    
    plt.show()


def run_standard_clustering():
    # Set random seed for reproducibility
    key = PRNGKey(42)
    
    # Generate 2^8 = 256 3D unit vectors (points on sphere)
    levels = 5  # 2^5 = 32 clusters
    n_samples = 2**8  # 256 points
    print(f"Generating {n_samples} points on the unit sphere...")
    data = generate_sphere_data(key, n_samples=n_samples)
    
    # Perform hierarchical binary clustering (2^5 = 32 clusters)
    print(f"Performing hierarchical binary clustering (levels={levels})...")
    centers, clustered_data = cluster(data, levels=levels)
    
    # Plot the results
    print("Plotting sphere projections...")
    plot_sphere_clusters(centers, clustered_data, title_suffix=" (Standard Clustering)")
    
    # Print statistics about cluster sizes
    n_clusters = clustered_data.shape[0]
    n_points_per_cluster = clustered_data.shape[1]
    print("\nCluster size statistics:")
    print(f"  Number of clusters: {n_clusters}")
    print(f"  Points per cluster: {n_points_per_cluster}")
    print(f"  Expected points per cluster: {n_samples / n_clusters:.1f}")
    
    print(f"\nDone! Results saved to 'hierarchical_clustering_sphere_result.png'")
    print(f"Created {2**levels} = {n_clusters} balanced clusters with {n_samples} points on the unit sphere")

def run_weighted_clustering():
    # Set random seed for reproducibility
    key = PRNGKey(43)  # Different seed to get different data
    
    # Generate 2^8 = 256 3D vectors with varying magnitudes
    levels = 5  # 2^4 = 16 clusters (fewer for better visualization)
    n_samples = 2**8  # 256 points
    print(f"Generating {n_samples} vectors with varying magnitudes...")
    data = generate_weighted_sphere_data(key, n_samples=n_samples)
    
    # Perform spherical clustering with weighted means
    print(f"Performing weighted spherical clustering (levels={levels})...")
    centers, log_weights, clustered_data = cluster_spherical(data, levels=levels)  # vs=None case still returns 3 values
    
    # Calculate exponential magnitudes for alpha values and normalize vectors
    flat_data = clustered_data.reshape(n_samples, -1)
    magnitudes = np.sqrt(np.sum(flat_data**2, axis=1))
    exp_magnitudes = np.exp(magnitudes)
    
    # Normalize the data for visualization (preserving weights in alpha)
    normalized_data = np.copy(flat_data)
    for i in range(len(normalized_data)):
        norm = np.sqrt(np.sum(normalized_data[i]**2))
        if norm > 1e-10:
            normalized_data[i] = normalized_data[i] / norm
    
    # Reshape the normalized data back to clustered format
    normalized_clustered_data = normalized_data.reshape(clustered_data.shape)
    
    # Plot the results with magnitude (log weight) alpha values and normalized vectors
    print("Plotting weighted spherical clustering...")
    plot_sphere_clusters(
        centers, 
        normalized_clustered_data, 
        weights=magnitudes,  # Using magnitudes (log weights) for visualization
        title_suffix=" (Exp-Weighted Clustering, Visualized with Log Weights)"
    )
    
    # Print statistics about cluster sizes and centers
    n_clusters = clustered_data.shape[0]
    n_points_per_cluster = clustered_data.shape[1]
    
    # Calculate cluster weights (sum of exponential magnitudes)
    cluster_labels = np.repeat(np.arange(n_clusters), n_points_per_cluster)
    
    cluster_weights = np.zeros(n_clusters)
    for i in range(n_clusters):
        cluster_weights[i] = np.sum(exp_magnitudes[cluster_labels == i])
    
    # Convert returned log_weights to weights for comparison
    # Get just the leaf node log_weights from the full tree
    n_nodes = log_weights.shape[0]
    n_leaf_nodes = n_clusters
    leaf_start_idx = n_nodes - n_leaf_nodes
    leaf_log_weights = log_weights[leaf_start_idx:]
    returned_weights = np.exp(leaf_log_weights)
    
    # Get center magnitudes
    n_centers = centers.shape[0]  # Calculate the total number of centers
    leaf_centers = centers[n_centers - n_clusters:]
    center_magnitudes = np.sqrt(np.sum(leaf_centers**2, axis=1))
    
    # Calculate von Mises-Fisher concentration parameters for each cluster directly from centroids
    kappa_values = np.zeros(n_clusters)
    for i in range(n_clusters):
        cluster_center = leaf_centers[i]
            
        # Get just the kappa value
        _, kappa, _, _, _, _ = estimate_vmf_parameters_from_centroid(cluster_center)
        kappa_values[i] = kappa
    
    print("\nCluster statistics:")
    print(f"  Number of clusters: {n_clusters}")
    print(f"  Points per cluster: {n_points_per_cluster}")
    print(f"  Total data magnitude: {np.sum(magnitudes):.2f}")
    print(f"  Total exp magnitude: {np.sum(exp_magnitudes):.2f}")
    print(f"  Min cluster exp-weight: {np.min(cluster_weights):.2f}")
    print(f"  Max cluster exp-weight: {np.max(cluster_weights):.2f}")
    print(f"  Exp-weight std dev: {np.std(cluster_weights):.2f}")
    print(f"  Exp-weight ratio (max/min): {np.max(cluster_weights)/np.min(cluster_weights):.2f}")
    print("\nReturned log-weights statistics:")
    print(f"  Min returned exp-weight: {np.min(returned_weights):.2f}")
    print(f"  Max returned exp-weight: {np.max(returned_weights):.2f}")
    print(f"  Returned exp-weight std dev: {np.std(returned_weights):.2f}")
    print(f"  Returned/computed weight ratio: {np.mean(returned_weights/cluster_weights):.4f}")
    
    # Print hierarchical log-weights information
    print("\nHierarchical log-weights structure:")
    # Calculate number of nodes at each level
    for level in range(levels + 1):
        start_idx = 2**level - 1
        end_idx = 2**(level+1) - 1
        level_weights = np.exp(log_weights[start_idx:end_idx])
        print(f"  Level {level}: nodes {start_idx}-{end_idx-1}, mean weight: {np.mean(level_weights):.2f}")
    
    print("\nCenter statistics:")
    print(f"  Min center magnitude: {np.min(center_magnitudes):.4f}")
    print(f"  Max center magnitude: {np.max(center_magnitudes):.4f}")
    print(f"  Average center magnitude: {np.mean(center_magnitudes):.4f}")
    print(f"  Center magnitude std dev: {np.std(center_magnitudes):.4f}")
    print("\nvon Mises-Fisher concentration statistics:")
    print(f"  Min kappa: {np.min(kappa_values):.2f}")
    print(f"  Max kappa: {np.max(kappa_values):.2f}")
    print(f"  Average kappa: {np.mean(kappa_values):.2f}")
    print(f"  Kappa std dev: {np.std(kappa_values):.2f}")
    print(f"  Avg angular std dev (degrees): {np.mean(np.rad2deg(1.0/np.sqrt(kappa_values))):.2f}")
    
    print(f"\nDone! Results saved to 'weighted_hierarchical_clustering_sphere_result.png'")
    print(f"Created {2**levels} = {n_clusters} clusters with {n_samples} points with varying magnitudes")

def run_clustering_with_values():
    """Demonstrate clustering with both xs and vs arrays."""
    # Set random seed for reproducibility
    key = PRNGKey(44)  # Different seed
    
    # Generate data vectors and companion values
    levels = 3  # 2^3 = 8 clusters (small number for clarity)
    n_samples = 2**7  # 128 points
    print(f"Generating {n_samples} vectors with varying magnitudes and companion values...")
    
    # Generate xs vectors (3D)
    key1, key2 = jax.random.split(key)
    xs_data = generate_weighted_sphere_data(key1, n_samples=n_samples)
    
    # Generate vs values (2D for simplicity)
    vs_data = jax.random.normal(key2, (n_samples, 2))
    
    # Perform clustering with both xs and vs
    print(f"Performing spherical clustering with companion values (levels={levels})...")
    centers, log_weights, weighted_means, clustered_xs, clustered_vs = cluster_spherical(xs_data, levels=levels, vs=vs_data)
    
    # Print basic statistics
    n_clusters = clustered_xs.shape[0]
    n_points_per_cluster = clustered_xs.shape[1]
    
    print("\nClustering with companion values:")
    print(f"  Number of clusters: {n_clusters}")
    print(f"  Points per cluster: {n_points_per_cluster}")
    print(f"  xs shape: {clustered_xs.shape}")
    print(f"  vs shape: {clustered_vs.shape}")
    
    # Print hierarchical log-weights information
    print("\n  Hierarchical log-weights structure:")
    # Calculate number of nodes at each level
    for level in range(levels + 1):
        start_idx = 2**level - 1
        end_idx = 2**(level+1) - 1
        level_weights = np.exp(log_weights[start_idx:end_idx])
        print(f"    Level {level}: nodes {start_idx}-{end_idx-1}, mean weight: {np.mean(level_weights):.2f}")
    
    # Calculate correlation between xs and vs within clusters
    for i in range(n_clusters):
        cluster_xs = clustered_xs[i]
        cluster_vs = clustered_vs[i]
        
        # Calculate xs magnitudes
        xs_mags = np.sqrt(np.sum(cluster_xs**2, axis=1))
        
        # Calculate vs magnitudes
        vs_mags = np.sqrt(np.sum(cluster_vs**2, axis=1))
        
        # Calculate correlation
        correlation = np.corrcoef(xs_mags, vs_mags)[0, 1]
        print(f"  Cluster {i}: xs-vs magnitude correlation: {correlation:.4f}")
    
    print(f"\nDone! Demonstrated clustering with companion values.")

def run_hierarchical_attention_demo():
    """Demonstrate the sub-quadratic attention algorithm using hierarchical clustering."""
    # Set random seed for reproducibility
    key = PRNGKey(45)
    
    # Generate data for demonstration
    dim = 64  # Vector dimension
    n_samples = 2**10  # 1024 points
    levels = 5  # 2^5 = 32 clusters
    
    print(f"Running hierarchical attention demo with {n_samples} vectors of dimension {dim}")
    
    # Generate keys with varying magnitudes
    key1, key2, key3 = jax.random.split(key, 3)
    keys = jax.random.normal(key1, (n_samples, dim))
    keys = keys / jnp.sqrt(jnp.sum(keys**2, axis=1, keepdims=True))  # Normalize
    
    # Generate random values
    values = jax.random.normal(key2, (n_samples, 32))  # 32-dim values
    
    # Generate a few test queries
    n_queries = 5
    queries = jax.random.normal(key3, (n_queries, dim))
    queries = queries / jnp.sqrt(jnp.sum(queries**2, axis=1, keepdims=True))  # Normalize
    
    # Cluster the keys and values together
    print(f"Performing hierarchical clustering (levels={levels})...")
    centers, log_weights, weighted_means, clustered_keys, clustered_values = cluster_spherical(
        keys, levels=levels, vs=values)
    
    # Get dimensions
    n_clusters = 2**levels
    n_per_cluster = n_samples // n_clusters
    
    # Print information about weighted means
    print(f"  Tree structure: {n_clusters} leaf nodes, {centers.shape[0]} total nodes")
    print(f"  Weighted means shape: {weighted_means.shape}")
    print(f"  Log weights shape: {log_weights.shape}")
    
    print("Computing exact attention (baseline)...")
    # Compute exact attention for comparison (baseline)
    exact_results = []
    for q in range(n_queries):
        query = queries[q]
        
        # Standard attention formula
        scores = jnp.exp(jnp.dot(keys, query))  # [n_samples]
        # Ensure proper shape for weighted sum
        weighted_sum = jnp.sum(scores[:, None] * values, axis=0)  # Sum across samples
        normalization = jnp.sum(scores)
        exact_result = weighted_sum / normalization
        exact_results.append(exact_result)
    
    exact_results = jnp.stack(exact_results)
    
    print("Computing approximate attention using hierarchical clustering...")
    print("  Standard approach: Only compute exact attention for top-scoring leaf clusters")
    print("  Residual approach: Also approximate pruned subtrees using precomputed weighted means")
    
    # Try different beam widths and with/without residual approximation
    beam_widths = [1, 2, 4, 8]
    approx_results = []
    approx_with_residual_results = []
    
    for beam_width in beam_widths:
        # Without residual approximation (no weighted means)
        approx_result = batched_approx_qkv_attention(
            queries, centers, log_weights, clustered_keys, clustered_values, 
            weighted_means=None, beam_width=beam_width)
        approx_results.append(approx_result)
        
        # With residual approximation (using weighted means)
        approx_with_residual = batched_approx_qkv_attention(
            queries, centers, log_weights, clustered_keys, clustered_values, 
            weighted_means=weighted_means, beam_width=beam_width)
        approx_with_residual_results.append(approx_with_residual)
    
    # Compute error metrics
    print("\nError comparison for different beam widths:")
    for i, beam_width in enumerate(beam_widths):
        # Standard approach errors
        errors = []
        # Residual approach errors
        residual_errors = []
        
        for q in range(n_queries):
            exact = exact_results[q]
            approx = approx_results[i][q]
            approx_residual = approx_with_residual_results[i][q]
            
            # Make sure shapes match - data outputs might have different shapes
            # This ensures we're comparing vectors of the same dimension
            if exact.shape != approx.shape:
                print(f"Warning: Shape mismatch - exact: {exact.shape}, approx: {approx.shape}")
                # Reshape if needed or pad/truncate as appropriate
                # For now, we'll convert both to 1D arrays if they have different shapes
                exact_flat = exact.flatten()
                approx_flat = approx.flatten()
                # Ensure same length by padding or truncating
                min_len = min(len(exact_flat), len(approx_flat))
                exact_flat = exact_flat[:min_len]
                approx_flat = approx_flat[:min_len]
                
                # Compute relative error on flattened arrays
                error = jnp.sqrt(jnp.sum((exact_flat - approx_flat)**2)) / jnp.sqrt(jnp.sum(exact_flat**2))
                
                # Do the same for residual error
                approx_residual_flat = approx_residual.flatten()[:min_len]
                residual_error = jnp.sqrt(jnp.sum((exact_flat - approx_residual_flat)**2)) / jnp.sqrt(jnp.sum(exact_flat**2))
            else:
                # Shapes match, compute errors normally
                error = jnp.sqrt(jnp.sum((exact - approx)**2)) / jnp.sqrt(jnp.sum(exact**2))
                residual_error = jnp.sqrt(jnp.sum((exact - approx_residual)**2)) / jnp.sqrt(jnp.sum(exact**2))
            
            errors.append(error)
            residual_errors.append(residual_error)
        
        mean_error = jnp.mean(jnp.array(errors))
        mean_residual_error = jnp.mean(jnp.array(residual_errors))
        
        print(f"  Beam width {beam_width}:")
        print(f"    Standard approach: mean relative error = {mean_error:.6f}")
        print(f"    With residual approximation: mean relative error = {mean_residual_error:.6f}")
        print(f"    Improvement: {(1 - mean_residual_error/mean_error)*100:.2f}%")
    
    # Estimate computational savings
    total_keys = n_samples
    keys_per_query = n_clusters * n_per_cluster // beam_widths[-1]  # for largest beam width
    
    print(f"\nComputational comparison:")
    print(f"  Total keys: {total_keys}")
    print(f"  Hierarchical clusters: {n_clusters} clusters with {n_per_cluster} keys each")
    print(f"  Approx. keys processed per query (beam={beam_widths[-1]}): ~{keys_per_query}")
    print(f"  Speedup factor: ~{total_keys / keys_per_query:.1f}x")
    
    print(f"\nDone! Demonstrated sub-quadratic attention using hierarchical clustering")

def main():
    # Run standard clustering on unit vectors
    run_standard_clustering()
    
    # Run weighted clustering on vectors with varying magnitudes
    run_weighted_clustering()
    
    # Run clustering with companion values
    run_clustering_with_values()
    
    # Run hierarchical attention demo
    run_hierarchical_attention_demo()

    # Run simplified JAX attention demo
    run_simplified_jax_attention_demo()


def run_stochastic_attention_demo():
    """Demonstrate the stochastic sampling approach to hierarchical attention."""
    # Set random seed for reproducibility
    key = PRNGKey(47)
    
    # Generate data for demonstration
    dim = 64  # Vector dimension
    n_samples = 2**10  # 1024 points
    levels = 5  # 2^5 = 32 clusters
    
    print(f"Running stochastic sampling hierarchical attention demo")
    print(f"Using {n_samples} vectors of dimension {dim}, with {2**levels} clusters")
    
    # Generate keys with varying magnitudes
    key1, key2, key3, key4 = jax.random.split(key, 4)
    keys = jax.random.normal(key1, (n_samples, dim))
    keys = keys / jnp.sqrt(jnp.sum(keys**2, axis=1, keepdims=True))  # Normalize
    
    # Generate random values
    values = jax.random.normal(key2, (n_samples, 32))  # 32-dim values
    
    # Generate a few test queries
    n_queries = 5
    queries = jax.random.normal(key3, (n_queries, dim))
    queries = queries / jnp.sqrt(jnp.sum(queries**2, axis=1, keepdims=True))  # Normalize
    
    # Cluster the keys and values together
    print(f"Performing hierarchical clustering (levels={levels})...")
    centers, log_weights, weighted_means, clustered_keys, clustered_values = cluster_spherical(
        keys, levels=levels, vs=values)
    
    # Print tree statistics
    n_clusters = 2**levels
    print(f"  Tree structure: {n_clusters} leaf clusters, {centers.shape[0]} total nodes")
    
    # Compute exact attention for comparison (baseline)
    print("Computing exact attention (baseline)...")
    exact_results = []
    for q in range(n_queries):
        query = queries[q]
        scores = jnp.exp(jnp.dot(keys, query))
        weighted_sum = jnp.sum(scores[:, None] * values, axis=0)
        normalization = jnp.sum(scores)
        exact_results.append(weighted_sum / normalization)
    
    exact_results = jnp.stack(exact_results)
    
    # Compare different attention methods
    methods = [
        ("Deterministic (simplified)", 
         lambda: batched_simple_approx_qkv_attention(
             queries, centers, log_weights, clustered_keys, clustered_values, weighted_means)),
        ("Stochastic (T=0.1)", 
         lambda: batched_stochastic_approx_qkv_attention(
             queries, centers, log_weights, clustered_keys, clustered_values, weighted_means, 0.1, key4)),
        ("Stochastic (T=0.5)", 
         lambda: batched_stochastic_approx_qkv_attention(
             queries, centers, log_weights, clustered_keys, clustered_values, weighted_means, 0.5, key4)),
        ("Stochastic (T=1.0)", 
         lambda: batched_stochastic_approx_qkv_attention(
             queries, centers, log_weights, clustered_keys, clustered_values, weighted_means, 1.0, key4)),
        ("Stochastic (T=2.0)", 
         lambda: batched_stochastic_approx_qkv_attention(
             queries, centers, log_weights, clustered_keys, clustered_values, weighted_means, 2.0, key4)),
        ("Stochastic (T=5.0)", 
         lambda: batched_stochastic_approx_qkv_attention(
             queries, centers, log_weights, clustered_keys, clustered_values, weighted_means, 5.0, key4))
    ]
    
    # Storage for results
    all_results = []
    execution_times = []
    errors = []
    
    # First do a warmup run to compile everything
    print("\nWarming up JIT compilation...")
    batched_simple_approx_qkv_attention(
        queries, centers, log_weights, clustered_keys, clustered_values, weighted_means)
    batched_stochastic_approx_qkv_attention(
        queries, centers, log_weights, clustered_keys, clustered_values, weighted_means, 1.0, key4)
    
    # Run all methods
    print("\nComparing different attention methods:")
    import time
    
    for name, method_fn in methods:
        print(f"\n{name}:")
        # Measure execution time
        start_time = time.time()
        results = method_fn()
        execution_time = time.time() - start_time
        
        # Store results
        all_results.append(results)
        execution_times.append(execution_time)
        
        # Calculate errors
        method_errors = []
        for q in range(n_queries):
            exact = exact_results[q]
            approx = results[q]
            
            # Ensure shapes match for error calculation
            if exact.shape != approx.shape:
                print(f"  Warning: Shape mismatch - exact: {exact.shape}, approx: {approx.shape}")
                exact_flat = exact.flatten()
                approx_flat = approx.flatten()
                min_len = min(len(exact_flat), len(approx_flat))
                exact_flat = exact_flat[:min_len]
                approx_flat = approx_flat[:min_len]
                error = jnp.sqrt(jnp.sum((exact_flat - approx_flat)**2)) / jnp.sqrt(jnp.sum(exact_flat**2))
            else:
                error = jnp.sqrt(jnp.sum((exact - approx)**2)) / jnp.sqrt(jnp.sum(exact**2))
            
            method_errors.append(error)
        
        mean_error = jnp.mean(jnp.array(method_errors))
        errors.append(mean_error)
        
        # Print statistics
        print(f"  Execution time: {execution_time:.6f} seconds ({n_queries/execution_time:.1f} queries/sec)")
        print(f"  Mean relative error: {mean_error:.6f}")
    
    # Print summary comparison
    print("\nMethod comparison summary:")
    print(f"{'Method':<20} {'Error':<10} {'Relative Speed':<15} {'Queries/sec':<12}")
    print("-" * 60)
    
    baseline_time = execution_times[0]  # Use deterministic as baseline
    for i, (name, _) in enumerate(methods):
        relative_speed = baseline_time / execution_times[i]
        queries_per_sec = n_queries / execution_times[i]
        print(f"{name:<20} {errors[i]:<10.6f} {relative_speed:<15.2f}x {queries_per_sec:<12.1f}")
    
    # Create error vs. temperature plot (skip the deterministic method)
    stochastic_temps = [0.1, 0.5, 1.0, 2.0, 5.0]
    stochastic_errors = errors[1:]  # Skip deterministic
    
    plt.figure(figsize=(10, 6))
    plt.plot(stochastic_temps, stochastic_errors, 'o-', linewidth=2)
    plt.axhline(y=errors[0], color='r', linestyle='--', label=f'Deterministic Error ({errors[0]:.6f})')
    plt.xlabel('Temperature')
    plt.ylabel('Mean Relative Error')
    plt.title('Effect of Temperature on Stochastic Attention Error')
    plt.grid(True)
    plt.legend()
    plt.xscale('log')  # Log scale for temperature
    plt.savefig('stochastic_attention_errors.png', dpi=300)
    
    print(f"\nDone! Error vs. temperature plot saved to 'stochastic_attention_errors.png'")
    print(f"Key observations:")
    
    if min(errors[1:]) < errors[0]:
        best_temp_idx = np.argmin(errors[1:])
        best_temp = stochastic_temps[best_temp_idx]
        improvement = (1 - min(errors[1:]) / errors[0]) * 100
        print(f"- Stochastic sampling with T={best_temp} improves accuracy by {improvement:.2f}%")
    else:
        print(f"- Deterministic selection provides the best accuracy in this test case")
    
    # Check if there's a trend in errors vs temperature
    if errors[1] < errors[-1]:
        print(f"- Lower temperatures perform better (more exploitation)")
    elif errors[-1] < errors[1]:
        print(f"- Higher temperatures perform better (more exploration)")
    else:
        print(f"- No clear trend between temperature and error")
    
    # Compare speed
    if min(execution_times[1:]) < execution_times[0]:
        print(f"- Stochastic approach can be faster than deterministic in some cases")
    else:
        print(f"- Deterministic approach is faster than stochastic sampling")
    
    print("\nNote: For gradient quality during training, the stochastic approach may provide")
    print("benefits beyond what is measured by simple error metrics, particularly for")
    print("avoiding local minima and improving generalization.")


def run_adaptive_stochastic_attention_demo():
    """Compare regular stochastic sampling with adaptive residual weighting."""
    # Set random seed for reproducibility
    key = PRNGKey(48)
    
    # Generate data for demonstration
    dim = 64  # Vector dimension
    n_samples = 2**10  # 1024 points
    levels = 5  # 2^5 = 32 clusters
    
    print(f"Running adaptive vs. standard stochastic attention comparison")
    print(f"Using {n_samples} vectors of dimension {dim}, with {2**levels} clusters")
    
    # Generate keys with more concentrated distribution to create higher attention variability
    # We'll create several concentrated clusters of keys to simulate more realistic attention patterns
    key1, key2, key3, key4, key5 = jax.random.split(key, 5)
    
    # Number of synthetic clusters to generate
    n_synthetic_clusters = 8
    points_per_cluster = n_samples // n_synthetic_clusters
    leftover_points = n_samples - (points_per_cluster * n_synthetic_clusters)
    
    # Generate synthetic cluster centers
    cluster_centers = jax.random.normal(key1, (n_synthetic_clusters, dim))
    cluster_centers = cluster_centers / jnp.sqrt(jnp.sum(cluster_centers**2, axis=1, keepdims=True))
    
    # Generate concentration factors (higher = more concentrated clusters)
    # Using different concentration levels creates more variability in attention patterns
    concentrations = jnp.array([5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0])
    
    # Function to generate vMF-distributed points (von Mises-Fisher)
    def generate_vmf_points(center, kappa, n_points, key):
        # Generate points concentrated around center with given kappa (concentration)
        center_dim = center.shape[0]
        
        # Step 1: Generate samples from a uniform distribution on a hypersphere cap
        key1, key2 = jax.random.split(key)
        
        # Sample from the angular distribution
        w = 1.0 + jnp.log(jax.random.uniform(key1, (n_points,))) / kappa
        
        # Generate random direction vectors orthogonal to center
        v = jax.random.normal(key2, (n_points, center_dim))
        
        # Make v orthogonal to center - properly vectorized version
        # Need to properly reshape center for broadcasting with each row of v
        center_reshaped = center.reshape(1, -1)  # Shape: (1, center_dim)
        dot_products = jnp.sum(v * center_reshaped, axis=1, keepdims=True)  # Shape: (n_points, 1)
        v = v - dot_products * center_reshaped  # Now broadcasts correctly
        
        # Normalize v - avoid division by zero
        v_norms = jnp.sqrt(jnp.sum(v**2, axis=1, keepdims=True))
        v = v / jnp.maximum(v_norms, 1e-10)  # Safe normalization
        
        # Combine to get the result (points on sphere concentrated around center)
        x = w[:, jnp.newaxis] * center_reshaped + jnp.sqrt(1 - w[:, jnp.newaxis]**2) * v
        
        return x
    
    # Generate points for each synthetic cluster with different concentrations
    all_keys = []
    cluster_keys = []
    
    for i in range(n_synthetic_clusters):
        n_points = points_per_cluster + (1 if i < leftover_points else 0)
        key5, subkey = jax.random.split(key5)
        
        # Generate concentrated points
        cluster_points = generate_vmf_points(
            cluster_centers[i], concentrations[i], n_points, subkey)
        
        all_keys.append(cluster_points)
        cluster_keys.extend([i] * n_points)  # Track which synthetic cluster each point belongs to
    
    # Combine all clusters
    keys = jnp.vstack(all_keys)
    
    # Double-check normalization
    keys = keys / jnp.sqrt(jnp.sum(keys**2, axis=1, keepdims=True))
    
    print(f"  Created {n_synthetic_clusters} synthetic clusters with varying concentrations:")
    for i in range(n_synthetic_clusters):
        print(f"    Cluster {i+1}: concentration={concentrations[i]:.1f}")
    
    # Generate random values
    values = jax.random.normal(key2, (n_samples, 32))  # 32-dim values
    
    # Generate test queries
    # Generate queries that are biased toward some of the synthetic clusters
    # to create situations where attention is more concentrated
    n_queries = 10  # More queries for better statistics
    
    # Mix of random queries and queries near cluster centers
    query_centers = cluster_centers[jnp.array([0, 2, 4, 7])]  # Select a subset of clusters
    
    # Generate some queries near the selected centers and some random ones
    near_center_queries = []
    for i, center in enumerate(query_centers):
        key3, subkey = jax.random.split(key3)
        # Generate 2 queries near each selected center with moderate concentration
        near_center_points = generate_vmf_points(center, 20.0, 2, subkey)
        near_center_queries.append(near_center_points)
    
    # 2 random queries
    key3, subkey = jax.random.split(key3)
    random_queries = jax.random.normal(subkey, (2, dim))
    random_queries = random_queries / jnp.sqrt(jnp.sum(random_queries**2, axis=1, keepdims=True))
    
    # Combine queries
    queries = jnp.vstack([jnp.vstack(near_center_queries), random_queries])
    
    # Ensure we have exactly n_queries
    queries = queries[:n_queries]
    
    # Ensure unit length
    queries = queries / jnp.sqrt(jnp.sum(queries**2, axis=1, keepdims=True))
    
    # Cluster the keys and values together
    print(f"Performing hierarchical clustering (levels={levels})...")
    centers, log_weights, weighted_means, clustered_keys, clustered_values = cluster_spherical(
        keys, levels=levels, vs=values)
    
    # Print tree statistics
    n_clusters = 2**levels
    print(f"  Tree structure: {n_clusters} leaf clusters, {centers.shape[0]} total nodes")
    
    # Compute exact attention for comparison (baseline)
    print("Computing exact attention (baseline)...")
    exact_results = []
    for q in range(n_queries):
        query = queries[q]
        scores = jnp.exp(jnp.dot(keys, query))
        weighted_sum = jnp.sum(scores[:, None] * values, axis=0)
        normalization = jnp.sum(scores)
        exact_results.append(weighted_sum / normalization)
    
    exact_results = jnp.stack(exact_results)
    
    # Temperature values to test
    temps = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    # Initialize results storage
    standard_errors = []
    adaptive_errors = []
    standard_times = []
    adaptive_times = []
    mass_ratios = []  # Average mass in best cluster
    max_mass_ratios = []  # Maximum mass in best cluster
    top_two_mass_ratios = []  # Average mass in top two clusters
    
    # First do a warmup run to compile everything
    print("\nWarming up JIT compilation...")
    batched_stochastic_approx_qkv_attention(
        queries[:1], centers, log_weights, clustered_keys, clustered_values, weighted_means, 1.0, key4)
    batched_adaptive_stochastic_approx_qkv_attention(
        queries[:1], centers, log_weights, clustered_keys, clustered_values, weighted_means, 1.0, key4)
    
    # Import time for measurements
    import time
    
    # Run tests for all temperature values
    print("\nRunning comparisons across temperature values:")
    
    for temp in temps:
        print(f"\nTemperature: {temp}")
        
        # Create a new key for this temperature
        temp_key = jax.random.fold_in(key4, int(temp * 100))
        
        # 1. Test standard stochastic approach
        start_time = time.time()
        standard_results = batched_stochastic_approx_qkv_attention(
            queries, centers, log_weights, clustered_keys, clustered_values, 
            weighted_means, temp, temp_key)
        standard_time = time.time() - start_time
        standard_times.append(standard_time)
        
        # 2. Test adaptive stochastic approach
        start_time = time.time()
        adaptive_results = batched_adaptive_stochastic_approx_qkv_attention(
            queries, centers, log_weights, clustered_keys, clustered_values, 
            weighted_means, temp, temp_key)
        adaptive_time = time.time() - start_time
        adaptive_times.append(adaptive_time)
        
        # Calculate errors for each query
        standard_query_errors = []
        adaptive_query_errors = []
        
        for q in range(n_queries):
            exact = exact_results[q]
            std_approx = standard_results[q]
            adapt_approx = adaptive_results[q]
            
            # Compute relative errors
            std_error = jnp.sqrt(jnp.sum((exact - std_approx)**2)) / jnp.sqrt(jnp.sum(exact**2))
            adapt_error = jnp.sqrt(jnp.sum((exact - adapt_approx)**2)) / jnp.sqrt(jnp.sum(exact**2))
            
            standard_query_errors.append(std_error)
            adaptive_query_errors.append(adapt_error)
        
        # Calculate average errors
        standard_error = jnp.mean(jnp.array(standard_query_errors))
        adaptive_error = jnp.mean(jnp.array(adaptive_query_errors))
        
        standard_errors.append(standard_error)
        adaptive_errors.append(adaptive_error)
        
        # Print results for this temperature
        speed_ratio = standard_time / adaptive_time
        error_improvement = (1 - adaptive_error / standard_error) * 100
        print(f"  Standard: Error={standard_error:.6f}, Time={standard_time:.6f}s")
        print(f"  Adaptive: Error={adaptive_error:.6f}, Time={adaptive_time:.6f}s")
        print(f"  Improvement: {error_improvement:.2f}% error reduction")
        print(f"  Speed: {speed_ratio:.2f}x ({adaptive_time/standard_time:.2f}x)")
        
        # Calculate attention mass capture statistics for this temperature
        # Check attention mass distribution for all queries
        leaf_start_idx = centers.shape[0] - clustered_keys.shape[0]
        leaf_nodes = jnp.arange(clustered_keys.shape[0]) + leaf_start_idx
        
        query_mass_ratios = []
        query_best_two_ratios = []
        
        for q_idx in range(n_queries):
            query = queries[q_idx]
            
            # Calculate leaf scores for this query
            def score_node(node_idx):
                centroid = centers[node_idx]
                return log_expected_query_mass(query, centroid) + log_weights[node_idx]
            
            leaf_scores = jax.vmap(score_node)(leaf_nodes)
            
            # Find the best leaf
            best_leaf_idx = jnp.argmax(leaf_scores)
            
            # Convert scores to probabilities (numerically stable)
            exp_leaf_scores = jnp.exp(leaf_scores - jnp.max(leaf_scores))
            total_leaf_mass = jnp.sum(exp_leaf_scores)
            
            # Estimate mass captured by best cluster
            best_mass_ratio = exp_leaf_scores[best_leaf_idx] / total_leaf_mass
            query_mass_ratios.append(best_mass_ratio)
            
            # Also calculate mass captured by top two clusters
            sorted_indices = jnp.argsort(-leaf_scores)  # Sort in descending order
            top_two_indices = sorted_indices[:2]
            top_two_mass = (exp_leaf_scores[top_two_indices[0]] + 
                           exp_leaf_scores[top_two_indices[1]]) / total_leaf_mass
            query_best_two_ratios.append(top_two_mass)
        
        # Average and max ratios across queries
        avg_best_mass_ratio = jnp.mean(jnp.array(query_mass_ratios)) 
        max_best_mass_ratio = jnp.max(jnp.array(query_mass_ratios))
        avg_best_two_ratio = jnp.mean(jnp.array(query_best_two_ratios))
        
        # Store various metrics for plotting
        mass_ratios.append(avg_best_mass_ratio)
        max_mass_ratios.append(max_best_mass_ratio)
        top_two_mass_ratios.append(avg_best_two_ratio)
        
        # Analyze the attention distribution per query
        gini_coefficients = []
        effective_clusters = []
        
        for q_idx in range(n_queries):
            query = queries[q_idx]
            
            # Get leaf scores for this query
            def score_node(node_idx):
                centroid = centers[node_idx]
                return log_expected_query_mass(query, centroid) + log_weights[node_idx]
            
            leaf_scores = jax.vmap(score_node)(leaf_nodes)
            exp_leaf_scores = jnp.exp(leaf_scores - jnp.max(leaf_scores))
            probs = exp_leaf_scores / jnp.sum(exp_leaf_scores)
            
            # Sort probabilities in descending order for analysis
            sorted_probs = jnp.sort(probs)[::-1]
            
            # Calculate Gini coefficient (measure of inequality)
            # Higher values mean more concentrated distribution
            n = len(sorted_probs)
            indices = jnp.arange(1, n+1)
            gini = 1 - 2 * jnp.sum((n + 1 - indices) * sorted_probs) / (n * jnp.sum(sorted_probs))
            gini_coefficients.append(gini)
            
            # Calculate effective number of clusters (1/sum of squared probs)
            effective_n = 1.0 / jnp.sum(probs**2)
            effective_clusters.append(effective_n)
        
        avg_gini = jnp.mean(jnp.array(gini_coefficients))
        avg_effective_n = jnp.mean(jnp.array(effective_clusters))
        
        print(f"  Attention concentration statistics:")
        print(f"    Best cluster mass: avg={avg_best_mass_ratio:.2%}, max={max_best_mass_ratio:.2%}")
        print(f"    Top 2 clusters mass: avg={avg_best_two_ratio:.2%}")
        print(f"    Gini coefficient: {avg_gini:.4f} (higher = more concentrated)")
        print(f"    Effective # of clusters: {avg_effective_n:.1f} of {n_clusters} total")
    
    # Create comparison plot
    plt.figure(figsize=(12, 8))
    
    # Error subplot
    plt.subplot(2, 1, 1)
    plt.plot(temps, standard_errors, 'o-', label="Standard Stochastic", linewidth=2)
    plt.plot(temps, adaptive_errors, 's-', label="Adaptive Residual Weighting", linewidth=2)
    plt.xlabel("Temperature")
    plt.ylabel("Mean Relative Error")
    plt.title("Error Comparison: Standard vs. Adaptive Stochastic Attention")
    plt.grid(True)
    plt.legend()
    plt.xscale('log')
    
    # Speed subplot
    plt.subplot(2, 1, 2)
    relative_speedup = [standard_times[i] / adaptive_times[i] for i in range(len(temps))]
    plt.plot(temps, relative_speedup, 'o-', color='green', linewidth=2)
    plt.axhline(y=1.0, color='r', linestyle='--', label='Equal Speed')
    plt.xlabel("Temperature")
    plt.ylabel("Relative Speed (>1 means Adaptive is faster)")
    plt.title("Speed Comparison: Adaptive vs. Standard (ratio of execution times)")
    plt.grid(True)
    plt.xscale('log')
    
    plt.tight_layout()
    plt.savefig('adaptive_vs_standard_comparison.png', dpi=300)
    
    # Create a second plot showing error reduction and attention mass metrics
    plt.figure(figsize=(10, 6))
    
    # Primary axis: Error reduction
    error_reduction = [(1 - adaptive_errors[i] / standard_errors[i]) * 100 for i in range(len(temps))]
    plt.plot(temps, error_reduction, 'o-', color='blue', linewidth=2, label="Error Reduction")
    
    # Secondary axis: Attention mass distributions
    ax2 = plt.gca().twinx()
    ax2.plot(temps, [m * 100 for m in mass_ratios], 's-', color='red', linewidth=2, label="Avg Best Cluster Mass")
    ax2.plot(temps, [m * 100 for m in max_mass_ratios], '^-', color='darkred', linewidth=2, label="Max Best Cluster Mass")
    ax2.plot(temps, [m * 100 for m in top_two_mass_ratios], 'd-', color='orange', linewidth=2, label="Avg Top-2 Clusters Mass")
    
    # Labels and formatting
    plt.xlabel("Temperature")
    plt.ylabel("Error Reduction (%)")
    ax2.set_ylabel("Attention Mass (%)")
    plt.title("Adaptive Weighting: Error Reduction vs. Attention Mass Distribution")
    plt.grid(True)
    plt.xscale('log')
    
    # Combined legend
    lines1, labels1 = plt.gca().get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    plt.tight_layout()
    plt.savefig('adaptive_weighting_analysis.png', dpi=300)
    
    # Create a third plot focusing on attention concentration patterns
    plt.figure(figsize=(12, 10))
    
    # Create a 2x2 subplot grid
    plt.subplot(2, 2, 1)
    plt.plot(temps, error_reduction, 'o-', color='blue', linewidth=2)
    plt.xlabel("Temperature")
    plt.ylabel("Error Reduction (%)")
    plt.title("Error Reduction vs Temperature")
    plt.grid(True)
    plt.xscale('log')
    
    # Plot mass in best cluster
    plt.subplot(2, 2, 2)
    plt.plot(temps, [m * 100 for m in mass_ratios], 's-', color='red', linewidth=2, label="Avg")
    plt.plot(temps, [m * 100 for m in max_mass_ratios], '^--', color='darkred', linewidth=2, label="Max")
    plt.xlabel("Temperature")
    plt.ylabel("Mass in Best Cluster (%)")
    plt.title("Attention Concentration")
    plt.legend()
    plt.grid(True)
    plt.xscale('log')
    
    # Plot correlation between error reduction and mass ratio
    plt.subplot(2, 2, 3)
    plt.scatter([m * 100 for m in mass_ratios], error_reduction, c='purple', s=100, alpha=0.7)
    
    # Add temperature labels to points
    for i, temp in enumerate(temps):
        plt.annotate(f'T={temp}', 
                    (mass_ratios[i] * 100, error_reduction[i]),
                    xytext=(5, 5), textcoords='offset points')
    
    # Add trend line
    z = np.polyfit([m * 100 for m in mass_ratios], error_reduction, 1)
    p = np.poly1d(z)
    plt.plot([min(m * 100 for m in mass_ratios), max(m * 100 for m in mass_ratios)], 
             [p(min(m * 100 for m in mass_ratios)), p(max(m * 100 for m in mass_ratios))], 
             "r--", alpha=0.7)
    
    plt.xlabel("Avg Mass in Best Cluster (%)")
    plt.ylabel("Error Reduction (%)")
    plt.title("Correlation: Error Reduction vs Attention Concentration")
    plt.grid(True)
    
    # Plot mass in top two clusters
    plt.subplot(2, 2, 4)
    plt.plot(temps, [m * 100 for m in top_two_mass_ratios], 'd-', color='orange', linewidth=2)
    plt.xlabel("Temperature")
    plt.ylabel("Mass in Top 2 Clusters (%)")
    plt.title("Top-2 Clusters Attention Concentration")
    plt.grid(True)
    plt.xscale('log')
    
    plt.tight_layout()
    plt.savefig('attention_concentration_analysis.png', dpi=300)
    
    # Calculate average improvements
    avg_error_reduction = sum(error_reduction) / len(error_reduction)
    avg_speedup = sum(relative_speedup) / len(relative_speedup)
    
    print("\nSummary Statistics:")
    print(f"  Average error reduction: {avg_error_reduction:.2f}%")
    print(f"  Average speed ratio: {avg_speedup:.2f}x")
    print(f"  Best temperature for adaptive method: {temps[np.argmin(adaptive_errors)]}")
    print(f"  Best temperature for standard method: {temps[np.argmin(standard_errors)]}")
    
    # Theoretical analysis
    print("\nTheoretical Analysis:")
    print(f"  Average mass in best cluster: {sum(mass_ratios)/len(mass_ratios):.2%}")
    
    # Maximum possible error reduction
    max_reduction = max(error_reduction)
    max_reduction_temp = temps[np.argmax(error_reduction)]
    print(f"  Maximum error reduction: {max_reduction:.2f}% at temperature {max_reduction_temp}")
    
    # Correlation between mass ratio and error reduction
    corr = np.corrcoef(mass_ratios, error_reduction)[0, 1]
    print(f"  Correlation between mass ratio and error reduction: {corr:.4f}")
    
    if corr < -0.5:
        print("  Strong negative correlation: Adaptive weighting is most effective")
        print("  when attention mass is distributed across many clusters")
    elif corr > 0.5:
        print("  Strong positive correlation: Adaptive weighting is most effective")
        print("  when attention mass is concentrated in a few clusters")
    
    print(f"\nDone! Comparison plots saved to 'adaptive_vs_standard_comparison.png' and 'adaptive_weighting_analysis.png'")


def run_simplified_jax_attention_demo():
    """Demonstrate the JAX-optimized version of hierarchical attention."""
    # Set random seed for reproducibility
    key = PRNGKey(46)
    
    # Generate data for demonstration
    dim = 64  # Vector dimension
    n_samples = 2**10  # 1024 points
    levels = 5  # 2^5 = 32 clusters
    
    print(f"Running JAX-optimized simplified hierarchical attention demo")
    print(f"Using {n_samples} vectors of dimension {dim}, with {2**levels} clusters")
    
    # Generate keys with varying magnitudes
    key1, key2, key3 = jax.random.split(key, 3)
    keys = jax.random.normal(key1, (n_samples, dim))
    keys = keys / jnp.sqrt(jnp.sum(keys**2, axis=1, keepdims=True))  # Normalize
    
    # Generate random values
    values = jax.random.normal(key2, (n_samples, 32))  # 32-dim values
    
    # Generate a few test queries
    n_queries = 5
    queries = jax.random.normal(key3, (n_queries, dim))
    queries = queries / jnp.sqrt(jnp.sum(queries**2, axis=1, keepdims=True))  # Normalize
    
    # Cluster the keys and values together
    print(f"Performing hierarchical clustering (levels={levels})...")
    centers, log_weights, weighted_means, clustered_keys, clustered_values = cluster_spherical(
        keys, levels=levels, vs=values)
    
    # Print tree statistics
    n_clusters = 2**levels
    print(f"  Tree structure: {n_clusters} leaf clusters, {centers.shape[0]} total nodes")
    
    # Compute exact attention for comparison (baseline)
    print("Computing exact attention (baseline)...")
    exact_results = []
    for q in range(n_queries):
        query = queries[q]
        scores = jnp.exp(jnp.dot(keys, query))
        weighted_sum = jnp.sum(scores[:, None] * values, axis=0)
        normalization = jnp.sum(scores)
        exact_results.append(weighted_sum / normalization)
    
    exact_results = jnp.stack(exact_results)
    
    # Run standard beam search with residual (beam=1)
    print("Computing standard beam search with residual approximation (beam=1)...")
    std_beam1_results = batched_approx_qkv_attention(
        queries, centers, log_weights, clustered_keys, clustered_values,
        weighted_means=weighted_means, beam_width=1)
    
    # Compile and run simplified JAX implementation
    print("Compiling and running JAX-optimized implementation...")
    print("  First run includes compilation time...")
    import time
    start_time = time.time()
    jax_results = batched_simple_approx_qkv_attention(
        queries, centers, log_weights, clustered_keys, clustered_values, weighted_means)
    compile_time = time.time() - start_time
    print(f"  Compilation + first execution time: {compile_time:.4f} seconds")
    
    # Run again to measure execution time
    print("  Measuring execution time (post-compilation)...")
    start_time = time.time()
    jax_results = batched_simple_approx_qkv_attention(
        queries, centers, log_weights, clustered_keys, clustered_values, weighted_means)
    execution_time = time.time() - start_time
    print(f"  Execution time: {execution_time:.4f} seconds ({n_queries/execution_time:.1f} queries/sec)")
    
    # Compute errors for both implementations
    std_errors = []
    jax_errors = []
    
    for q in range(n_queries):
        exact = exact_results[q]
        std_approx = std_beam1_results[q]
        jax_approx = jax_results[q]
        
        # Ensure shapes match for error calculation
        if exact.shape != std_approx.shape or exact.shape != jax_approx.shape:
            print(f"Warning: Shape mismatch - exact: {exact.shape}, std: {std_approx.shape}, jax: {jax_approx.shape}")
            # Use flattened arrays with minimum length
            exact_flat = exact.flatten()
            std_flat = std_approx.flatten()
            jax_flat = jax_approx.flatten()
            min_len = min(len(exact_flat), len(std_flat), len(jax_flat))
            
            # Compute errors on truncated flat arrays
            std_error = jnp.sqrt(jnp.sum((exact_flat[:min_len] - std_flat[:min_len])**2)) / jnp.sqrt(jnp.sum(exact_flat[:min_len]**2))
            jax_error = jnp.sqrt(jnp.sum((exact_flat[:min_len] - jax_flat[:min_len])**2)) / jnp.sqrt(jnp.sum(exact_flat[:min_len]**2))
        else:
            # Compute errors normally
            std_error = jnp.sqrt(jnp.sum((exact - std_approx)**2)) / jnp.sqrt(jnp.sum(exact**2))
            jax_error = jnp.sqrt(jnp.sum((exact - jax_approx)**2)) / jnp.sqrt(jnp.sum(exact**2))
        
        std_errors.append(std_error)
        jax_errors.append(jax_error)
    
    # Print error comparison
    mean_std_error = jnp.mean(jnp.array(std_errors))
    mean_jax_error = jnp.mean(jnp.array(jax_errors))
    
    print("\nError comparison:")
    print(f"  Standard beam=1 with residual: mean relative error = {mean_std_error:.6f}")
    print(f"  JAX-optimized implementation: mean relative error = {mean_jax_error:.6f}")
    
    if mean_jax_error < mean_std_error:
        improvement = (1 - mean_jax_error/mean_std_error) * 100
        print(f"  JAX implementation is more accurate by {improvement:.2f}%")
    elif mean_std_error < mean_jax_error:
        degradation = (mean_jax_error/mean_std_error - 1) * 100
        print(f"  JAX implementation is less accurate by {degradation:.2f}%")
    else:
        print("  Both implementations have identical accuracy")
    
    # Computational comparison
    print("\nComputational comparison:")
    print(f"  Total keys processed in exact attention: {n_samples}")
    print(f"  Keys processed in simplified attention: {clustered_keys.shape[1]}")
    print(f"  Speedup factor from keys alone: {n_samples/clustered_keys.shape[1]:.1f}x")
    print(f"  Additional speedup from JAX optimization and residual approximation")
    
    print(f"\nDone! Demonstrated JAX-optimized hierarchical attention")


if __name__ == "__main__":
    # Uncomment the function you want to run
    # main()
    #run_simplified_jax_attention_demo()
    #run_stochastic_attention_demo()
    run_adaptive_stochastic_attention_demo()
