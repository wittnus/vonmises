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
from lib import cluster, cluster_spherical

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
    centers, clustered_data = cluster_spherical(data, levels=levels)
    
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

def main():
    # Run standard clustering on unit vectors
    run_standard_clustering()
    
    # Run weighted clustering on vectors with varying magnitudes
    run_weighted_clustering()

if __name__ == "__main__":
    main()
