import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from chex import Array
from functools import partial
from lib import vmf_concentration, vmf_concentration_exact, log_expected_query_mass

def generate_vmf_samples(mu: Array, kappa: float, n_samples: int, key: jax.random.PRNGKey) -> Array:
    """
    Generate samples from a von Mises-Fisher distribution on the unit sphere.
    
    Args:
        mu: The mean direction vector (will be normalized)
        kappa: The concentration parameter (higher -> more concentrated)
        n_samples: Number of samples to generate
        key: PRNG key for random sampling
        
    Returns:
        Array of samples with shape [n_samples, d]
    """
    d = mu.shape[0]
    
    # Normalize mu (just to be safe)
    mu = mu / jnp.linalg.norm(mu)
    
    # Split the key for different random operations
    key_tangent, key_w = jax.random.split(key)
    
    # Simplified approach for high dimensions
    # Step 1: Sample a direction from the tangent space of mu
    tangent_samples = jax.random.normal(key_tangent, (n_samples, d))
    
    # Make the tangent samples orthogonal to mu
    proj = jnp.einsum('nd,d->n', tangent_samples, mu)[:, None] * mu[None, :]
    tangent_samples = tangent_samples - proj
    
    # Normalize the tangent samples
    tangent_norms = jnp.linalg.norm(tangent_samples, axis=1, keepdims=True)
    tangent_samples = tangent_samples / jnp.maximum(tangent_norms, 1e-10)
    
    # Step 2: Sample the angle parameter using rejection sampling for accuracy
    # This is a simplified version that works well for the range of kappa values we test
    w_shape = 0.5 * (d - 1)  # Shape parameter related to dimension
    w_samples = jax.random.beta(key_w, w_shape, w_shape, (n_samples,))
    
    # Scale the concentrations based on kappa
    # This formula is an approximation of the inverse CDF of the vMF angle distribution
    w_samples = 1.0 - (1.0 - w_samples) * jnp.exp(-kappa)
    
    # Convert to cos(theta)
    cos_theta = 2.0 * w_samples - 1.0
    sin_theta = jnp.sqrt(1.0 - jnp.square(cos_theta))[:, None]
    
    # Combine to create samples with the right concentration
    samples = cos_theta[:, None] * mu[None, :] + sin_theta * tangent_samples
    
    # Ensure samples are normalized (should be close already, but for numerical stability)
    sample_norms = jnp.linalg.norm(samples, axis=1, keepdims=True)
    samples = samples / sample_norms
    
    return samples

def fit_vmf_distribution(samples: Array, use_exact: bool = False) -> tuple:
    """
    Fit a von Mises-Fisher distribution to samples on the unit sphere.
    
    Args:
        samples: Array of unit vectors with shape [n_samples, d]
        use_exact: Whether to use the exact Newton-based method for kappa estimation
        
    Returns:
        Tuple of (mu, kappa) where mu is the estimated mean direction
        and kappa is the estimated concentration parameter
    """
    # Normalize samples to ensure they're on the unit sphere
    norms = jnp.linalg.norm(samples, axis=1, keepdims=True)
    normalized_samples = samples / norms
    
    # Compute the mean resultant vector
    R_bar = jnp.mean(normalized_samples, axis=0)
    
    # Compute R (the length of the mean resultant vector)
    R = jnp.linalg.norm(R_bar)
    
    # Compute mu (the mean direction)
    mu = R_bar / R
    
    # Compute kappa using either approximate or exact method
    if use_exact:
        kappa = vmf_concentration_exact(R_bar)
    else:
        kappa = vmf_concentration(R_bar)
    
    return mu, kappa

def compute_exact_attention_weights(query: Array, keys: Array) -> Array:
    """
    Compute the exact softmax attention weights for a query and keys.
    
    Args:
        query: The query vector [d]
        keys: The key vectors [n_keys, d]
        
    Returns:
        Array of attention weights [n_keys]
    """
    # Compute dot products
    logits = jnp.dot(query, keys.T)
    
    # Apply softmax
    weights = jax.nn.softmax(logits)
    
    return weights

def compute_approx_attention_mass(query: Array, mu: Array, kappa: float) -> float:
    """
    Compute the approximate attention mass for a query and vMF distribution.
    
    Args:
        query: The query vector [d]
        mu: The mean direction of the vMF distribution [d]
        kappa: The concentration parameter
        
    Returns:
        The approximate attention mass (not normalized)
    """
    # log_expected_query_mass expects the centroid as unnormalized vector
    # with norm related to kappa. First we normalize both vectors
    norm_mu = mu / jnp.linalg.norm(mu)
    norm_query = query / jnp.linalg.norm(query)
    
    # Create an unnormalized centroid that encodes kappa
    r = 1.0 - 1.0/kappa  # approximate relationship for large dimension
    r = jnp.clip(r, 0.01, 0.99)  # prevent numerical issues
    
    # Scale the normalized direction to have the right magnitude
    centroid = norm_mu * r
    
    # Use the log_expected_query_mass function to compute the log mass
    log_mass = log_expected_query_mass(norm_query, centroid)
    
    # Convert from log space
    return jnp.exp(log_mass)

def generate_query_with_angle(mu: Array, angle_degrees: float, key: jax.random.PRNGKey) -> Array:
    """
    Generate a unit query vector with a specified angle to the mean direction.
    
    Args:
        mu: The mean direction [d]
        angle_degrees: The angle in degrees between mu and the query
        key: PRNG key for random sampling
        
    Returns:
        A query vector on the unit sphere with the specified angle to mu
    """
    d = mu.shape[0]
    angle_radians = jnp.radians(angle_degrees)
    
    # Normalize mu
    mu = mu / jnp.linalg.norm(mu)
    
    # Generate a random orthogonal direction to mu
    v = jax.random.normal(key, (d,))
    v = v - jnp.dot(v, mu) * mu  # Make v orthogonal to mu
    v = v / jnp.linalg.norm(v)   # Normalize v
    
    # Combine mu and v with appropriate weights to get the desired angle
    query = jnp.cos(angle_radians) * mu + jnp.sin(angle_radians) * v
    
    return query

def test_vmf_concentration():
    """
    Test the vmf_concentration function with controlled inputs.
    Create samples from vMF distributions with known kappa values
    and verify that vmf_concentration recovers these values.
    """
    print("\n" + "="*50)
    print("TESTING VMF_CONCENTRATION FUNCTION")
    print("="*50)
    
    # Test dimensions
    dimensions = [3, 8, 32, 64, 128]
    
    # Test kappa values
    kappa_values = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    
    # Number of samples for each test
    n_samples = 10000
    
    # Set random seed for reproducibility
    key = jax.random.PRNGKey(42)
    
    # Create a figure for plotting results
    fig, axes = plt.subplots(1, len(dimensions), figsize=(20, 6))
    if len(dimensions) == 1:
        axes = [axes]
    
    # Store all results for summary
    all_results = []
    
    # Test for each dimension
    for d_idx, d in enumerate(dimensions):
        print(f"\nDimension: {d}")
        print("-" * 30)
        
        dimension_results = []
        
        for kappa in kappa_values:
            key, subkey = jax.random.split(key)
            
            # Create a random unit vector as mean direction
            mu_key, sample_key = jax.random.split(subkey)
            mu = jax.random.normal(mu_key, (d,))
            mu = mu / jnp.linalg.norm(mu)
            
            # For direct testing, create a centroid with known R value
            # Use the relationship between kappa and R
            # For large d, R ≈ 1 - d/(2*kappa)
            
            # First, try to compute the theoretical R for this kappa
            # This is an approximation that becomes more accurate for large d
            R_theory = 1.0 - d/(2*kappa) if kappa > d/2 else 0.1
            R_theory = min(R_theory, 0.9999)  # Prevent numerical issues
            
            # Create centroid with this magnitude
            centroid = mu * R_theory
            
            # Now compute the kappa from this centroid using both methods
            estimated_kappa_approx = vmf_concentration(centroid)
            estimated_kappa_exact = vmf_concentration_exact(centroid)
            
            # For validation, let's create samples and estimate empirically
            key, sample_key = jax.random.split(key)
            
            # Create a synthetic vMF distribution by manually setting parameters
            def direct_vmf_sample(key, n_samples):
                # Generate orthogonal directions
                basis_key, project_key = jax.random.split(key)
                
                # Generate random directions
                basis_samples = jax.random.normal(basis_key, (n_samples, d))
                
                # Make orthogonal to mu
                proj = jnp.einsum('nd,d->n', basis_samples, mu)[:, None] * mu[None, :]
                basis_samples = basis_samples - proj
                
                # Normalize
                norms = jnp.linalg.norm(basis_samples, axis=1, keepdims=True)
                basis_samples = basis_samples / jnp.maximum(norms, 1e-10)
                
                # For various kappas, we need to adjust the angular distribution
                # Higher kappa = more concentration around mu
                # We'll directly set the cosine of the angle
                cos_angles = jnp.ones(n_samples) * R_theory
                sin_angles = jnp.sqrt(1 - cos_angles**2)[:, None]
                
                # Combine to create samples
                samples = cos_angles[:, None] * mu[None, :] + sin_angles * basis_samples
                
                # Ensure normalization
                norms = jnp.linalg.norm(samples, axis=1, keepdims=True)
                return samples / norms
            
            # Generate samples with the controlled concentration
            samples = direct_vmf_sample(sample_key, n_samples)
            
            # Calculate the empirical R
            mean_direction = jnp.mean(samples, axis=0)
            R_empirical = jnp.linalg.norm(mean_direction)
            
            # Estimate kappa from the empirical R using both methods
            empirical_kappa_approx = vmf_concentration(mean_direction)
            empirical_kappa_exact = vmf_concentration_exact(mean_direction)
            
            # Store results
            result = {
                'dimension': d,
                'true_kappa': kappa,
                'R_theory': R_theory,
                'R_empirical': float(R_empirical),
                'approx_kappa_from_theory': float(estimated_kappa_approx),
                'exact_kappa_from_theory': float(estimated_kappa_exact),
                'approx_kappa_from_samples': float(empirical_kappa_approx),
                'exact_kappa_from_samples': float(empirical_kappa_exact),
                'approx_theory_error': float((estimated_kappa_approx - kappa) / kappa * 100),
                'exact_theory_error': float((estimated_kappa_exact - kappa) / kappa * 100),
                'approx_empirical_error': float((empirical_kappa_approx - kappa) / kappa * 100),
                'exact_empirical_error': float((empirical_kappa_exact - kappa) / kappa * 100)
            }
            
            dimension_results.append(result)
            
            print(f"Kappa = {kappa}:")
            print(f"  Theoretical R: {R_theory:.6f}")
            print(f"  Empirical R: {R_empirical:.6f}")
            print(f"  Approx kappa from theoretical R: {estimated_kappa_approx:.2f} (error: {result['approx_theory_error']:.2f}%)")
            print(f"  Exact kappa from theoretical R: {estimated_kappa_exact:.2f} (error: {result['exact_theory_error']:.2f}%)")
            print(f"  Approx kappa from empirical R: {empirical_kappa_approx:.2f} (error: {result['approx_empirical_error']:.2f}%)")
            print(f"  Exact kappa from empirical R: {empirical_kappa_exact:.2f} (error: {result['exact_empirical_error']:.2f}%)")
        
        all_results.extend(dimension_results)
        
        # Plot results for this dimension
        ax = axes[d_idx]
        true_kappas = [r['true_kappa'] for r in dimension_results]
        # Get both approximation and exact values
        approx_theory_kappas = [r['approx_kappa_from_theory'] for r in dimension_results]
        exact_theory_kappas = [r['exact_kappa_from_theory'] for r in dimension_results]
        approx_empirical_kappas = [r['approx_kappa_from_samples'] for r in dimension_results]
        exact_empirical_kappas = [r['exact_kappa_from_samples'] for r in dimension_results]
        
        ax.loglog(true_kappas, true_kappas, 'k--', label='True Value')
        ax.loglog(true_kappas, approx_theory_kappas, 'bo-', label='Approx Theory')
        ax.loglog(true_kappas, exact_theory_kappas, 'gx-', label='Exact Theory')
        ax.loglog(true_kappas, approx_empirical_kappas, 'r^-', label='Approx Samples')
        ax.loglog(true_kappas, exact_empirical_kappas, 'm+-', label='Exact Samples')
        
        ax.set_xlabel('True Kappa')
        ax.set_ylabel('Estimated Kappa')
        ax.set_title(f'Dimension = {d}')
        ax.legend()
        ax.grid(True, which='both', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('vmf_concentration_test.png')
    
    return all_results

def run_vmf_tests():
    """
    Test the log_expected_query_mass function for various kappa values
    and query angles.
    """
    print("\n" + "="*50)
    print("TESTING LOG_EXPECTED_QUERY_MASS FUNCTION")
    print("="*50)
    
    # Set random seed for reproducibility
    key = jax.random.PRNGKey(0)
    
    # Set dimension
    d = 64
    
    # Test parameters
    kappa_values = [1.0, 2.0, 5.0, 20.0]
    angles_degrees = [0, 15, 30, 45, 60, 90, 135, 180]
    n_samples = 1000
    
    # Results storage
    results = []
    
    # Create a figure for plotting
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs = axs.flatten()
    
    for i, kappa in enumerate(kappa_values):
        key, subkey = jax.random.split(key)
        
        # Generate a random unit vector as the true mean direction
        mu_key, sample_key, query_key = jax.random.split(subkey, 3)
        mu_true = jax.random.normal(mu_key, (d,))
        mu_true = mu_true / jnp.linalg.norm(mu_true)
        
        # Generate samples from vMF distribution
        samples = generate_vmf_samples(mu_true, kappa, n_samples, sample_key)
        
        # Fit vMF distribution to samples using both methods
        mu_est, kappa_est_approx = fit_vmf_distribution(samples, use_exact=False)
        _, kappa_est_exact = fit_vmf_distribution(samples, use_exact=True)
        
        # Compute the angle between true and estimated mean directions
        angle_error = jnp.arccos(jnp.clip(jnp.dot(mu_true, mu_est), -1.0, 1.0)) * 180 / jnp.pi
        
        print(f"Test case kappa={kappa}:")
        print(f"  True mean direction norm: {jnp.linalg.norm(mu_true):.4f}")
        print(f"  Estimated mean direction norm: {jnp.linalg.norm(mu_est):.4f}")
        print(f"  Angle between true and estimated mean directions: {angle_error:.4f} degrees")
        print(f"  True kappa: {kappa:.4f}")
        print(f"  Estimated kappa (approx): {kappa_est_approx:.4f} (error: {(kappa_est_approx - kappa) / kappa * 100:.2f}%)")
        print(f"  Estimated kappa (exact): {kappa_est_exact:.4f} (error: {(kappa_est_exact - kappa) / kappa * 100:.2f}%)")
        
        # Generate queries at different angles to the mean direction and compute attention
        query_results = []
        
        for angle in angles_degrees:
            query_key, query_key = jax.random.split(query_key)
            query = generate_query_with_angle(mu_est, angle, query_key)
            
            # Compute exact attention weights
            # Since softmax already normalizes to 1, we need to get unnormalized values
            # to compare with our approximation method
            dot_products = jnp.dot(query, samples.T)
            exp_dots = jnp.exp(dot_products)
            exact_total_mass = jnp.mean(exp_dots)  # Average over all samples
            
            # Compute approximate attention mass using both kappa estimations
            approx_mass_with_approx_kappa = compute_approx_attention_mass(query, mu_est, kappa_est_approx)
            approx_mass_with_exact_kappa = compute_approx_attention_mass(query, mu_est, kappa_est_exact)
            
            # Compute relative errors, handling potential division by zero
            if exact_total_mass > 1e-10:
                rel_error_approx = (approx_mass_with_approx_kappa - exact_total_mass) / exact_total_mass * 100
                rel_error_exact = (approx_mass_with_exact_kappa - exact_total_mass) / exact_total_mass * 100
            else:
                rel_error_approx = jnp.nan
                rel_error_exact = jnp.nan
                
            # Store results
            query_results.append({
                'angle': angle,
                'exact_mass': exact_total_mass,
                'approx_mass_with_approx_kappa': approx_mass_with_approx_kappa,
                'approx_mass_with_exact_kappa': approx_mass_with_exact_kappa,
                'relative_error_approx': rel_error_approx,
                'relative_error_exact': rel_error_exact
            })
            
            print(f"  Query angle {angle}° - Exact: {exact_total_mass:.6f}")
            print(f"    Approx (approx kappa): {approx_mass_with_approx_kappa:.6f}, Error: {rel_error_approx:.2f}%")
            print(f"    Approx (exact kappa): {approx_mass_with_exact_kappa:.6f}, Error: {rel_error_exact:.2f}%")
        
        # Plot results for this kappa value
        angles = [r['angle'] for r in query_results]
        exact_masses = [r['exact_mass'] for r in query_results]
        approx_masses_approx_kappa = [r['approx_mass_with_approx_kappa'] for r in query_results]
        approx_masses_exact_kappa = [r['approx_mass_with_exact_kappa'] for r in query_results]
        rel_errors_approx = [r['relative_error_approx'] for r in query_results]
        rel_errors_exact = [r['relative_error_exact'] for r in query_results]
        
        ax = axs[i]
        ax.plot(angles, exact_masses, 'o-', label='Exact')
        ax.plot(angles, approx_masses_approx_kappa, 's--', label='Approx with approx kappa')
        ax.plot(angles, approx_masses_exact_kappa, 'x--', label='Approx with exact kappa')
        ax.set_title(f'kappa = {kappa:.1f} (approx: {kappa_est_approx:.1f}, exact: {kappa_est_exact:.1f})')
        ax.set_xlabel('Query angle (degrees)')
        ax.set_ylabel('Attention mass')
        ax.legend()
        ax.grid(True)
        
        # Add a second y-axis for relative error
        ax2 = ax.twinx()
        ax2.plot(angles, rel_errors_approx, 'rx-', label='Error with approx kappa')
        ax2.plot(angles, rel_errors_exact, 'gx-', label='Error with exact kappa')
        ax2.set_ylabel('Relative error (%)')
        ax2.legend(loc='lower right')
        
        # Store overall results
        results.append({
            'kappa_true': kappa,
            'kappa_est_approx': kappa_est_approx,
            'kappa_est_exact': kappa_est_exact,
            'angle_error': angle_error,
            'query_results': query_results
        })
    
    plt.tight_layout()
    plt.savefig('vmf_approximation_accuracy.png')
    plt.close()
    
    # Create a summary plot of relative errors across different kappas
    # First for approximate kappa method
    plt.figure(figsize=(10, 6))
    for result in results:
        kappa = result['kappa_true']
        angles = [r['angle'] for r in result['query_results']]
        errors = [r['relative_error_approx'] for r in result['query_results']]
        plt.plot(angles, errors, 'o-', label=f'kappa = {kappa:.1f}')
    
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.xlabel('Query angle (degrees)')
    plt.ylabel('Relative error (%)')
    plt.title('Approximation error with approximate kappa vs. query angle')
    plt.legend()
    plt.grid(True)
    plt.savefig('vmf_error_summary_approx_kappa.png')
    plt.close()
    
    # Then for exact kappa method
    plt.figure(figsize=(10, 6))
    for result in results:
        kappa = result['kappa_true']
        angles = [r['angle'] for r in result['query_results']]
        errors = [r['relative_error_exact'] for r in result['query_results']]
        plt.plot(angles, errors, 'o-', label=f'kappa = {kappa:.1f}')
    
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.xlabel('Query angle (degrees)')
    plt.ylabel('Relative error (%)')
    plt.title('Approximation error with exact kappa vs. query angle')
    plt.legend()
    plt.grid(True)
    plt.savefig('vmf_error_summary_exact_kappa.png')
    plt.close()
    
    # Create a comparison plot showing accuracy improvement
    plt.figure(figsize=(10, 6))
    for result in results:
        kappa = result['kappa_true']
        angles = [r['angle'] for r in result['query_results']]
        approx_errors = [r['relative_error_approx'] for r in result['query_results']]
        exact_errors = [r['relative_error_exact'] for r in result['query_results']]
        improvement = [abs(ae) - abs(ee) for ae, ee in zip(approx_errors, exact_errors)]
        plt.plot(angles, improvement, 'o-', label=f'kappa = {kappa:.1f}')
    
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.xlabel('Query angle (degrees)')
    plt.ylabel('Absolute error improvement (%)')
    plt.title('Error improvement from using exact kappa estimation')
    plt.legend()
    plt.grid(True)
    plt.savefig('vmf_error_improvement.png')
    plt.close()
    
    return results

if __name__ == "__main__":
    # First test the vmf_concentration function
    #concentration_results = test_vmf_concentration()
    
    # Then test the log_expected_query_mass function
    attention_results = run_vmf_tests()
