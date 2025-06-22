import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Set global font size to 20 for all plot text
plt.rcParams.update({'font.size': 20})

def generate_pareto_data(alpha, beta, n=1000):
    """Generate n points from Pareto distribution"""
    return stats.pareto.rvs(b=alpha, scale=beta, size=n)

def compute_quartiles(data):
    """Compute quartiles Q1, Q2 (median), Q3"""
    Q1 = np.percentile(data, 25)
    Q2 = np.percentile(data, 50)  # Median
    Q3 = np.percentile(data, 75)
    return Q1, Q2, Q3

def compute_mad(data):
    """Compute Median Absolute Deviation (MAD)"""
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    return mad

def estimate_h_star_direct(data, Q1, M, Q3):
    """H*-based method using direct MAD calculation"""
    # Even though we calculate H* from data, we use the same formulas as H* estimate
    # This follows the table specification
    alpha = 0.5 * (Q3 - Q1) / (Q1 + Q3 - 2*M)
    G = 0.5 * (Q1 + Q3 - 2*M) / (M - Q1)
    return alpha, G

def estimate_h_star_estimate(Q1, M, Q3):
    """H*-based method using quartile estimation"""
    # From equation (eq:h_approx), the approximations are:
    # H* ≈ (1/4)(Q3 - Q1)
    # α ≈ (1/2)((Q3 - Q1)/(Q1 + Q3 - 2M))
    # G ≈ (1/2)((Q1 + Q3 - 2M)/(M - Q1))
    
    alpha = 0.5 * (Q3 - Q1) / (Q1 + Q3 - 2*M)
    G = 0.5 * (Q1 + Q3 - 2*M) / (M - Q1)
    return alpha, G

def estimate_midpoint(Q1, M, Q3):
    """Midpoint method estimation"""
    numerator = 3*Q1 - Q3
    alpha = 1 + numerator / (2*M - 3*Q1 + Q3)
    G = 1 - 2 * numerator / (2*M + 3*Q1 - Q3)
    return alpha, G

def estimate_one_trapezoid(Q1, M, Q3):
    """One Trapezoid method estimation"""
    numerator = 3*Q1 - Q3
    alpha = 1 + 0.5 * numerator / (Q3 - Q1)
    G = 1 - 0.5 * numerator / Q1
    return alpha, G

def estimate_two_trapezoids(Q1, M, Q3):
    """Two Trapezoids method estimation"""
    numerator = 3*Q1 - Q3
    alpha = 1 + 2 * numerator / (2*M - 5*Q1 + 3*Q3)
    G = 1 - 4 * numerator / (2*M + 7*Q1 - Q3)
    return alpha, G

def estimate_quartiles_average(Q1, M, Q3):
    """Quartiles Average method estimation"""
    numerator = 3*Q1 - Q3
    alpha = 1 + 3 * numerator / (2*M - 7*Q1 + 5*Q3)
    G = 1 - 6 * numerator / (2*M + 11*Q1 - Q3)
    return alpha, G

def estimate_simpson_18(Q1, M, Q3):
    """Simpson 1/8 method estimation"""
    numerator = 3*Q1 - Q3
    alpha = 1 + 0.75 * numerator / (M - 2*Q1 + Q3)
    G = 1 - 3 * numerator / (2*M + 5*Q1 - Q3)
    return alpha, G

def estimate_quantile_based(Q1, M, Q3):
    """Quantile-based method using log formulas"""
    alpha = 1 / np.log(Q3 / M)
    log_ratio = np.log(Q3 / M)
    G = log_ratio / (2 - log_ratio)
    return alpha, G

def estimate_mle(data):
    """Maximum Likelihood Estimation for Pareto parameters"""
    beta_mle = np.min(data)
    n = len(data)
    alpha_mle = n / np.sum(np.log(data / beta_mle))
    G_mle = 1 / (2*alpha_mle - 1)
    return alpha_mle, G_mle, beta_mle

def calculate_percentage_error(true_value, estimated_value):
    """Calculate percentage error"""
    return 100 * abs(true_value - estimated_value) / true_value

def run_simulations_for_sample_size(true_alpha, true_beta, n_points, n_simulations, methods):
    """Run multiple simulations for a given sample size and return average errors"""
    all_alpha_errors = {method[0]: [] for method in methods}
    all_alpha_errors['MLE'] = []
    all_gini_errors = {method[0]: [] for method in methods}
    all_gini_errors['MLE'] = []
    
    # True Gini coefficient
    true_gini = 1 / (2*true_alpha - 1)
    
    for _ in range(n_simulations):
        # Generate data
        data = generate_pareto_data(true_alpha, true_beta, n_points)
        
        # Compute quartiles
        Q1, Q2, Q3 = compute_quartiles(data)
        M = Q2  # Median
        
        # Apply each method
        for method_name, method_func in methods:
            try:
                if method_name == 'H* direct':
                    alpha_est, gini_est = method_func(data, Q1, M, Q3)
                else:
                    alpha_est, gini_est = method_func(Q1, M, Q3)
                
                alpha_error = calculate_percentage_error(true_alpha, alpha_est)
                gini_error = calculate_percentage_error(true_gini, gini_est)
                
                all_alpha_errors[method_name].append(alpha_error)
                all_gini_errors[method_name].append(gini_error)
            except:
                pass
        
        # MLE estimation
        try:
            alpha_mle, gini_mle, _ = estimate_mle(data)
            alpha_error_mle = calculate_percentage_error(true_alpha, alpha_mle)
            gini_error_mle = calculate_percentage_error(true_gini, gini_mle)
            
            all_alpha_errors['MLE'].append(alpha_error_mle)
            all_gini_errors['MLE'].append(gini_error_mle)
        except:
            pass
    
    # Calculate average errors
    avg_alpha_errors = {}
    avg_gini_errors = {}
    
    for method_name in all_alpha_errors:
        if all_alpha_errors[method_name]:
            avg_alpha_errors[method_name] = np.mean(all_alpha_errors[method_name])
            avg_gini_errors[method_name] = np.mean(all_gini_errors[method_name])
        else:
            avg_alpha_errors[method_name] = None
            avg_gini_errors[method_name] = None
    
    return avg_alpha_errors, avg_gini_errors

def create_pareto_plots():
    # Parameters
    true_alpha = 1.5
    true_beta = 1.0
    n_simulations = 100
    
    # Sample sizes for the Pareto distribution
    n = np.array([100, 200, 400, 800])
    x = np.arange(len(n))  # x-coordinates for the groups
    
    # Methods to use
    methods = [
        ('H* direct', estimate_h_star_direct),
        ('H* estimate', estimate_h_star_estimate),
        ('Midpoint', estimate_midpoint),
        ('One Trapezoid', estimate_one_trapezoid),
        ('Two Trapezoids', estimate_two_trapezoids),
        ('Quartiles Average', estimate_quartiles_average),
        ('Simpson 1/8', estimate_simpson_18),
        ('Quantile-based', estimate_quantile_based)
    ]
    
    # Run simulations for each sample size
    alpha_results = {}
    gini_results = {}
    
    for sample_size in n:
        print(f"Running simulations for n={sample_size}...")
        alpha_results[sample_size], gini_results[sample_size] = run_simulations_for_sample_size(
            true_alpha, true_beta, sample_size, n_simulations, methods)
    
    # Extract errors for each method across all sample sizes
    # Alpha errors
    mle_alpha_errors = np.array([alpha_results[sample_size]['MLE'] for sample_size in n])
    h_direct_alpha_errors = np.array([alpha_results[sample_size]['H* direct'] for sample_size in n])
    h_estimate_alpha_errors = np.array([alpha_results[sample_size]['H* estimate'] for sample_size in n])
    midpoint_alpha_errors = np.array([alpha_results[sample_size]['Midpoint'] for sample_size in n])
    one_trap_alpha_errors = np.array([alpha_results[sample_size]['One Trapezoid'] for sample_size in n])
    two_trap_alpha_errors = np.array([alpha_results[sample_size]['Two Trapezoids'] for sample_size in n])
    quartiles_avg_alpha_errors = np.array([alpha_results[sample_size]['Quartiles Average'] for sample_size in n])
    simpson_alpha_errors = np.array([alpha_results[sample_size]['Simpson 1/8'] for sample_size in n])
    quantile_alpha_errors = np.array([alpha_results[sample_size]['Quantile-based'] for sample_size in n])
    
    # Gini errors
    mle_gini_errors = np.array([gini_results[sample_size]['MLE'] for sample_size in n])
    h_direct_gini_errors = np.array([gini_results[sample_size]['H* direct'] for sample_size in n])
    h_estimate_gini_errors = np.array([gini_results[sample_size]['H* estimate'] for sample_size in n])
    midpoint_gini_errors = np.array([gini_results[sample_size]['Midpoint'] for sample_size in n])
    one_trap_gini_errors = np.array([gini_results[sample_size]['One Trapezoid'] for sample_size in n])
    two_trap_gini_errors = np.array([gini_results[sample_size]['Two Trapezoids'] for sample_size in n])
    quartiles_avg_gini_errors = np.array([gini_results[sample_size]['Quartiles Average'] for sample_size in n])
    simpson_gini_errors = np.array([gini_results[sample_size]['Simpson 1/8'] for sample_size in n])
    quantile_gini_errors = np.array([gini_results[sample_size]['Quantile-based'] for sample_size in n])
    
    # Set the width for each bar (smaller to fit all methods)
    width = 0.08
    
    # ===== ALPHA PLOTS =====
    
    # Only show data for n=100 and n=200 (first two elements)
    x_small = np.array([0, 1])  # positions for 100 and 200
    # Only show data for n=400 and n=800 (last two elements)
    x_large = np.array([0, 1])  # positions for 400 and 800
    
    # Use slightly wider bars for fewer methods
    width_selected = 0.1
    
    # ---------------
    # Plot 1: Selected methods for n=100 and n=200 (Alpha, excluding H* estimate and Quantile-based)
    # ---------------
    plt.figure(figsize=(10, 6))
    
    plt.bar(x_small - 3*width_selected, mle_alpha_errors[:2], width=width_selected, label='MLE', color='red')
    plt.bar(x_small - 2*width_selected, h_direct_alpha_errors[:2], width=width_selected, label='H* direct', color='green')
    plt.bar(x_small - 1*width_selected, midpoint_alpha_errors[:2], width=width_selected, label='Midpoint', color='orange')
    plt.bar(x_small + 0*width_selected, one_trap_alpha_errors[:2], width=width_selected, label='One Trap', color='purple')
    plt.bar(x_small + 1*width_selected, two_trap_alpha_errors[:2], width=width_selected, label='Two Trap', color='brown')
    plt.bar(x_small + 2*width_selected, quartiles_avg_alpha_errors[:2], width=width_selected, label='Quart Avg', color='pink')
    plt.bar(x_small + 3*width_selected, simpson_alpha_errors[:2], width=width_selected, label='Simpson', color='gray')
    
    plt.xlabel('')
    plt.ylabel('Relative Error (%)')
    plt.xticks(x_small, [100, 200])
    plt.legend(loc='upper right', fontsize=15, ncol=1)
    plt.title('Alpha α Estimation (n=100, 200)')
    plt.tight_layout()
    plt.savefig('pareto_alpha_n100_n200.png', bbox_inches='tight')
    plt.show()
    
    # ---------------
    # Plot 2: Selected methods for n=400 and n=800 (Alpha, excluding H* estimate and Quantile-based)
    # ---------------
    plt.figure(figsize=(10, 6))
    
    plt.bar(x_large - 3*width_selected, mle_alpha_errors[2:], width=width_selected, label='MLE', color='red')
    plt.bar(x_large - 2*width_selected, h_direct_alpha_errors[2:], width=width_selected, label='H* direct', color='green')
    plt.bar(x_large - 1*width_selected, midpoint_alpha_errors[2:], width=width_selected, label='Midpoint', color='orange')
    plt.bar(x_large + 0*width_selected, one_trap_alpha_errors[2:], width=width_selected, label='One Trap', color='purple')
    plt.bar(x_large + 1*width_selected, two_trap_alpha_errors[2:], width=width_selected, label='Two Trap', color='brown')
    plt.bar(x_large + 2*width_selected, quartiles_avg_alpha_errors[2:], width=width_selected, label='Quart Avg', color='pink')
    plt.bar(x_large + 3*width_selected, simpson_alpha_errors[2:], width=width_selected, label='Simpson', color='gray')
    
    plt.xlabel('')
    plt.ylabel('Relative Error (%)')
    plt.xticks(x_large, [400, 800])
    plt.legend(loc='upper right', fontsize=15, ncol=1)
    plt.title('Alpha α Estimation (n=400, 800)')
    plt.tight_layout()
    plt.savefig('pareto_alpha_n400_n800.png', bbox_inches='tight')
    plt.show()
    
    # ===== GINI PLOTS =====
    
    # ---------------
    # Plot 3: Selected methods for n=100 and n=200 (Gini, excluding H* estimate and Quantile-based)
    # ---------------
    plt.figure(figsize=(10, 6))
    
    plt.bar(x_small - 3*width_selected, mle_gini_errors[:2], width=width_selected, label='MLE', color='red')
    plt.bar(x_small - 2*width_selected, h_direct_gini_errors[:2], width=width_selected, label='H* direct', color='green')
    plt.bar(x_small - 1*width_selected, midpoint_gini_errors[:2], width=width_selected, label='Midpoint', color='orange')
    plt.bar(x_small + 0*width_selected, one_trap_gini_errors[:2], width=width_selected, label='One Trap', color='purple')
    plt.bar(x_small + 1*width_selected, two_trap_gini_errors[:2], width=width_selected, label='Two Trap', color='brown')
    plt.bar(x_small + 2*width_selected, quartiles_avg_gini_errors[:2], width=width_selected, label='Quart Avg', color='pink')
    plt.bar(x_small + 3*width_selected, simpson_gini_errors[:2], width=width_selected, label='Simpson', color='gray')
    
    plt.xlabel('')
    plt.ylabel('Relative Error (%)')
    plt.xticks(x_small, [100, 200])
    plt.legend(loc='upper right', fontsize=15, ncol=1)
    plt.title('Gini G Estimation (n=100, 200)')
    plt.tight_layout()
    plt.savefig('pareto_gini_n100_n200.png', bbox_inches='tight')
    plt.show()
    
    # ---------------
    # Plot 4: Selected methods for n=400 and n=800 (Gini, excluding H* estimate and Quantile-based)
    # ---------------
    plt.figure(figsize=(10, 6))
    
    plt.bar(x_large - 3*width_selected, mle_gini_errors[2:], width=width_selected, label='MLE', color='red')
    plt.bar(x_large - 2*width_selected, h_direct_gini_errors[2:], width=width_selected, label='H* direct', color='green')
    plt.bar(x_large - 1*width_selected, midpoint_gini_errors[2:], width=width_selected, label='Midpoint', color='orange')
    plt.bar(x_large + 0*width_selected, one_trap_gini_errors[2:], width=width_selected, label='One Trap', color='purple')
    plt.bar(x_large + 1*width_selected, two_trap_gini_errors[2:], width=width_selected, label='Two Trap', color='brown')
    plt.bar(x_large + 2*width_selected, quartiles_avg_gini_errors[2:], width=width_selected, label='Quart Avg', color='pink')
    plt.bar(x_large + 3*width_selected, simpson_gini_errors[2:], width=width_selected, label='Simpson', color='gray')
    
    plt.xlabel('')
    plt.ylabel('Relative Error (%)')
    plt.xticks(x_large, [400, 800])
    plt.legend(loc='upper right', fontsize=15, ncol=1)
    plt.title('Gini G Estimation (n=400, 800)')
    plt.tight_layout()
    plt.savefig('pareto_gini_n400_n800.png', bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*80)
    print(f"SUMMARY STATISTICS (α={true_alpha}, G={1/(2*true_alpha-1):.3f}, {n_simulations} simulations per sample size)")
    print("="*80)
    
    # Create summary dataframes
    alpha_summary_df = pd.DataFrame(alpha_results).round(2)
    gini_summary_df = pd.DataFrame(gini_results).round(2)
    
    print("\nALPHA ESTIMATION ERRORS:")
    print(alpha_summary_df)
    
    print("\nGINI ESTIMATION ERRORS:")
    print(gini_summary_df)
    
    print("\n" + "="*80)
    print("FILES SAVED:")
    print("="*80)
    print("Alpha plots:")
    print("1. pareto_alpha_n100_n200.png")
    print("2. pareto_alpha_n400_n800.png") 
    print("\nGini plots:")
    print("3. pareto_gini_n100_n200.png")
    print("4. pareto_gini_n400_n800.png")

if __name__ == "__main__":
    create_pareto_plots()