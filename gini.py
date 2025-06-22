import numpy as np
import pandas as pd
from scipy import stats

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

def estimate_beta_quantile(M, Q3):
    """Estimate beta using quantile formula"""
    return M**2 / Q3

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

def run_single_simulation(true_alpha, true_beta, n_points, methods):
    """Run a single simulation and return errors for all methods"""
    # Generate data
    data = generate_pareto_data(true_alpha, true_beta, n_points)
    
    # Compute quartiles
    Q1, Q2, Q3 = compute_quartiles(data)
    M = Q2  # Median
    
    # True Gini coefficient
    true_gini = 1 / (2*true_alpha - 1)
    
    results = {}
    
    # Apply each method
    for method_name, method_func in methods:
        try:
            if method_name == 'H* direct':
                alpha_est, gini_est = method_func(data, Q1, M, Q3)
            else:
                alpha_est, gini_est = method_func(Q1, M, Q3)
            
            alpha_error = calculate_percentage_error(true_alpha, alpha_est)
            gini_error = calculate_percentage_error(true_gini, gini_est)
            
            # Beta estimation (using quantile formula for all methods)
            beta_est = estimate_beta_quantile(M, Q3)
            beta_error = calculate_percentage_error(true_beta, beta_est)
            
            results[method_name] = {
                'alpha_est': alpha_est,
                'alpha_error': alpha_error,
                'gini_est': gini_est,
                'gini_error': gini_error,
                'beta_est': beta_est,
                'beta_error': beta_error
            }
        except Exception as e:
            results[method_name] = None
    
    # MLE estimation
    try:
        alpha_mle, gini_mle, beta_mle = estimate_mle(data)
        alpha_error_mle = calculate_percentage_error(true_alpha, alpha_mle)
        gini_error_mle = calculate_percentage_error(true_gini, gini_mle)
        beta_error_mle = calculate_percentage_error(true_beta, beta_mle)
        
        results['MLE'] = {
            'alpha_est': alpha_mle,
            'alpha_error': alpha_error_mle,
            'gini_est': gini_mle,
            'gini_error': gini_error_mle,
            'beta_est': beta_mle,
            'beta_error': beta_error_mle
        }
    except:
        results['MLE'] = None
    
    return results

def format_latex_table(error_data, title, parameter_symbol):
    """Format error data as LaTeX table"""
    # Create header
    latex_lines = []
    latex_lines.append("\\begin{table}[h!]")
    latex_lines.append(f"\\caption{{{title}}}")
    latex_lines.append("\\centering")
    latex_lines.append("\\begin{tabular}{l" + "c" * 10 + "}")
    latex_lines.append("\\toprule")
    
    # Multi-row header
    alpha_values = [f"${a:.1f}$" for a in np.arange(1.1, 2.1, 0.1)]
    latex_lines.append(f"\\multirow{{2}}{{*}}{{\\textbf{{Method}}}} & \\multicolumn{{10}}{{c}}{{{parameter_symbol}}} \\\\")
    latex_lines.append("& " + " & ".join(alpha_values) + " \\\\")
    latex_lines.append("\\midrule")
    
    # Method names mapping for LaTeX
    method_latex_names = {
        'H* direct': '$H^{*}$ direct',
        'H* estimate': '$H^{*}$ estimate',
        'Midpoint': 'Midpoint',
        'One Trapezoid': 'One Trapezoid',
        'Two Trapezoids': 'Two Trapezoids',
        'Quartiles Average': 'Quartiles Average',
        'Simpson 1/8': 'Simpson 1/8',
        'MLE': 'MLE',
        'Quantile-based': 'Quantile-based'
    }
    
    # Format data rows
    for method in error_data:
        row_values = []
        latex_method_name = method_latex_names.get(method, method)
        
        for alpha in np.arange(1.1, 2.1, 0.1):
            alpha_key = f"{alpha:.1f}"
            if alpha_key in error_data[method]:
                row_values.append(f"${error_data[method][alpha_key]:.1f}$")
            else:
                row_values.append("--")
        
        latex_lines.append(f"{latex_method_name:20} & " + " & ".join(row_values) + " \\\\")
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")
    
    return "\n".join(latex_lines)

def main():
    # Parameters
    n_points = 1000
    true_beta = 1.0
    alpha_values = np.arange(1.1, 2.1, 0.1)
    n_simulations = 100
    
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
    
    # Initialize error storage
    alpha_errors = {method[0]: {} for method in methods}
    alpha_errors['MLE'] = {}
    gini_errors = {method[0]: {} for method in methods}
    gini_errors['MLE'] = {}
    
    print(f"Running {n_simulations} simulations for each alpha value...")
    print("-" * 50)
    
    for true_alpha in alpha_values:
        print(f"Processing α = {true_alpha:.1f}...")
        
        # Store results for all simulations
        all_sim_results = {method[0]: {'alpha_error': [], 'gini_error': []} 
                          for method in methods}
        all_sim_results['MLE'] = {'alpha_error': [], 'gini_error': []}
        
        # Run simulations
        for sim in range(n_simulations):
            sim_results = run_single_simulation(true_alpha, true_beta, n_points, methods)
            
            # Collect results
            for method_name in all_sim_results.keys():
                if method_name in sim_results and sim_results[method_name] is not None:
                    all_sim_results[method_name]['alpha_error'].append(sim_results[method_name]['alpha_error'])
                    all_sim_results[method_name]['gini_error'].append(sim_results[method_name]['gini_error'])
        
        # Calculate average errors
        alpha_key = f"{true_alpha:.1f}"
        for method_name in all_sim_results.keys():
            if all_sim_results[method_name]['alpha_error']:
                alpha_errors[method_name][alpha_key] = np.mean(all_sim_results[method_name]['alpha_error'])
                gini_errors[method_name][alpha_key] = np.mean(all_sim_results[method_name]['gini_error'])
    
    # Generate LaTeX tables
    print("\n" + "="*80)
    print("LATEX TABLE: TAIL INDEX α ESTIMATION ERRORS")
    print("="*80)
    alpha_table = format_latex_table(alpha_errors, 
                                    "Average Percentage Errors for Tail Index $\\alpha$ Estimation", 
                                    "Tail Index $\\alpha$")
    print(alpha_table)
    
    print("\n" + "="*80)
    print("LATEX TABLE: GINI COEFFICIENT G ESTIMATION ERRORS")
    print("="*80)
    gini_table = format_latex_table(gini_errors, 
                                   "Average Percentage Errors for Gini Coefficient $G$ Estimation", 
                                   "Gini Coefficient $G$")
    print(gini_table)
    
    # Calculate overall average errors
    print("\n" + "="*80)
    print("OVERALL AVERAGE PERCENTAGE ERRORS")
    print("="*80)
    
    for method in alpha_errors:
        if alpha_errors[method]:
            avg_alpha_error = np.mean(list(alpha_errors[method].values()))
            avg_gini_error = np.mean(list(gini_errors[method].values()))
            print(f"{method:20} - α: {avg_alpha_error:6.2f}%, G: {avg_gini_error:6.2f}%")
    
    print("\n" + "="*80)
    print(f"Simulation completed: {n_simulations} runs per alpha value")
    print("="*80)

if __name__ == "__main__":
    main()