# Gini Coefficient Estimation

A Python toolkit for calculating, visualizing, and analyzing Gini coefficients to measure statistical dispersion and inequality in datasets.

## Overview

The Gini coefficient is a statistical measure commonly used to gauge economic inequality within a population. Originally developed by Italian statistician Corrado Gini in 1912, it ranges from 0 (perfect equality) to 1 (maximum inequality). This repository provides tools to calculate Gini coefficients, create visualizations, and identify outliers in your data.

## Features

- **Gini Coefficient Calculation**: Core functionality to compute Gini coefficients for any numeric dataset
- **Data Visualization**: Generate bar plots and charts to visualize inequality distributions
- **Outlier Detection**: Identify and analyze outliers in Gini coefficient datasets
- **Easy Integration**: Simple Python modules that can be imported into larger projects

## Files

- `gini.py` - Core Gini coefficient calculation functions
- `gini_barplot.py` - Visualization tools for creating bar plots of Gini data
- `gini_outlliers.py` - Outlier detection and analysis utilities

## Installation

### Prerequisites

- Python 3.7 or higher
- Required packages (install via pip):

```bash
pip install numpy pandas matplotlib seaborn scipy
```

### Setup

1. Clone the repository:
```bash
git clone https://github.com/wenqifu/[repository-name].git
cd [repository-name]
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Gini Coefficient Calculation

```python
from gini import calculate_gini

# Example with income data
income_data = [20000, 30000, 40000, 50000, 100000, 200000]
gini_coefficient = calculate_gini(income_data)
print(f"Gini coefficient: {gini_coefficient:.3f}")
```

### Creating Visualizations

```python
from gini_barplot import create_gini_barplot

# Create a bar plot visualization
data = [...]  # Your dataset
create_gini_barplot(data, title="Income Inequality Analysis")
```

### Outlier Detection

```python
from gini_outlliers import detect_outliers

# Identify outliers in your dataset
outliers = detect_outliers(your_data)
print(f"Found {len(outliers)} outliers")
```

## Understanding Gini Coefficients

- **0.0**: Perfect equality (everyone has the same value)
- **0.0 - 0.3**: Relatively equal distribution
- **0.3 - 0.5**: Moderate inequality
- **0.5 - 0.7**: High inequality
- **0.7 - 1.0**: Very high inequality
- **1.0**: Maximum inequality (one person has everything)

## Applications

This toolkit can be used for analyzing:
- Income and wealth inequality
- Resource distribution
- Performance metrics across teams or regions
- Any dataset where you want to measure concentration or dispersion

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

- Gini, C. (1912). "Measurement of Inequality of Incomes". *The Economic Journal*
- World Bank Gini coefficient data and methodology
- OECD Income Distribution Database

## Support

If you encounter any issues or have questions, please open an issue on GitHub.

---

**Author**: wenqifu  
**Last Updated**: 2025
