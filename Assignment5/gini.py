def calculate_gini_impurity(sample_sizes):
    
    if not sample_sizes or sum(sample_sizes) == 0:
        return 0.0

    total_samples = sum(sample_sizes)
    probabilities = [size / total_samples for size in sample_sizes]
    gini_impurity = 1 - sum(p ** 2 for p in probabilities)

    print(gini_impurity)

# Example usage:
sample_sizes = [0.25, 0.6, 0.05, 0.1]
calculate_gini_impurity(sample_sizes)