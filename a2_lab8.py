import numpy as np

def bin_continuous_feature(feature, binning_type='equal_width', num_bins=10):
    if binning_type == 'equal_width':
        # Calculate bin width
        bin_width = (np.max(feature) - np.min(feature)) / num_bins
        
        # Create bins
        bins = [np.min(feature) + i * bin_width for i in range(num_bins)]
        bins.append(np.max(feature))  # Add upper bound of the last bin
        
        # Bin the feature values
        binned_feature = np.digitize(feature, bins)
        
        return binned_feature, bins
    
    elif binning_type == 'frequency':
        # Calculate bin edges based on frequency
        bin_edges = np.linspace(np.min(feature), np.max(feature), num_bins + 1)
        
        # Bin the feature values
        binned_feature = np.digitize(feature, bin_edges)
        
        return binned_feature, bin_edges

# Example usage:
# Assuming you have loaded your dataset into a variable named 'data'
# Assuming 'feature_to_bin' is the feature you want to bin
# Adjust the parameters as needed

# Apply equal width binning with 5 bins
binned_feature_equal_width, bins_equal_width = bin_continuous_feature(data['feature_to_bin'], binning_type='equal_width', num_bins=5)

# Apply frequency binning with 10 bins
binned_feature_frequency, bin_edges_frequency = bin_continuous_feature(data['feature_to_bin'], binning_type='frequency', num_bins=10)

# Print the results
print("Equal Width Bins:", bins_equal_width)
print("Frequency Bins:", bin_edges_frequency)
