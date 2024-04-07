import numpy as np

class MyDecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y, binning_type='equal_width', num_bins=10):
        self.tree = self._build_tree(X, y, max_depth=self.max_depth, binning_type=binning_type, num_bins=num_bins)

    def predict(self, X):
        if self.tree is None:
            raise Exception("Tree not built. Call fit() first.")
        return np.array([self._predict_tree(sample, self.tree) for sample in X])

    def _build_tree(self, X, y, current_depth=0, max_depth=None, binning_type='equal_width', num_bins=10):
        if max_depth is not None and current_depth >= max_depth:
            return {'class': np.bincount(y).argmax()}

        if len(np.unique(y)) == 1:
            return {'class': y[0]}

        best_feature, threshold = self._find_best_split(X, y, binning_type=binning_type, num_bins=num_bins)

        if best_feature is None:
            return {'class': np.bincount(y).argmax()}

        left_indices = np.where(X[:, best_feature] <= threshold)[0]
        right_indices = np.where(X[:, best_feature] > threshold)[0]

        decision_node = {
            'feature_index': best_feature,
            'threshold': threshold,
            'left': self._build_tree(X[left_indices], y[left_indices], current_depth=current_depth + 1, max_depth=max_depth),
            'right': self._build_tree(X[right_indices], y[right_indices], current_depth=current_depth + 1, max_depth=max_depth)
        }

        return decision_node

    def _predict_tree(self, sample, tree):
        if 'class' in tree:
            return tree['class']
        feature_index = tree['feature_index']
        threshold = tree['threshold']
        if sample[feature_index] <= threshold:
            return self._predict_tree(sample, tree['left'])
        else:
            return self._predict_tree(sample, tree['right'])

    def _calculate_information_gain(self, X, y):
        # Implement information gain calculation here
        # You can use any method to calculate information gain, e.g., entropy or Gini impurity
        # For simplicity, let's use entropy here
        classes = np.unique(y)
        n_samples = len(y)
        entropy_parent = 0

        for c in classes:
            prob_c = np.sum(y == c) / n_samples
            entropy_parent -= prob_c * np.log2(prob_c)

        return entropy_parent

    def _find_best_split(self, X, y, binning_type='equal_width', num_bins=10):
        n_features = X.shape[1]
        best_feature = None
        best_info_gain = -np.inf
        best_threshold = None

        for feature in range(n_features):
            # Implement splitting and calculate information gain for each feature
            # Here, you can use any method for splitting, e.g., binning for categorical data
            # For simplicity, let's use scikit-learn's DecisionTreeClassifier to calculate information gain
            if len(np.unique(X[:, feature])) > 1:
                if binning_type == 'equal_width':
                    binned_feature, bins = self._equal_width_binning(X[:, feature], num_bins)
                elif binning_type == 'frequency':
                    binned_feature, bins = self._frequency_binning(X[:, feature], num_bins)
                else:
                    raise ValueError("Invalid binning type. Choose 'equal_width' or 'frequency'.")
                for threshold in bins[1:-1]:
                    left_indices = np.where(X[:, feature] <= threshold)[0]
                    right_indices = np.where(X[:, feature] > threshold)[0]
                    info_gain = self._calculate_information_gain(y) - \
                                (len(left_indices) / len(y)) * self._calculate_information_gain(y[left_indices]) - \
                                (len(right_indices) / len(y)) * self._calculate_information_gain(y[right_indices])
                    if info_gain > best_info_gain:
                        best_info_gain = info_gain
                        best_feature = feature
                        best_threshold = threshold
            else:
                continue

        return best_feature, best_threshold

    def _equal_width_binning(self, feature, num_bins):
        # Calculate bin width
        bin_width = (np.max(feature) - np.min(feature)) / num_bins

        # Create bins
        bins = [np.min(feature) + i * bin_width for i in range(num_bins)]
        bins.append(np.max(feature))  # Add upper bound of the last bin

        # Bin the feature values
        binned_feature = np.digitize(feature, bins)

        return binned_feature, bins

    def _frequency_binning(self, feature, num_bins):
        # Calculate bin edges based on frequency
        bin_edges = np.linspace(np.min(feature), np.max(feature), num_bins + 1)

        # Bin the feature values
        binned_feature = np.digitize(feature, bin_edges)

        return binned_feature, bin_edges
