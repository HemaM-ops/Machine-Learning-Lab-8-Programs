import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load your dataset
dataset_path = "C:\\Users\\mahad\\OneDrive\\Desktop\\SEM 4 Courses\\ML\\Legality\\Datasets\\English_Abstractive_Embeddings_Fasttext.xlsx"
data = pd.read_excel(dataset_path)


# Assuming you have a target variable named 'Judgement Status'
target = data["Judgement Status"].values

# Dropping the target variable from the dataset
data.drop(columns=["Judgement Status"], inplace=True)

# Splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

# Function to calculate information gain
# (Keep the rest of the code the same)

# Function to calculate information gain
def calculate_information_gain(X, y):
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

# Function to find the best feature for splitting
def find_best_split(X, y):
    n_features = X.shape[1]
    best_feature = None
    best_info_gain = -np.inf
    
    for feature in range(n_features):
        # Implement splitting and calculate information gain for each feature
        # Here, you can use any method for splitting, e.g., binning for categorical data
        # For simplicity, let's use scikit-learn's DecisionTreeClassifier to calculate information gain
        clf = DecisionTreeClassifier(criterion='entropy', max_depth=1)
        clf.fit(X[:, feature].reshape(-1, 1), y)
        info_gain = calculate_information_gain(X[:, feature], y) - clf.tree_.impurity[0]
        
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = feature
    
    return best_feature

# Function to build the decision tree
def build_decision_tree(X, y):
    # Base case: if all labels are the same, return a leaf node with that label
    if len(np.unique(y)) == 1:
        return {'class': y[0]}
    
    # Find the best feature to split on
    best_feature = find_best_split(X, y)
    
    # Split the data based on the best feature
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=1)
    clf.fit(X[:, best_feature].reshape(-1, 1), y)
    threshold = clf.tree_.threshold[0]
    left_indices = np.where(X[:, best_feature] <= threshold)[0]
    right_indices = np.where(X[:, best_feature] > threshold)[0]
    
    # Create the decision node
    decision_node = {
        'feature_index': best_feature,
        'threshold': threshold
    }
    
    # Recursively build the left and right subtrees
    decision_node['left'] = build_decision_tree(X[left_indices], y[left_indices])
    decision_node['right'] = build_decision_tree(X[right_indices], y[right_indices])
    
    return decision_node

# Function to predict using the decision tree
def predict(tree, sample):
    if 'class' in tree:
        return tree['class']
    
    feature_index = tree['feature_index']
    threshold = tree['threshold']
    
    if sample[feature_index] <= threshold:
        return predict(tree['left'], sample)
    else:
        return predict(tree['right'], sample)

# Build the decision tree
decision_tree = build_decision_tree(X_train, y_train)

# Make predictions on the test set
predictions = [predict(decision_tree, sample) for sample in X_test]

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)
