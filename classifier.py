import numpy as np
import pandas as pd

class DecisionTree:
    def __init__(self):
        """
        Initializes a DecisionTree instance.
        The tree structure will be built and stored in `self.tree`.
        """
        self.tree = None  # To store the tree structure


    def create_treeID3(self, data, target_column ,validation_split=0.2, C45 = False):
        """
        Creates a decision tree using the C4.5 algorithm.
        
        Args:
        - data: A pandas DataFrame containing the dataset.
        - target_column: The column name that holds the target (class) values.
        - validation_split : The fraction of data to use for validation(default is 0.2)

        This method will start the tree building process by calling `build_tree`.
        """
        self.target_column = target_column

        #Split the data into training ans validation sets
        validation_size = int(len(data) * validation_split)
        validation_data = data.sample(validation_size, random_state=42)
        training_data = data.drop(validation_data.index)

        # Recursively build the tree using training dataset
        self.tree = self.build_tree(training_data,C45)




    def build_tree(self, data, C45=False):
        """
        Recursively builds the decision tree based on the dataset using either ID3 or C4.5.

        Args:
        - data: A pandas DataFrame containing the dataset to build the tree from.
        - C45: Boolean flag to indicate whether to use C4.5 (True) or ID3 (False).

        Returns:
        - A decision tree represented as a nested dictionary, where each node contains 
        the feature to split on and the corresponding subtrees.

        Terminal conditions:
        - All rows have the same target value: return the target value.
        - No more features to split on: return the majority class.
        """
        # Terminal condition: all rows have the same target value
        if len(data[self.target_column].unique()) == 1:
            return data[self.target_column].iloc[0]

        # Terminal condition: no more features to split on
        if len(data.columns) == 1:  # Only the target column left
            return data[self.target_column].mode()[0]  # Return the majority class

        # Calculate the dataset's initial entropy
        probabilities = data[self.target_column].value_counts(normalize=True).values
        df_entropy = self.calculate_entropy(probabilities)

        # Find the best column to split on
        highest_gain_col = ""
        highest_gain = 0

        for col in data.columns:
            if col == self.target_column:  # Skip the target column
                continue

            # Calculate weighted entropy for ID3
            weighted_entropy = self.calculate_weighted_entropy(data, col)

            # Calculate information gain or gain ratio depending on C4.5 flag
            if not C45:
                info_gain = df_entropy - weighted_entropy  # ID3 uses information gain
            else:
                # Calculate split information for C4.5
                split_info = self.calculate_split_information(data, col)
                # Gain Ratio = Information Gain / Split Information
                info_gain = (df_entropy - weighted_entropy) / split_info if split_info != 0 else 0

            # Track the column with the highest gain (or gain ratio)
            if info_gain > highest_gain:
                highest_gain = info_gain
                highest_gain_col = col

        # If no column provides information gain, return majority class
        if highest_gain == 0:
            return data[self.target_column].mode()[0]

        # Create subtree
        tree = {highest_gain_col: {}}

        # Split the dataset based on the best column and recurse
        unique_values = data[highest_gain_col].unique()

        for unique_value in unique_values:
            filtered_data = data[data[highest_gain_col] == unique_value]
            subset_without_column = filtered_data.drop(columns=[highest_gain_col])        
            subtree = self.build_tree(subset_without_column, C45)
            tree[highest_gain_col][unique_value] = subtree

        return tree

    def calculate_split_information(self, data, column):
        """
        Calculates the split information for a given feature (used in C4.5 algorithm).
        
        Args:
        - data: A pandas DataFrame containing the dataset.
        - column: The name of the column to calculate the split information for.

        Returns:
        - The calculated split information value.
        """
        unique_values = data[column].unique()
        total_rows = len(data)
        split_info = 0

        for value in unique_values:
            subset = data[data[column] == value]
            subset_weight = len(subset) / total_rows
            if subset_weight > 0:
                split_info -= subset_weight * np.log2(subset_weight)

        return split_info


    def calculate_weighted_entropy(self, data, split_column):
        """
        Calculates the weighted entropy for a given feature column.
        
        Args:
        - data: A pandas DataFrame containing the dataset.
        - split_column: The name of the column to evaluate for splitting.

        Returns:
        - The weighted entropy for the given column.
        """
        unique_values = data[split_column].unique()
        total_rows = len(data)
        weighted_entropy = 0

        for value in unique_values:
            subset = data[data[split_column] == value]
            probabilities = (
                subset[self.target_column].value_counts(normalize=True).values
            )
            subset_entropy = self.calculate_entropy(probabilities)

            subset_weight = len(subset) / total_rows
            weighted_entropy += subset_weight * subset_entropy

        return weighted_entropy

    def calculate_entropy(self, probabilities, base=2):
        """
        Calculates the entropy for a set of probabilities.
        
        Args:
        - probabilities: A list or array of probabilities.
        - base: The logarithmic base to use. Default is 2 (binary entropy).

        Returns:
        - The calculated entropy value.
        """
        probabilities = np.array(probabilities)
        probabilities = probabilities[probabilities > 0]

        if base == 2:
            log_func = np.log2
        elif base == np.e:
            log_func = np.log
        else:
            log_func = lambda x: np.log(x) / np.log(base)

        return -np.sum(probabilities * log_func(probabilities))


    def prune_tree(self, validation_data, tree=None):
        """
        Prunes the decision tree using a validation dataset to improve generalization.

        Args:
        - tree: The decision tree to prune.
        - validation_data: A pandas DataFrame used for validation.

        Returns:
        - The pruned decision tree.
        """
        if tree is None:
            tree = self.tree

        # Base case: if the tree is a leaf
        if not isinstance(tree, dict):
            return tree

        # Get the root feature and its subtrees
        root = list(tree.keys())[0]
        subtrees = tree[root]

        # Recursively prune subtrees
        for value, subtree in subtrees.items():
            subset = validation_data[validation_data[root] == value]

            # If the subset is empty, skip pruning this branch
            if subset.empty:
                continue

            # Recursively prune the current subtree
            subtrees[value] = self.prune_tree(subset, subtree)

        # Evaluate whether to replace subtree with a leaf
        predictions = self.predict(validation_data.drop(columns=[self.target_column]))
        original_accuracy = (predictions == validation_data[self.target_column]).mean()

        if not validation_data.empty:
            majority_class = validation_data[self.target_column].mode()[0]
            leaf_accuracy = (validation_data[self.target_column] == majority_class).mean()
        else:
            leaf_accuracy = 0  # No data means no accuracy

        # Replace the subtree with a leaf if it improves accuracy
        if leaf_accuracy >= original_accuracy:
            return majority_class

        return tree


    def predict_row(self, row, tree=None):
        """
        Predicts the target value for a single row using the decision tree.

        Args:
        - row: The input data row, typically a pandas Series or dictionary.
        - tree: The current subtree or the full tree to use for prediction (default is None).

        Returns:
        - The predicted value based on the row's feature values.
        
        If the tree reaches a leaf node, the corresponding class value is returned.
        """
        # If no tree is provided, use the full tree (self.tree)
        if tree is None:
            
            tree = self.tree

        # If tree is a leaf node, return the predicted value (the leaf value)
        if not isinstance(tree, dict):
            return tree


        # Get the first feature from the tree (root feature)
        root_feature = next(iter(tree))

        # Get the value of the feature from the current row
        feature_value = row[root_feature]

        # Check if the feature value exists in the tree's options
        if feature_value in tree[root_feature]:
            # Recursively call predict on the subtree corresponding to the feature value
            return self.predict_row(row, tree[root_feature][feature_value])
        else:
            # If the feature value is not seen in the tree, return None
            return None

    def predict(self, dataset, decision_tree = None):
        """
        Predicts the target values for a dataset using the decision tree.

        Args:
        - dataset: A pandas DataFrame containing the data to predict.

        Returns:
        - A pandas Series containing the predicted values for each row in the dataset.
        """
        predictions = []

        for index, row in dataset.iterrows():
            # Predict the target value for the current row and append it to predictions
            predicted_value = self.predict_row(row,tree = decision_tree)
            predictions.append(predicted_value)

        return pd.Series(predictions, index=dataset.index)

    def display_tree(self, tree=None, indent=""):
        """
        Recursively displays the tree structure in a readable format.
        
        Args:
        - tree: The decision tree to display (default is None, which uses `self.tree`).
        - indent: The string to prepend to each line for indentation (default is an empty string).

        Displays the tree by recursively printing each feature and corresponding values.
        """
        if tree is None:
            tree = self.tree

        if not isinstance(tree, dict):
            print(indent + f"--> {tree}")
        else:
            for key, value in tree.items():
                print(indent + f"{key}")
                for sub_key, sub_tree in value.items():
                    print(indent + f"  [{sub_key}]")
                    self.display_tree(sub_tree, indent + "    ")


   