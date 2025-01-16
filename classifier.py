import numpy as np
import pandas as pd

# Load the dataset
df = pd.read_csv("Titanic-Dataset.csv")
df = df.drop(columns=["PassengerId", "Name", "Cabin", "Embarked", "Ticket"])

mean_age = df["Age"].mean()
df["Age"] = df["Age"].fillna(mean_age)

age_bins = [0, 12, 18, 60, np.inf]
age_labels = ["Child", "Teen", "Adult", "Senior"]
df["Age"] = pd.cut(df["Age"], bins=age_bins, labels=age_labels, right=False)

fare_bins = [-1, 7.91, 14.454, 31.0, np.inf]
fare_labels = ["Low", "Medium", "High", "Very High"]
df["Fare"] = pd.cut(df["Fare"], bins=fare_bins, labels=fare_labels, right=False)

df["Age"] = df["Age"].cat.codes
df["Fare"] = df["Fare"].cat.codes
df["Sex"] = pd.factorize(df["Sex"])[0]

df = df.iloc[:, 1:].assign(Survived=df.iloc[:, 0])

print(df.head())


class DecisionTree:
    def __init__(self):
        self.tree = None  # To store the tree structure

    def create_treeC45(self, data, target_column):
        """
        Creates a decision tree using the C4.5 algorithm.
        """
        self.target_column = target_column

        # Recursively build the tree
        self.tree = self.build_tree(data)

    def build_tree(self, data):
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

            # Calculate information gain
            weighted_entropy = self.calculate_weighted_entropy(data, col)
            info_gain = df_entropy - weighted_entropy

            # Track the column with the highest gain
            if info_gain > highest_gain:
                highest_gain = info_gain
                highest_gain_col = col

        # If no column provides information gain, return majority class
        if highest_gain == 0:
            return data[self.target_column].mode()[0]

        # Create subtree
        tree = {highest_gain_col: {}}

        # Split the dataset based on the best column and recurse
        for value in data[highest_gain_col].unique():
            subset = data[data[highest_gain_col] == value].drop(
                columns=[highest_gain_col]
            )
            subtree = self.build_tree(subset)
            tree[highest_gain_col][value] = subtree

        return tree

    def calculate_weighted_entropy(self, data, split_column):
        """
        Calculates the weighted entropy for a column.
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
        Calculates entropy for a set of probabilities.
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

    def display_tree(self, tree=None, indent=""):
        """
        Recursively display the tree in a readable format.
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


dt = DecisionTree()
dt.create_treeC45(df, "Survived")
print("Generated Decision Tree:")
dt.display_tree()
