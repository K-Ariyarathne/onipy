import numpy as np
import pandas as pd

df = pd.read_csv('Titanic-Dataset.csv')
df = df.drop(columns=['PassengerId', 'Name','Cabin',"Embarked","Ticket"])


mean_age = df['Age'].mean()
df['Age'] = df['Age'].fillna(mean_age)

na_counts = df.isna().sum()
df['Age'] = df['Age'].astype(int)
df['Sex'] = pd.factorize(df['Sex'])[0]
df['Sex'] = pd.factorize(df['Sex'])[0]

df = df.iloc[:, 1:].assign(Survived=df.iloc[:, 0])

class DecisionTree:
    def __init__(self):
        self.tree = None  # To store the tree structure

    def create_treeC45(self, data, target_column):
        self.target_column = target_column
        self.data = data

        # Calculate the dataset's initial entropy
        probabilities = data[target_column].value_counts(normalize=True).values
        df_entropy = self.calculate_entropy(probabilities)

        highest_gain_col = ''
        highest_gain = 0

        for col in data.columns:
            if col == target_column:  # Skip the target column
                continue

            # Calculate information gain
            weighted_entropy = self.calculate_weighted_entropy(col)
            info_gain = df_entropy - weighted_entropy

            # Track the column with the highest gain
            if info_gain > highest_gain:
                highest_gain = info_gain
                highest_gain_col = col

        print(f"Best split column: {highest_gain_col}")
        print(f"Highest information gain: {highest_gain}")

        # Placeholder for actual tree-building logic
        self.tree = {"split_column": highest_gain_col, "info_gain": highest_gain}

    def calculate_weighted_entropy(self, split_column):
        # Get the unique values in the split column
        unique_values = self.data[split_column].unique()
        total_rows = len(self.data)

        weighted_entropy = 0

        for value in unique_values:
            subset = self.data[self.data[split_column] == value]
            probabilities = subset[self.target_column].value_counts(normalize=True).values
            subset_entropy = self.calculate_entropy(probabilities)

            subset_weight = len(subset) / total_rows
            weighted_entropy += subset_weight * subset_entropy

        return weighted_entropy

    def calculate_entropy(self, probabilities, base=2):
        probabilities = np.array(probabilities)
        probabilities = probabilities[probabilities > 0]  
        
        if base == 2:
            log_func = np.log2
        elif base == np.e:
            log_func = np.log
        else:
            log_func = lambda x: np.log(x) / np.log(base)

        return -np.sum(probabilities * log_func(probabilities))


dt = DecisionTree()
dt.create_treeC45(df,"Survived")