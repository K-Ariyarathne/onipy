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

class decisionTree:
    def __init__(self):
        pass
        

    def create_treeC45(self,data, target_column):

        self.target_column = target_column
        self.data = data

        probabilities = data[target_column].value_counts(normalize=True).values
        df_entropy = self.calculate_entropy(probabilities)

         


    def calculate_weighted_entropy(self, split_column):
        """
        Calculate the weighted entropy for a given split.

        :param data: The dataset (Pandas DataFrame).
        :param split_column: The column to split the data on.
        :param target_column: The column to calculate entropy for (e.g., the target variable).
        :return: The weighted entropy of the split.
        """
        # Get the unique values in the split column
        unique_values =  self.data[split_column].unique()
        total_rows = len( self.data)
        
        weighted_entropy = 0
        
        # Iterate through each unique value to calculate its contribution
        for value in unique_values:
            subset =  self.data[ self.data[split_column] == value]
            probabilities = subset[self.target_column].value_counts(normalize=True).values  # Proportions of target classes
            subset_entropy = self.calculate_entropy(probabilities)  # Entropy of the subset
            
            # Calculate the weighted contribution of this subset
            subset_weight = len(subset) / total_rows  # Proportion of rows in this subset
            weighted_entropy += subset_weight * subset_entropy
        
        return weighted_entropy

    
    def calculate_entropy(self,probabilities, base=2):

        probabilities = np.array(probabilities)
        if base == 2:
            log_func = np.log2
        elif base == np.e:
            log_func = np.log
        else:
            log_func = lambda x: np.log(x) / np.log(base)
        return -np.sum(probabilities * log_func(probabilities))


dt = decisionTree(data = df)