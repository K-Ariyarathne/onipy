from classifier import DecisionTree

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


# Step 1: Shuffle the dataset using numpy
shuffled_df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Step 2: Split the dataset into training and test sets (80-20 split)
train_size = int(0.9 * len(shuffled_df))  # 80% for training
train_df = shuffled_df[:train_size]
test_df = shuffled_df[train_size:]

# Step 3: Create and train the decision tree model
dt = DecisionTree()
dt.create_treeID3(train_df, "Survived", C45=True)



predictions = dt.predict(test_df.drop(columns=['Survived']))  
actual = test_df['Survived']

accuracy = (predictions == actual).mean()
print(f"Accuracy: {accuracy * 100:.2f}%")
