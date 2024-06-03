import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Data Cleaning:

# 1. Handle Missing Values:
#    - 'Age': Fill missing ages with the median age.
train_df['Age'].fillna(train_df['Age'].median(), inplace=True)
test_df['Age'].fillna(test_df['Age'].median(), inplace=True)

#    - 'Cabin': Impute missing cabin values with 'Unknown'
train_df['Cabin'].fillna('Unknown', inplace=True)
test_df['Cabin'].fillna('Unknown', inplace=True)

#    - 'Embarked': Fill missing Embarked values with the most frequent embarkation point
most_freq_embarked = train_df['Embarked'].mode()[0] 
train_df['Embarked'].fillna(most_freq_embarked, inplace=True)
test_df['Embarked'].fillna(most_freq_embarked, inplace=True)

# 2. Feature Engineering:
#    - Extract title from 'Name'
train_df['Title'] = train_df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
test_df['Title'] = test_df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

#    - Create 'FamilySize' and 'IsAlone' features
train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1
test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch'] + 1

train_df['IsAlone'] = 0
train_df['IsAlone'].loc[train_df['FamilySize'] == 1] = 1

test_df['IsAlone'] = 0
test_df['IsAlone'].loc[test_df['FamilySize'] == 1] = 1

# 3. Data Transformation:
#    - Convert 'Sex', 'Embarked', and 'Title' to numerical using one-hot encoding
train_df = pd.get_dummies(train_df, columns=['Sex', 'Embarked', 'Title'], drop_first=True)
test_df = pd.get_dummies(test_df, columns=['Sex', 'Embarked', 'Title'], drop_first=True)

# EDA:

# 1. Basic Statistics:
print(train_df.describe())

# 2. Visualizations:
#    - Survival Rates by Gender
sns.barplot(x='Survived', y='Sex_male', data=train_df)
plt.title('Survival Rates by Gender')
plt.show()

#    - Survival Rates by Class
sns.barplot(x='Survived', y='Pclass', data=train_df)
plt.title('Survival Rates by Passenger Class')
plt.show()

#    - Relationship between Age and Survival
sns.histplot(x='Age', hue='Survived', data=train_df, multiple='stack')
plt.title('Age Distribution by Survival')
plt.show()

#    - Survival Rates by Family Size
sns.boxplot(x='Survived', y='FamilySize', data=train_df)
plt.title('Survival Rates by Family Size')
plt.show()

# 3. Correlation Matrix:
correlation_matrix = train_df.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.title('Correlation Matrix')
plt.show()

# 4. Pair Plots:
sns.pairplot(train_df[['Survived', 'Pclass', 'Age', 'Fare', 'FamilySize']], hue='Survived')
plt.show()
