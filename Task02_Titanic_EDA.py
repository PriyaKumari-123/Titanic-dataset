
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load Titanic dataset (sample from public repository)
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Display first few records
print("First 5 Records of the Dataset:\n", df.head())

# Shape of dataset
print("\nDataset Shape:", df.shape)

# Checking for missing values
print("\nMissing Values:\n", df.isnull().sum())

# Filling missing 'Age' with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Dropping 'Cabin' due to too many missing values
df.drop(columns=['Cabin'], inplace=True)

# Basic Information
print("\nDataset Info:")
print(df.info())

# EDA: Survival count
plt.figure(figsize=(6,4))
sns.countplot(x='Survived', data=df)
plt.title('Survival Count (0 = Not Survived, 1 = Survived)')
plt.show()

# EDA: Survival by Gender
plt.figure(figsize=(6,4))
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title('Survival by Gender')
plt.show()

# EDA: Age Distribution
plt.figure(figsize=(6,4))
sns.histplot(df['Age'], bins=20, kde=True)
plt.title('Age Distribution')
plt.show()

# EDA: Class Distribution
plt.figure(figsize=(6,4))
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title('Survival by Passenger Class')
plt.show()
