import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('secom.data',header=None,sep='\s+')
print(df.head())

labels=pd.read_csv('secom_labels.data',header=None,sep='\s+')
print(labels.head())

df['label']=labels[0].values
print(df.shape)

print(f"Value counts of labels:\n{df['label'].value_counts()}")

missing_per_column=df.isnull().sum()
print(f"Missing values per column:\n{missing_per_column}")

total_missing=missing_per_column.sum()
print(f"Total missing values in the dataset: {total_missing}")

missing_ratio=missing_per_column/len(df)*100
missing_ratio=missing_ratio[missing_ratio>0].sort_values(ascending=False)
print(f"Missing value ratio per column:\n{missing_ratio}")

plt.figure(figsize=(12,6))
sns.barplot(x=missing_ratio.index, y=missing_ratio.values)
plt.xticks(rotation=90)
plt.title('Missing Value Ratio per Column')
plt.ylabel('Missing Value Ratio (%)')
plt.show()

plt.figure(figsize=(12,6))
sns.scatterplot(x=df.index, y=df.isnull().sum(axis=1))
plt.title("Features with Missing Values per Sample")
plt.xlabel("Sample Index")
plt.ylabel("Number of Missing Features")
plt.show()
