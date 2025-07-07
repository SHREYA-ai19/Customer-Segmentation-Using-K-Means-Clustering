# K-Means Clustering Project 

# Step 1: Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

#  Step 2: Load Dataset (Mall Customers Data)
df = pd.read_csv("C:/Users/PC/Downloads/Mall_Customers.csv")
print(df.head())

# Step 3: View First Few Rows
print("First 5 rows of the dataset:")
print(df.head())

# Step 4: Basic Info and Missing Values
print("\nData Info:")
print(df.info())

print("\nCheck for Missing Values:")
print(df.isnull().sum())

# Step 5: Exploratory Data Analysis
sns.set(style="whitegrid")

plt.figure(figsize=(6, 4))
sns.histplot(df['Age'], bins=10, kde=True)
plt.title('Distribution of Age')
plt.show()

plt.figure(figsize=(6, 4))
sns.boxplot(x=df['Annual Income (k$)'])
plt.title('Boxplot of Annual Income')
plt.show()

# Step 6: Select Relevant Features for Clustering
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Step 7: Normalize Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 8: Use Elbow Method to Choose Number of Clusters
wcss = []
for i in range(1, 6):  

    kmeans = KMeans(n_clusters=i)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(6, 4))
plt.plot(range(1, 6), wcss, marker='o')  
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS (Inertia)')
plt.grid(True)
plt.show()

# Step 9: Apply K-Means Clustering
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Step 10: Visualize the Clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', 
                hue='Cluster', data=df, palette='Set2', s=100)
plt.title('Customer Segments using K-Means Clustering')
plt.legend(title='Cluster')
plt.show()

# Step 11:Cluster Summary
print("\nCluster Value Counts:")
print(df['Cluster'].value_counts())

# Step 16:customer final segments
print("Here are the final customer segments with cluster numbers:")