# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Step 1: Load the dataset
data = pd.read_csv(r"C:\Users\PC\OneDrive\Desktop\snapdeal_project\Synthetic_Mall_Customers.csv")
print("First 5 rows of the data:")
print(data.head())

# Step 2: Select features for clustering
features = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
print("Data shape used for clustering:", features.shape)

# Step 3: Standardize the features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(features)

# Step 4: Determine the optimal number of clusters using the Elbow Method
wcss = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

# Plotting the Elbow curve
plt.figure(figsize=(8, 5))
plt.plot(K_range, wcss, marker='o', linestyle='--')
plt.title("Elbow Method - Optimal Number of Clusters")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("WCSS")
plt.grid(True)
plt.show()

# Step 5: Run K-Means with optimal number of clusters 
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
cluster_labels = kmeans.fit_predict(scaled_data)

# Step 6: Add cluster labels to the original dataset
data['Cluster'] = cluster_labels
print("\nData with Cluster Labels:")
print(data.head())

# Step 7: Visualize clusters using scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'], 
            c=data['Cluster'], cmap='viridis', s=50)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Segments based on Clustering')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()

# Step 8: Cluster count visualization
plt.figure(figsize=(6,4))
sns.countplot(x='Cluster', data=data, palette='Set2')
plt.title('Number of Customers per Cluster')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.grid(True)
plt.show()

# Step 9: Summary statistics per cluster
print("\nMean values per cluster:")
print(data.groupby('Cluster').mean(numeric_only=True))

print("\nMedian values per cluster:")
print(data.groupby('Cluster').median(numeric_only=True))

print("\nStandard deviation per cluster:")
print(data.groupby('Cluster').std(numeric_only=True))
