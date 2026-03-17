import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")

print(data.columns)

X = data[['Sleep Duration', 'Quality of Sleep', 'Stress Level']].copy()
X = X.dropna()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

data['Cluster'] = clusters

plt.scatter(data['Sleep Duration'], data['Quality of Sleep'], c=clusters)
plt.xlabel("Sleep Duration")
plt.ylabel("Quality of Sleep")
plt.title("Sleep Pattern Clustering")
plt.show()

print(data.groupby('Cluster')[['Sleep Duration', 'Quality of Sleep', 'Stress Level']].mean())

data.to_csv("clustered_sleep_data.csv", index=False)