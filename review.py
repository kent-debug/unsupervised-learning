import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats.mstats import winsorize

# Load data
r3 = pd.read_csv(r'C:\Users\KENNY PC\OneDrive\Desktop\Ai project\Unsupervised learning models\unsupervised-learning\Social_Media_Post.csv')

# Drop columns with high missing values
r3_cleaned = r3.drop(columns=['status_id','Column1','Column2','Column3','Column4'])

# Convert 'status_published' to datetime
r3_cleaned['status_published'] = pd.to_datetime(r3_cleaned['status_published'])

# Handling outliers with winsorization
continuous_vars = ['num_reactions', 'num_comments', 'num_shares', 'num_likes', 'num_loves', 'num_wows', 'num_hahas', 'num_sads']
Q1 = r3_cleaned[continuous_vars].quantile(0.25)
Q3 = r3_cleaned[continuous_vars].quantile(0.75)
IQR = Q3 - Q1
outliers_iqr = ((r3_cleaned[continuous_vars] < (Q1 - 1.5 * IQR)) | (r3_cleaned[continuous_vars] > (Q3 + 1.5 * IQR))).any(axis=1)
r3_cleaned[continuous_vars] = r3_cleaned[continuous_vars].apply(lambda x: winsorize(x, limits=[0.05, 0.05]))

# Normalize data
scaler = StandardScaler()
r3_scaled = scaler.fit_transform(r3_cleaned[continuous_vars])

# Determine optimal number of clusters using elbow method
inertia = []
k_range = range(1, 6)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(r3_scaled)
    inertia.append(kmeans.inertia_)

# Plot elbow curve
plt.figure(figsize=(8, 6))
plt.plot(k_range, inertia, marker='o', linestyle='-', color='b')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Curve for Optimal k')
plt.xticks(k_range)
plt.grid(True)
plt.show()

# Perform clustering with optimal k
optimal_k = 3  # Adjust based on elbow curve analysis
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(r3_scaled)

# Assign cluster labels
cluster_labels = kmeans.labels_
r3_cleaned['cluster'] = cluster_labels

# Visualize clusters
sns.pairplot(r3_cleaned, vars=continuous_vars, hue='cluster', palette='viridis')
plt.show()

# Analyze and interpret results based on cluster characteristics
print(r3_cleaned[['status_type', 'status_published', 'cluster']])
