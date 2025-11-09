import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Step 1: Load the dataset
df = pd.read_csv("sales_data_sample.csv", encoding='latin1')

# Step 2: View basic information about data
df.head()          # Shows first 5 rows
df.tail()          # Shows last 5 rows
df.describe()      # Summary statistics (mean, std, min, max, etc.)
df.dtypes          # Shows data types of each column
df.isnull()        # Checks for missing (null) values
df.isnull().sum()  # Counts number of null values per column

# Step 3: Select only numeric columns useful for clustering
df2 = df[['ORDERNUMBER','QUANTITYORDERED','ORDERLINENUMBER','QTR_ID','MONTH_ID','YEAR_ID','MSRP']]

# Step 4: Drop missing values (if any)
df2 = df2.dropna()
df2.isnull().sum()  # Verify no nulls remain

# Step 5: Standardize the data (mean=0, std=1)
# This ensures all features are on the same scale for KMeans
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df2)

# Step 6: Use the Elbow Method to find optimal number of clusters (k)
distortions = []
k_range = range(1, 12)
for n in k_range:
    # Create KMeans model for each number of clusters
    kmean = KMeans(n_clusters=n, random_state=45, n_init=11)
    
    # Fit model on the scaled data
    kmean.fit(scaled_data)
    
    # Store the inertia (sum of squared distances to nearest cluster center)
    distortions.append(kmean.inertia_)

# Step 7: Plot the Elbow curve
plt.plot(k_range, distortions, 'bx-')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Distortion')
plt.show()


# -------------------- THEORY --------------------
# 🧠 THEORY:
# 1️⃣ Standardization:
#    Since all features have different units/scales (like QUANTITYORDERED vs MSRP),
#    we use StandardScaler() to normalize them so that each contributes equally
#    to the distance calculation in KMeans.

# 2️⃣ KMeans:
#    - It divides data into 'k' clusters.
#    - Each point belongs to the cluster with the nearest centroid (center point).
#    - The algorithm minimizes the 'inertia' value, which is the sum of squared
#      distances between data points and their respective cluster centers.

# 3️⃣ Inertia:
#    - It measures how tightly the data points are grouped within each cluster.
#    - Lower inertia means better compact clusters.
#    - Formula: Inertia = Σ (distance of each point to its cluster center)^2

# 4️⃣ Elbow Method:
#    - We try multiple values of 'k' and record their inertia.
#    - When plotted, inertia decreases rapidly at first, then slows down.
#    - The point where the curve bends like an "elbow" is the optimal k.
#    - Beyond this point, adding more clusters doesn’t significantly
#      improve the model (inertia decreases very little).

# Example Interpretation:
# - If your graph bends at k=2 and continues bending slightly till k=5,
#   it means clusters between 2 and 5 could represent natural groupings in data.
#   But usually, we choose the first strong bend (the “elbow”), e.g., k=2 or 3.
