import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

cols = ["area", "perimeter", "compactness", "length", "width", "asymmetry", "groove", "class"]
df = pd.read_csv("../Dataset/seeds_dataset.txt", names=cols, sep="\s+")
# print(df)

# for i in range(len(cols) - 1):
#     for j in range(i + 1, len(cols) - 1):
#         x_label = cols[i]
#         y_label = cols[j]
#         sns.scatterplot(x=x_label, y=y_label, data=df, hue="class")
#         plt.show()

""" K-means clustering """
from sklearn.cluster import KMeans

x = "perimeter"
y = "asymmetry"
X = df[[x, y]].values  # k-means two x (x to x graph)

kmeans = KMeans(n_clusters=3).fit(X)  # n_cluster(k) we know before
clusters = kmeans.labels_
cluster_df = pd.DataFrame(np.hstack((X, clusters.reshape(-1, 1))), columns=[x, y, "class"])

# K Means classes
sns.scatterplot(x=x, y=y, hue="class", data=cluster_df)
plt.plot()
plt.show()

# Original classes
sns.scatterplot(x=x, y=y, hue="class", data=df)
plt.plot()
plt.show()

""" Higher Dimension(All X) """
X = df[cols[:-1]].values
kmeans = KMeans(n_clusters=3).fit(X)
cluster_df = pd.DataFrame(np.hstack((X, kmeans.labels_.reshape(-1, 1))), columns=df.columns)

# K Means classes
sns.scatterplot(x=x, y=y, hue="class", data=cluster_df)
plt.plot()
plt.show()

# Original classes
sns.scatterplot(x=x, y=y, hue="class", data=df)
plt.plot()
plt.show()

""" Principal Component Analysis (PCA) """
""" minimize residuals, maximize varience """
from sklearn.decomposition import PCA
# two dimension n_components = 2
pca = PCA(n_components=2)
# X (220,7) , transform_x = (220,2)
X = df[cols[:-1]].values
transformed_x = pca.fit_transform(X) # take in 7 dimension into 2 dimension representation


kmeans_pca_df = df.DataFrame(np.hstack((transformed_x,kmeans.labels_.reshape(-1,1))),columns = ["pca1","pca2","cla"])
truth_pca_df = df.DataFrame(np.hstack((transformed_x,df["class"].values.reshape(-1,1))),columns = ["pca1","pca2","cla"])

# K Means classes
sns.scatterplot(x="pca1", y="pca2", hue="class", data=kmeans_pca_df)
plt.plot()
plt.show()

# Truth classes
sns.scatterplot(x="pca1", y="pca2", hue="class", data=truth_pca_df)
plt.plot()
plt.show()