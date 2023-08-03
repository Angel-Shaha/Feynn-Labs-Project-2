# Feynn-Labs-Project-2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
df=pd.read_csv(r'C:\Users\HP\Downloads\Indian automoble buying behavour study 1.0.csv')
df.tail(10)
df.shape
df.info()
X=df.iloc[:, [10,12]].values
from sklearn.cluster import KMeans
wcss=[]
for i in range(1, 11): 
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X) 
    wcss.append(kmeans.inertia_)
    plt.plot(range(1,11),wcss)
plt.title('The elbow method')
plt.xlabel('No. of clusters')
plt.ylabel('WCSS Values')
plt.show()
kmeansmodel = KMeans(n_clusters=5, init='k-means++', random_state=0)
y_kmeans = kmeansmodel.fit_predict(X)
plt.scatter(X[y_kmeans == 0,0], X[y_kmeans == 0,1], s= 50, c="red", label='Customer 1')
plt.scatter(X[y_kmeans == 1,0], X[y_kmeans == 1,1], s= 50, c="black", label='Customer 2')
plt.scatter(X[y_kmeans == 2,0], X[y_kmeans == 2,1], s= 50, c="blue", label='Customer 3')
plt.title('Clusters of customers')
plt.xlabel('Total Salary')
plt.ylabel('Price')
plt.legend()
plt.show()
