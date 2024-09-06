import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df=pd.read_csv('Mall_Customers.csv')
df.head()

df.describe()
df.info()
df.isnull().sum()

df['Gender'].value_counts()

df['Age'].value_counts()

sn.distplot(df['Spending Score (1-100)'])

cols=['CustomerID', 'Gender', 'Age', 'Annual Income (k$)','Spending Score (1-100)']
for i in cols:
    plt.figure()
    sn.histplot(df[i])

cols=['CustomerID', 'Age', 'Annual Income (k$)','Spending Score (1-100)']
for x in cols:
    plt.figure()
    sn.kdeplot(df[x],shade=True)

sn.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
X


sn.scatterplot(X, x= "Annual Income (k$)", y= "Spending Score (1-100)")
plt.show()




sse = []
for i in range(1, 11):
    km = KMeans(n_clusters =  i)
    km.fit(df[['Annual Income (k$)', 'Spending Score (1-100)']])
    sse.append(km.inertia_)

sse


plt.xlabel('Number of clusters')
plt.ylabel("SSE")
plt.plot(range(1,11), sse)


km = KMeans(n_clusters = 5)
predicted = km.fit_predict(df[['Annual Income (k$)', 'Spending Score (1-100)']])
predicted

df['Cluster'] = predicted
df


df1 = df[df.Cluster==0] 
df2 = df[df.Cluster==1] 
df3 = df[df.Cluster==2]
df4 = df[df.Cluster==3]
df5 = df[df.Cluster==4]
plt.scatter(df1['Annual Income (k$)'],df1['Spending Score (1-100)'],color='green')
plt.scatter(df2['Annual Income (k$)'],df2['Spending Score (1-100)'],color='red')
plt.scatter(df3['Annual Income (k$)'],df3['Spending Score (1-100)'],color='black')
plt.scatter(df4['Annual Income (k$)'],df4['Spending Score (1-100)'],color='c')
plt.scatter(df5['Annual Income (k$)'],df5['Spending Score (1-100)'],color='blue')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],
            color='purple',marker='*',label='centroid')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()












