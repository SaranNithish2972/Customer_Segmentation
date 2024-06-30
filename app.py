import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Title of the web app
st.title('Mall Customers Segmentation')

# Function to load the default dataset
def load_default_data():
    return pd.read_csv('Mall_Customers.csv')

# Load the dataset
df = load_default_data()
st.write("Using default dataset as no file was uploaded.")
st.write("Dataset Head:")
st.dataframe(df.head())

buffer = io.StringIO()
df.info(buf=buffer)
s = buffer.getvalue()
st.text(s)

# Selecting features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values

# Standardizing the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow Method for finding optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot the elbow graph
st.write("Elbow Method for Optimal Number of Clusters")
plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
st.pyplot(plt)

# KMeans Clustering
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)
Y = kmeans.fit_predict(X_scaled)

# Add cluster to the dataframe
df['Cluster'] = Y

# Display the clusters
st.write("Clusters in the dataset:")
st.dataframe(df.head())

# Visualize the clusters
st.write("Cluster Visualization")
plt.figure(figsize=(10, 5))
sns.scatterplot(x=df['Annual Income (k$)'], y=df['Spending Score (1-100)'], hue=df['Cluster'], palette='viridis')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
st.pyplot(plt)

# User input for prediction
st.write("Predict the cluster for a new customer:")
gender = st.selectbox("Gender", ("Male", "Female"))
age = st.number_input("Age", min_value=1, max_value=100, value=25)
annual_income = st.number_input("Annual Income (k$)", min_value=0, max_value=200, value=50)
spending_score = st.number_input("Spending Score (1-100)", min_value=0, max_value=100, value=50)

if st.button("Predict Cluster"):
    # Preprocess the new data
    new_data = [[annual_income, spending_score]]
    new_data_scaled = scaler.transform(new_data)
    
    # Predict the cluster
    predicted_cluster = kmeans.predict(new_data_scaled)
    
    st.write(f"The predicted cluster for the new customer is: {predicted_cluster[0]}")
