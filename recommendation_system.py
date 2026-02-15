import pandas as pd
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# 1. Create a dummy dataset of 10 users (In real life, this would be a CSV file with 50,000+ rows)
data = {
    'User_ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Age': [25, 26, 22, 45, 48, 42, 30, 31, 35, 40],
    'Time_Spent_Minutes': [120, 110, 130, 15, 20, 10, 80, 85, 90, 25],
    'Items_Clicked': [50, 45, 55, 5, 8, 3, 30, 32, 28, 10]
}

df = pd.DataFrame(data)

# 2. Select the features we want to use for clustering (Ignore User_ID)
features = df[['Age', 'Time_Spent_Minutes', 'Items_Clicked']]

# 3. Initialize and Train the K-Means Model (We will create 3 clusters)
kmeans = KMeans(n_clusters=3, random_state=42)
df['User_Segment'] = kmeans.fit_predict(features)

# 4. Analyze the Clusters
print("--- User Segmentation Results ---\n")
print(df[['User_ID', 'User_Segment']])

print("\n--- Recommendation Logic ---")
print("If a new product is heavily clicked by Users in Segment 0,")
print("the system will automatically recommend it to all other users in Segment 0!")