# Intelligent Content Recommendation System

A Machine Learning based recommendation engine designed to provide personalized content suggestions by analyzing user behavior and preferences.

## 🚀 Overview
This project implements an **Unsupervised Learning** approach using the **K-Means Clustering** algorithm. It segments users into distinct groups based on interaction patterns (time spent, clicks, age), allowing for highly targeted "Content-Based" or "Collaborative" style recommendations.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (K-Means Clustering)
* **Development Environment:** VS Code / Jupyter Notebook
* **Certified Skills:** IBM Python for Data Science & Data Analysis

## 📈 Key Features
* **User Segmentation:** Automatically groups thousands of users into clusters based on multi-dimensional behavior data.
* **Feature Scaling:** Prepares raw data for clustering to ensure high accuracy in distance-based calculations.
* **Efficient Data Retrieval:** Utilizes Python dictionaries and Hash Maps for fast recommendation lookups.
* **Scalable Architecture:** Designed to handle increasing datasets while maintaining performance.

## 📋 How It Works
1. **Data Preprocessing:** Cleans and structures raw user interaction logs into a Pandas DataFrame.
2. **Cluster Optimization:** Uses the "Elbow Method" logic to determine the optimal number of user segments.
3. **Model Training:** Applies the K-Means algorithm to find centroids for each user cluster.
4. **Recommendation Logic:** When a user is identified in a cluster, the system recommends top-rated content from other users in that same segment.



## 💻 How to Run
1. Install dependencies:
   ```bash
   pip install pandas scikit-learn numpy
