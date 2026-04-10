# Hybrid E-commerce Intelligence System with A* Search and Contextual Bandit Orchestration

## Description
An intelligent e-commerce recommendation system that combines A* search for efficient product navigation with machine learning models for personalization. A contextual bandit (simplified reinforcement learning) orchestrates between collaborative filtering (KNN) and purchase prediction (Random Forest) to adaptively recommend products based on user segments identified through unsupervised clustering.

## Project Components

| Component | Technology / Algorithm | Purpose |
|-----------|----------------------|---------|
| Informed Search | A* Algorithm | Find optimal product path from category to target item using heuristic based on purchase probability |
| User Segmentation | K-Means Clustering | Group users into personas (e.g., bargain hunters, impulse buyers) |
| Collaborative Filtering | K-Nearest Neighbors (KNN) | Generate "users who liked this also liked" recommendations |
| Purchase Prediction | Random Forest Classifier | Predict probability of user buying a specific product |
| Strategy Orchestration | Contextual Bandit (LinUCB) | Decide whether to show KNN or Random Forest recommendations to maximize engagement |
| Data Processing | Pandas, NumPy | Clean and transform RetailRocket e-commerce dataset |
| Evaluation | Scikit-learn, Matplotlib | Measure silhouette score, F1-score, cumulative regret, and CTR |

## Learning Types

| Component | Type of Learning | Algorithm/Method | Purpose |
|-----------|-----------------|------------------|---------|
| Product Path Finding | Informed Search | A* Search | Find optimal path from category to target product using heuristic function |
| User Segmentation | Unsupervised Learning | K-Means Clustering | Group users into personas based on browsing/purchase behavior |
| Collaborative Filtering | Instance-Based Learning | K-Nearest Neighbors (KNN) | Recommend items liked by similar users |
| Purchase Prediction | Supervised Learning (Classification) | Random Forest | Predict probability a user will buy a specific item |
| Strategy Orchestration | Reinforcement Learning (Simplified) | Contextual Bandit (LinUCB) | Decide which recommendation method to show to maximize engagement |

## Dataset
- **Source:** RetailRocket Dataset (Kaggle)
- **Size:** ~2.5 million e-commerce events (views, carts, purchases)
- **Features:** User ID, Item ID, Event type, Timestamp, Price, Category
