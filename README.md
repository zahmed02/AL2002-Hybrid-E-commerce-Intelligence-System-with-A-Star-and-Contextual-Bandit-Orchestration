# Hybrid E-commerce Intelligence System with A* Search and Contextual Bandit Orchestration

## Description
An intelligent e-commerce recommendation system that combines A* search for efficient product navigation with machine learning models for personalization. A contextual bandit (simplified reinforcement learning) orchestrates between collaborative filtering (KNN) and purchase prediction (Random Forest) to adaptively recommend products based on user segments identified through unsupervised clustering.

## Computational Paradigm

| Component | Algorithm | Learning Paradigm | Purpose |
|-----------|-----------|-------------------|---------|
| Product Path Finding | A* Search | Informed Search | Find optimal product category path |
| User Segmentation | K-Means | Unsupervised | Group users into personas |
| Collaborative Filtering | KNN | Instance-Based | Recommend based on similar users |
| Purchase Prediction | Random Forest | Supervised (Classification) | Predict buy probability |
| Strategy Orchestration | Contextual Bandit | Reinforcement Learning | Choose best recommendation method |
| Data Preprocessing | Pandas / NumPy | Data Engineering | Clean and transform dataset |
| Baseline Comparison | Linear Regression | Supervised (Regression) | Compare against Random Forest |

## Evaluation Metrics: 
Silhouette Score (clustering), F1-Score & RMSE (Random Forest + Regression baseline), Cumulative Regret (Bandit), Path Cost (A*)

##Tools:
Python, pandas, numpy, scikit-learn, matplotlib

## Dataset
- **Source:** RetailRocket Dataset (Kaggle)
- **Size:** ~2.5 million e-commerce events (views, carts, purchases)
- **Features:** User ID, Item ID, Event type, Timestamp, Price, Category
