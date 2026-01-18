AI Enabled Recommendation Engine for an E-commerce Platform
📌 Project Overview

The goal of this project is to design and develop an AI-driven recommendation engine for an e-commerce platform. The system leverages user behavior and product data to generate personalized product recommendations. By analyzing historical interactions between users and items, the recommendation engine aims to improve user engagement and sales on the platform.

The project is developed in milestones, progressing from data preparation to model building and evaluation.

🧩 Milestone 1 – Data Preparation
🎯 Objectives

Prepare clean, structured datasets suitable for model development

Handle missing values, duplicates, and data inconsistencies

Merge multiple datasets into a single dataset for convenience

Construct a user–item interaction matrix to be used for recommendation algorithms

✅ Tasks Completed

Loaded user, movie, ratings, links, and tags datasets from the MovieLens dataset

Removed duplicate rows from all datasets

Checked and handled missing values:

Filled missing tmdbId values in links.csv with 0

Merged datasets into combined_cleaned.csv for easy access

Built a user–item interaction matrix where:

Rows represent users (userId)

Columns represent movies (movieId)

Values represent ratings

Saved processed outputs to the output/ folder for further model development

▶️ How to Run

(Optional) Place the MovieLens CSV files in a data/ folder in your local copy of the repository.
Run the data preparation script:

python scripts/data_prep.py

🧠 Milestone 2 – Model Building
🎯 Objectives

Develop and train the core recommendation model

Select and implement an appropriate recommendation algorithm

Perform initial model evaluation and benchmarking

✅ Tasks Completed

Implemented an Item-Based Collaborative Filtering recommendation algorithm

Used cosine similarity to compute item–item similarity from the user–item interaction matrix

Generated personalized recommendations for users based on item similarity scores

Built a rating prediction function using weighted similarity scores

Evaluated initial model performance using Root Mean Square Error (RMSE)

📊 Model Details

Algorithm: Item-Based Collaborative Filtering

Similarity Metric: Cosine Similarity

Dataset Size:

Users: 610

Items (Movies): 9724

📈 Initial Performance Benchmark

RMSE: 0.8903 (evaluated on a subset of users and items)

This benchmark confirms that the recommendation model is functioning correctly and meets initial performance expectations.

▶️ How to Run

Ensure the prepared user–item matrix is available in the output/ folder.
Run the model training and evaluation script:

python scripts/model_training.py
