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


---

## 📊 Milestone 3 – Evaluation and Refinement

### 🎯 Objectives

- Evaluate the recommendation model performance  
- Analyze results using standard metrics (Precision, Recall, F1-score)  
- Refine the model and test recommendation scenarios  

### ✅ Tasks Completed

- Implemented a model evaluation script (`model_evaluation.py`)  
- Used a train-test split approach by hiding a portion of user ratings and predicting them  
- Computed evaluation metrics:
  - Precision  
  - Recall  
  - F1-score  
- Tested Top-K recommendation scenario (Top-5 recommendations)  
- Refined recommendation logic and similarity computation to improve accuracy  

### 📈 Evaluation Results (Top-5 Recommendations)

- **Average Precision:** 0.3947  
- **Average Recall:** 0.2060  
- **Average F1-score:** 0.2292  

These results are realistic and acceptable for collaborative filtering on sparse datasets. The evaluation confirms that the recommendation model generates meaningful personalized recommendations.

### ▶️ How to Run

Ensure the user–item interaction matrix is available in the `output/` folder.  
Run the evaluation script:

```bash
python scripts/model_evaluation.py


Movie Recommendation System – Milestone 4
📌 Overview

This project is a Movie Recommendation System built using User-Based Collaborative Filtering with cosine similarity.

Milestone 4 focuses on converting the recommendation model into a complete user-facing application with authentication, navigation, and explainable recommendations.

🚀 Features Added in Milestone 4

🔐 Login & Logout (Email + Password)

📊 Dashboard with user insights

🎬 Personalized movie recommendations

🧠 Reason shown for each recommendation

❄️ Cold-start handling using popularity-based fallback

🧭 Navigation bar (Dashboard, Recommendations, About)

🧠 Algorithm Used

User-Based Collaborative Filtering

Cosine Similarity for user similarity calculation

Popularity-based fallback for new users

⚙️ Tech Stack

Python

Streamlit

Pandas

NumPy

Scikit-learn

▶️ Run the Project
pip install -r requirements.txt
streamlit run app.py
🎯 Objective

To demonstrate how a collaborative filtering model can be integrated into a real-world, user-facing application with explainability and session management.

