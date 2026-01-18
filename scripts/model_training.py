import os
import pandas as pd

# -------------------------------
# 1. Set up paths
# -------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "output")

# -------------------------------
# 2. Load user-item matrix
# -------------------------------
user_item_path = os.path.join(DATA_DIR, "user_item_matrix.csv")

print("Loading user-item interaction matrix...")
user_item_df = pd.read_csv(user_item_path, index_col=0)
print("✅ User-item matrix loaded successfully!")

# -------------------------------
# 3. Quick check
# -------------------------------
print("Shape of user-item matrix:", user_item_df.shape)
print(user_item_df.iloc[:5, :5])

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# -------------------------------
# 4. Build Item-Item Similarity Matrix
# -------------------------------

print("\nComputing item-item similarity matrix...")

# Transpose so rows = items, columns = users
item_user_matrix = user_item_df.T

# Compute cosine similarity
item_similarity = cosine_similarity(item_user_matrix)

# Convert to DataFrame for readability
item_similarity_df = pd.DataFrame(
    item_similarity,
    index=item_user_matrix.index,
    columns=item_user_matrix.index
)

print("✅ Item-item similarity matrix computed!")
print(item_similarity_df.iloc[:5, :5])

# -------------------------------
# 5. Generate Recommendations
# -------------------------------

def recommend_items(user_id, user_item_df, item_similarity_df, top_n=5):
    """
    Recommend top N items for a given user based on item similarity
    """
    # Get user ratings
    user_ratings = user_item_df.loc[user_id]

    # Items already rated by the user
    rated_items = user_ratings[user_ratings > 0].index

    # Scores for candidate items
    scores = {}

    for item in rated_items:
        similar_items = item_similarity_df[item]

        for similar_item, similarity_score in similar_items.items():
            if user_ratings[similar_item] == 0:
                scores[similar_item] = scores.get(similar_item, 0) + similarity_score

    # Sort items by score
    ranked_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return ranked_items[:top_n]


# -------------------------------
# 6. Test Recommendation for One User
# -------------------------------

test_user_id = 1
recommendations = recommend_items(
    test_user_id,
    user_item_df,
    item_similarity_df,
    top_n=5
)

print(f"\nTop recommendations for User {test_user_id}:")
for item_id, score in recommendations:
    print(f"Item {item_id} | Score: {score:.4f}")

from sklearn.metrics import mean_squared_error
import math

# -------------------------------
# 7. Model Evaluation (RMSE)
# -------------------------------

def predict_rating(user_id, item_id, user_item_df, item_similarity_df):
    """
    Predict rating for a given user-item pair
    """
    user_ratings = user_item_df.loc[user_id]
    rated_items = user_ratings[user_ratings > 0]

    if rated_items.empty:
        return 0

    similarities = item_similarity_df[item_id][rated_items.index]
    weighted_sum = (similarities * rated_items).sum()
    similarity_sum = similarities.sum()

    if similarity_sum == 0:
        return 0

    return weighted_sum / similarity_sum


# Evaluate on a small sample
actual_ratings = []
predicted_ratings = []

print("\nEvaluating model performance...")

for user_id in user_item_df.index[:50]:  # small subset for speed
    user_rated_items = user_item_df.loc[user_id]
    rated_items = user_rated_items[user_rated_items > 0]

    for item_id, actual_rating in rated_items.head(2).items():
        pred_rating = predict_rating(
            user_id,
            item_id,
            user_item_df,
            item_similarity_df
        )

        if pred_rating > 0:
            actual_ratings.append(actual_rating)
            predicted_ratings.append(pred_rating)

rmse = math.sqrt(mean_squared_error(actual_ratings, predicted_ratings))
print(f"✅ RMSE (initial benchmark): {rmse:.4f}")
