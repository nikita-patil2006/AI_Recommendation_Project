import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from random import sample

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "output")

# Load matrix
user_item_path = os.path.join(DATA_DIR, "user_item_matrix.csv")
print("Loading user-item matrix...")

user_item_df = pd.read_csv(user_item_path, index_col=0)
print("Matrix loaded. Shape:", user_item_df.shape)

# --------- Use subset of items for speed ----------
subset_items = user_item_df.columns[:1500]   # change to 3000 later
user_item_df = user_item_df[subset_items]

# Build item similarity model
print("Building item similarity model...")
item_user_matrix = user_item_df.T
item_similarity = cosine_similarity(item_user_matrix)

item_similarity_df = pd.DataFrame(
    item_similarity,
    index=item_user_matrix.index,
    columns=item_user_matrix.index
)

print("Item similarity model ready!")

# -------------------------------
# Recommendation Function
# -------------------------------
def recommend_items(user_ratings, top_n=5):
    rated_items = user_ratings[user_ratings > 0].index
    scores = {}

    for item in rated_items:
        similar_items = item_similarity_df[item]

        for similar_item, sim_score in similar_items.items():
            if user_ratings[similar_item] == 0:
                scores[similar_item] = scores.get(similar_item, 0) + sim_score

    ranked_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [item for item, _ in ranked_items[:top_n]]

# -------------------------------
# Train-Test Evaluation
# -------------------------------
def precision_recall_f1(user_id, k=5):
    user_ratings = user_item_df.loc[user_id]
    rated_items = user_ratings[user_ratings > 0].index.tolist()

    # Skip users with very few ratings
    if len(rated_items) < 5:
        return None

    # Split train/test
    test_items = set(sample(rated_items, int(0.2 * len(rated_items))))
    temp_ratings = user_ratings.copy()

    # Hide test items
    temp_ratings[list(test_items)] = 0

    # Recommend
    rec_items = set(recommend_items(temp_ratings, top_n=k))

    # Metrics
    hit = len(test_items & rec_items)
    precision = hit / k
    recall = hit / len(test_items)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    return precision, recall, f1

# -------------------------------
# Run Evaluation
# -------------------------------
precisions, recalls, f1s = [], [], []

print("\nEvaluating model...")

for user_id in user_item_df.index[:200]:   # first 200 users
    result = precision_recall_f1(user_id, k=5)
    if result:
        p, r, f1 = result
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

print("\n--- Evaluation Results (Top-5) ---")
print("Average Precision:", round(np.mean(precisions), 4))
print("Average Recall:", round(np.mean(recalls), 4))
print("Average F1-score:", round(np.mean(f1s), 4))
