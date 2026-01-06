import os
import pandas as pd

# -------------------------------
# 1. Set up paths relative to project root
# -------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------
# 2. CSV file paths
# -------------------------------
movies_path = os.path.join(DATA_DIR, "movies.csv")
ratings_path = os.path.join(DATA_DIR, "ratings.csv")
links_path = os.path.join(DATA_DIR, "links.csv")
tags_path = os.path.join(DATA_DIR, "tags.csv")

# -------------------------------
# 3. Load CSVs
# -------------------------------
print("Loading CSV files...")
movies = pd.read_csv(movies_path)
ratings = pd.read_csv(ratings_path)
links = pd.read_csv(links_path)
tags = pd.read_csv(tags_path)
print("✅ CSV files loaded successfully!\n")

# -------------------------------
# 4. Data Cleaning
# -------------------------------

# Remove duplicate rows
movies.drop_duplicates(inplace=True)
ratings.drop_duplicates(inplace=True)
links.drop_duplicates(inplace=True)
tags.drop_duplicates(inplace=True)

# Check for missing values and handle them
print("Missing values per dataset:")
print("Movies:\n", movies.isna().sum(), "\n")
print("Ratings:\n", ratings.isna().sum(), "\n")
print("Links:\n", links.isna().sum(), "\n")
print("Tags:\n", tags.isna().sum(), "\n")

# Fill missing TMDb IDs with 0 (optional) or drop rows with missing critical info
links['tmdbId'] = links['tmdbId'].fillna(0)

# -------------------------------
# 5. Merge datasets for convenience
# -------------------------------
combined = pd.merge(ratings, movies, on="movieId", how="left")
combined = pd.merge(combined, links, on="movieId", how="left")
combined.to_csv(os.path.join(OUTPUT_DIR, "combined_cleaned.csv"), index=False)
print("✅ Combined cleaned dataset saved as 'combined_cleaned.csv'")

# -------------------------------
# 6. Build User-Item Interaction Matrix
# -------------------------------
# Rows: users, Columns: movies, Values: ratings
user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating')

# Optional: fill NaNs with 0 if your model needs dense matrix
user_item_matrix_filled = user_item_matrix.fillna(0)

# Save matrix to CSV
user_item_matrix_filled.to_csv(os.path.join(OUTPUT_DIR, "user_item_matrix.csv"))
print("✅ User-Item Interaction Matrix saved as 'user_item_matrix.csv'")

# -------------------------------
# 7. Quick checks
# -------------------------------
print("\nUser-Item Matrix sample:")
print(user_item_matrix_filled.iloc[:5, :5])
