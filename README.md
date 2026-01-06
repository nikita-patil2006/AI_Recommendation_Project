# AI Enabled Recommendation Engine for an E-commerce Platform

## Project Overview
The goal of this project is to design and develop an AI-driven recommendation engine for an e-commerce platform. The system leverages user behavior and product data to generate personalized product recommendations. By analyzing historical interactions between users and items, the recommendation engine aims to improve user engagement and sales on the platform.


## Milestone 1 – Data Preparation

### Objectives
- Prepare clean, structured datasets suitable for model development.
- Handle missing values, duplicates, and data inconsistencies.
- Merge multiple datasets into a single dataset for convenience.
- Construct a user–item interaction matrix to be used for recommendation algorithms.

### Tasks Completed
- Loaded user, movie, ratings, links, and tags datasets from the MovieLens dataset.
- Removed duplicate rows from all datasets.
- Checked and handled missing values:
  - Filled missing `tmdbId` in `links.csv` with 0.
- Merged datasets into `combined_cleaned.csv` for easy access.
- Built a **user–item interaction matrix** where:
  - Rows represent users (`userId`)
  - Columns represent movies (`movieId`)
  - Values represent ratings
- Saved outputs to the `output/` folder for further model development.

---

## How to Run

1. (Optional) Place the MovieLens CSV files in a `data/` folder in your local copy of the repository.  
2. Run the data preparation script:
   ```bash
   python scripts/data_prep.py
