import os
import random
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="AI Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# ======================================================
# SESSION STATE
# ======================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user_email" not in st.session_state:
    st.session_state.user_email = None

if "user_id" not in st.session_state:
    st.session_state.user_id = None

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

# ======================================================
# PATHS
# ======================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOVIES_PATH = os.path.join(BASE_DIR, "data", "movies.csv")
RATINGS_PATH = os.path.join(BASE_DIR, "data", "ratings.csv")

movies_df = pd.read_csv(MOVIES_PATH)
ratings_df = pd.read_csv(RATINGS_PATH)

# ======================================================
# DATA PREPARATION
# ======================================================
user_item_matrix = ratings_df.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
).fillna(0)

similarity = cosine_similarity(user_item_matrix)
similarity_df = pd.DataFrame(
    similarity,
    index=user_item_matrix.index,
    columns=user_item_matrix.index
)

movie_map = dict(zip(movies_df.movieId, movies_df.title))
avg_ratings = ratings_df.groupby("movieId")["rating"].mean().round(2).to_dict()

# ======================================================
# HELPER FUNCTIONS
# ======================================================
def get_user_top_movies(user_id, top_k=3):
    """Movies the user liked the most"""
    top_movies = (
        ratings_df[ratings_df.userId == user_id]
        .sort_values("rating", ascending=False)
        .head(top_k)["movieId"]
        .tolist()
    )
    return [movie_map.get(mid, "a movie you liked") for mid in top_movies]


def recommend_movies(user_id, top_n=5):
    if user_id not in user_item_matrix.index:
        return []

    similar_users = similarity_df[user_id].sort_values(ascending=False)[1:6]
    scores = {}

    for sim_user, sim_score in similar_users.items():
        for movie_id, rating in user_item_matrix.loc[sim_user].items():
            if user_item_matrix.loc[user_id, movie_id] == 0 and rating > 0:
                scores[movie_id] = scores.get(movie_id, 0) + sim_score * rating

    ranked_movies = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

    liked_movies = get_user_top_movies(user_id)

    recommendations = []
    for movie_id, score in ranked_movies:
        if liked_movies:
            reason = f"Recommended because you liked **{random.choice(liked_movies)}**"
        else:
            reason = "Popular among users with similar taste"

        recommendations.append({
            "title": movie_map.get(movie_id, "Unknown"),
            "avg_rating": avg_ratings.get(movie_id, "N/A"),
            "score": round(score, 2),
            "reason": reason
        })

    return recommendations

# ======================================================
# LOGIN PAGE
# ======================================================
if not st.session_state.logged_in:
    st.title("🔐 Login to MovieRec AI")

    col1, col2 = st.columns(2)

    with col1:
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")

    if st.button("Login"):
        if email and password:
            st.session_state.logged_in = True
            st.session_state.user_email = email
            st.session_state.user_id = random.choice(user_item_matrix.index.tolist())
            st.rerun()
        else:
            st.error("Please enter both email and password")

    st.caption("⚠ Demo login only — no real authentication")
    st.stop()

# ======================================================
# SIDEBAR NAVBAR
# ======================================================
st.sidebar.title("🎬 MovieRec AI")
st.sidebar.markdown(f"👤 **{st.session_state.user_email}**")
st.sidebar.divider()

if st.sidebar.button("📊 Dashboard"):
    st.session_state.page = "Dashboard"

if st.sidebar.button("🎬 Recommendations"):
    st.session_state.page = "Recommendations"

if st.sidebar.button("ℹ️ About System"):
    st.session_state.page = "About"

st.sidebar.divider()

if st.sidebar.button("🚪 Logout"):
    st.session_state.logged_in = False
    st.session_state.user_email = None
    st.session_state.user_id = None
    st.rerun()

# ======================================================
# DASHBOARD PAGE
# ======================================================
if st.session_state.page == "Dashboard":
    st.title("📊 Dashboard")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Users", ratings_df.userId.nunique())
    col2.metric("Total Movies", movies_df.movieId.nunique())
    col3.metric("Total Ratings", len(ratings_df))

    st.markdown("""
    ### 🎯 System Overview
    - Uses **Collaborative Filtering**
    - Learns user preferences from ratings
    - Finds similar users using cosine similarity
    - Provides explainable recommendations
    """)

# ======================================================
# RECOMMENDATIONS PAGE
# ======================================================
elif st.session_state.page == "Recommendations":
    st.title("🎬 Personalized Movie Recommendations")

    top_n = st.slider("Number of recommendations", 3, 10, 5)

    if st.button("🎯 Get Recommendations"):
        recommendations = recommend_movies(st.session_state.user_id, top_n)

        cols = st.columns(3)
        for i, rec in enumerate(recommendations):
            with cols[i % 3]:
                st.markdown(
                    f"""
                    <div style="
                        background:#111827;
                        padding:18px;
                        border-radius:16px;
                        color:white;
                        margin-bottom:20px;
                        box-shadow:0 4px 12px rgba(0,0,0,0.3);
                    ">
                        <h4>🎬 {rec['title']}</h4>
                        <p>⭐ Avg Rating: {rec['avg_rating']}</p>
                        <p>🎯 Score: {rec['score']}</p>
                        <p>🧠 <i>{rec['reason']}</i></p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

# ======================================================
# ABOUT PAGE
# ======================================================
elif st.session_state.page == "About":
    st.title("ℹ️ About This System")

    st.markdown("""
    ### 🧠 Algorithm
    - User-based Collaborative Filtering
    - Cosine similarity for user matching

    ### ✅ Key Features
    - Login & session handling
    - Explainable AI recommendations
    - Cold-start tolerant design
    - Modular dashboard-based UI

    ### 🚀 Use Cases
    - OTT platforms
    - Personalized content systems
    - Recommendation research demos
    """)

# ======================================================
# FOOTER
# ======================================================
st.markdown("---")
st.markdown(
    "<center>AI Movie Recommendation System • Streamlit Demo</center>",
    unsafe_allow_html=True
)
