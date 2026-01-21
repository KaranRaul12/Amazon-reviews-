import streamlit as st
import pandas as pd
import plotly.express as px
from datasets import load_dataset
import itertools

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Amazon Review Intelligence",
    page_icon="🛒",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------
st.markdown("""
<style>

body { background-color: #F4F6F8; }

.header {
    text-align: center;
    font-size: 38px;
    font-weight: 700;
    color: #1A2A33;
    padding-top: 10px;
    padding-bottom: 15px;
}

.metric-card {
    padding: 18px;
    background: white !important;
    border-radius: 14px !important;
    box-shadow: 0 3px 10px rgba(0,0,0,0.10) !important;
    text-align: center !important;
}

.product-card {
    background: white;
    padding: 25px;
    border-radius: 14px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.08);
    margin-bottom: 25px;
}

.section-title {
    font-size: 20px;
    font-weight: 600;
    color: #2C3E50;
    margin-top: 10px;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# CATEGORY MAPPING
# --------------------------------------------------
CATEGORY_MAP = {
    "Books": "Books",
    "Electronics": "Electronics",
    "Clothing": "Clothing_Shoes_and_Jewelry"
}

# --------------------------------------------------
# STREAMING DATA LOADER
# --------------------------------------------------
@st.cache_data(show_spinner=True)
def load_amazon_sample(category_name, n=500):
    ds = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023-parquet",
        name=category_name,
        split="train",
        streaming=True
    )
    sample = list(itertools.islice(ds, n))
    return pd.DataFrame(sample)

# --------------------------------------------------
# SIDEBAR CONTROLS
# --------------------------------------------------
st.sidebar.title("⚙️ Controls")

category = st.sidebar.selectbox(
    "Choose Category",
    ["Books", "Electronics", "Clothing"]
)

search_query = st.sidebar.text_input(
    "🔍 Search Product Title"
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
raw_df = load_amazon_sample(CATEGORY_MAP[category], n=500)

# --------------------------------------------------
# CLEAN / NORMALIZE COLUMNS
# --------------------------------------------------
if "title" in raw_df.columns:
    raw_df.rename(columns={"title": "product_title"}, inplace=True)

if "reviewText" in raw_df.columns:
    raw_df.rename(columns={"reviewText": "review_text"}, inplace=True)

raw_df["rating"] = raw_df["rating"].astype(float)

raw_df["product_title"] = raw_df["product_title"].fillna("Unknown Product")

# --------------------------------------------------
# SENTIMENT ASSIGNMENT
# --------------------------------------------------
raw_df["sentiment"] = raw_df["rating"].apply(
    lambda r: "Positive" if r >= 4 else "Neutral" if r == 3 else "Negative"
)

sentiment_map = {"Positive": 1, "Neutral": 0, "Negative": -1}
raw_df["sentiment_score"] = raw_df["sentiment"].map(sentiment_map)

# --------------------------------------------------
# AGGREGATED PRODUCT METRICS
# --------------------------------------------------
product_df = raw_df.groupby("product_title").agg(
    avg_rating=("rating", "mean"),
    total_reviews=("rating", "count"),
    avg_sentiment=("sentiment_score", "mean")
).reset_index()

# --------------------------------------------------
# FILTER BY SEARCH
# --------------------------------------------------
filtered_df = product_df.copy()

if search_query:
    filtered_df = filtered_df[
        filtered_df["product_title"].str.contains(search_query, case=False, na=False)
    ]

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown(f"<div class='header'>🛒 Amazon Review Intelligence — {category}</div>", unsafe_allow_html=True)
st.write("Real Amazon reviews analysis using the McAuley Lab 2023 Parquet Dataset")

# --------------------------------------------------
# KPI CARDS
# --------------------------------------------------
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.metric("Total Products", len(product_df))
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.metric("Avg Rating", round(product_df["avg_rating"].mean(), 2))
    st.markdown("</div>", unsafe_allow_html=True)

with c3:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.metric("Total Reviews Sampled", len(raw_df))
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# --------------------------------------------------
# PRODUCT DISPLAY
# --------------------------------------------------
st.subheader("📦 Products")

if filtered_df.empty:
    st.warning("No matching products found.")
else:
    for _, row in filtered_df.iterrows():

        st.markdown("<div class='product-card'>", unsafe_allow_html=True)

        # Product Title
        st.markdown(f"## {row['product_title']}")

        # Ratings
        st.metric("⭐ Average Rating", round(row["avg_rating"], 2))
        st.caption(f"Total Reviews: {row['total_reviews']}")

        # Sentiment Meter
        st.markdown("<div class='section-title'>🎛 Sentiment Meter</div>", unsafe_allow_html=True)
        st.progress((row["avg_sentiment"] + 1) / 2)

        # Sentiment Breakdown
        pr = raw_df[raw_df["product_title"] == row["product_title"]]
        pos = (pr["sentiment"] == "Positive").sum()
        neu = (pr["sentiment"] == "Neutral").sum()
        neg = (pr["sentiment"] == "Negative").sum()

        sentiment_data = pd.DataFrame({
            "Sentiment": ["Positive", "Neutral", "Negative"],
            "Count": [pos, neu, neg]
        })

        st.markdown("<div class='section-title'>📊 Sentiment Breakdown</div>", unsafe_allow_html=True)

        fig = px.pie(sentiment_data, values="Count", names="Sentiment")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------
# SMART CHATBOT
# --------------------------------------------------
st.markdown("---")
st.subheader("🤖 Smart Recommendation Assistant")

user_q = st.text_input(
    "Ask something like:",
    placeholder="Best phone? Best fiction book? Comfortable shirts?"
)

if user_q:
    top_item = product_df.sort_values("avg_rating", ascending=False).iloc[0]
    st.success(
        f"### Recommended Product\n"
        f"**{top_item['product_title']}**\n"
        f"- ⭐ Rating: {round(top_item['avg_rating'], 2)}\n"
        f"- 📦 Category: {category}"
    )
