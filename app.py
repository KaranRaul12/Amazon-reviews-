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

.main-header {
    text-align: center;
    color: #1A2A33;
    font-size: 42px;
    font-weight: 700;
    padding-top: 10px;
}

.metric-card {
    padding: 18px;
    background: white;
    border-radius: 14px;
    box-shadow: 0 3px 10px rgba(0,0,0,0.1);
    text-align: center;
}

.product-card {
    background: white;
    padding: 22px;
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
# DATA LOADER (STREAMING FROM MCAULEY LAB)
# --------------------------------------------------
CATEGORY_MAP = {
    "Books": "Books",
    "Electronics": "Electronics",
    "Clothing": "Clothing_Shoes_and_Jewelry"
}

@st.cache_data(show_spinner=True)
def load_amazon_data(domain, samples=400):
    ds = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        name=domain,
        split="full",
        streaming=True
    )
    subset = list(itertools.islice(ds, samples))
    return pd.DataFrame(subset)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.title("⚙️ Controls")

category = st.sidebar.selectbox(
    "Choose Category", 
    ["Books", "Electronics", "Clothing"]
)

search_query = st.sidebar.text_input("🔍 Search Product Title")

# Load data dynamically
raw_df = load_amazon_data(CATEGORY_MAP[category])


# --------------------------------------------------
# DATA PREP
# --------------------------------------------------
raw_df = raw_df.rename(columns={
    "rating": "rating",
    "title": "product_title",
    "review_text": "review_text"
})

raw_df["rating"] = raw_df["rating"].astype(float)

raw_df["sentiment"] = raw_df["rating"].apply(
    lambda r: "Positive" if r >= 4 else "Neutral" if r == 3 else "Negative"
)

sentiment_map = {"Positive": 1, "Neutral": 0, "Negative": -1}
raw_df["sentiment_score"] = raw_df["sentiment"].map(sentiment_map)

product_df = raw_df.groupby("product_title").agg(
    avg_rating=("rating", "mean"),
    review_count=("rating", "count"),
    avg_sentiment=("sentiment_score", "mean")
).reset_index()


# --------------------------------------------------
# FILTER
# --------------------------------------------------
filtered_df = product_df.copy()

if search_query:
    filtered_df = filtered_df[filtered_df["product_title"].str.contains(search_query, case=False, na=False)]


# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown(f"<h1 class='main-header'>🛒 Amazon Review Intelligence ({category})</h1>", unsafe_allow_html=True)
st.markdown("### Real Amazon review analytics powered by McAuley Lab (2023 dataset)")


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
# PRODUCT CARDS
# --------------------------------------------------
st.subheader("📦 Products")

if filtered_df.empty:
    st.warning("No matching products found.")
else:
    for _, row in filtered_df.iterrows():

        st.markdown("<div class='product-card'>", unsafe_allow_html=True)

        # Title
        st.markdown(f"## {row['product_title']}")

        # Rating row
        st.metric("⭐ Average Rating", round(row["avg_rating"], 2))
        st.caption(f"Total Reviews: {row['review_count']}")

        # Sentiment Meter
        st.markdown("<div class='section-title'>🎛 Sentiment Meter</div>", unsafe_allow_html=True)
        st.progress((row["avg_sentiment"] + 1) / 2)

        # Sentiment Charts
        product_reviews = raw_df[raw_df["product_title"] == row["product_title"]]

        pos = (product_reviews["sentiment"] == "Positive").sum()
        neu = (product_reviews["sentiment"] == "Neutral").sum()
        neg = (product_reviews["sentiment"] == "Negative").sum()

        sentiment_df = pd.DataFrame({
            "Sentiment": ["Positive", "Neutral", "Negative"],
            "Count": [pos, neu, neg]
        })

        st.markdown("<div class='section-title'>📊 Sentiment Breakdown</div>", unsafe_allow_html=True)

        fig_pie = px.pie(sentiment_df, values="Count", names="Sentiment")
        st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)


# --------------------------------------------------
# CHATBOT
# --------------------------------------------------
st.markdown("---")
st.subheader("🤖 Smart Recommendation Assistant")

user_q = st.text_input(
    "Ask something like:",
    placeholder="Best phone? Best mindset book? Comfortable shirts?"
)

if user_q:
    best = product_df.sort_values("avg_rating", ascending=False).iloc[0]
    
    st.success(
        f"### Recommended Product\n"
        f"**{best['product_title']}**\n"
        f"- ⭐ Rating: {round(best['avg_rating'], 2)}\n"
        f"- 📨 Category: {category}"
    )
