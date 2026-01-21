import streamlit as st
import pandas as pd
import plotly.express as px

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
    background: white;
    border-radius: 14px;
    box-shadow: 0 3px 10px rgba(0,0,0,0.10);
    text-align: center;
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
# LOAD DATA
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("master_dataset.csv")
    return df

raw_df = load_data()

# --------------------------------------------------
# SENTIMENT MAPPING
# --------------------------------------------------
raw_df["sentiment"] = raw_df["rating"].apply(
    lambda r: "Positive" if r >= 4 else "Neutral" if r == 3 else "Negative"
)

sentiment_map = {"Positive": 1, "Neutral": 0, "Negative": -1}
raw_df["sentiment_score"] = raw_df["sentiment"].map(sentiment_map)

# --------------------------------------------------
# PRODUCT LEVEL AGGREGATION
# --------------------------------------------------
product_df = raw_df.groupby(["product_title", "domain"]).agg(
    avg_rating=("rating", "mean"),
    total_reviews=("rating", "count"),
    avg_sentiment=("sentiment_score", "mean")
).reset_index()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.title("⚙️ Filters")

domain = st.sidebar.selectbox(
    "Choose Category:",
    ["All", "Books", "Electronics", "Clothing"]
)

search_query = st.sidebar.text_input("🔍 Search Product")

# --------------------------------------------------
# FILTERING LOGIC
# --------------------------------------------------
filtered_df = product_df.copy()

if domain != "All":
    filtered_df = filtered_df[filtered_df["domain"] == domain]

if search_query:
    filtered_df = filtered_df[
        filtered_df["product_title"].str.contains(search_query, case=False, na=False)
    ]

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown("<div class='header'>🛒 Amazon Review Intelligence Dashboard</div>", unsafe_allow_html=True)
st.write("Analysis of Books, Electronics and Clothing • Sentiment • Review Summary • Recommendation Engine")

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
    st.metric("Average Rating", round(product_df["avg_rating"].mean(), 2))
    st.markdown("</div>", unsafe_allow_html=True)

with c3:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.metric("Total Reviews", len(raw_df))
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

        # Product Title
        st.markdown(f"## {row['product_title']}")

        # Metrics
        st.metric("⭐ Average Rating", round(row["avg_rating"], 2))
        st.caption(f"Total Reviews: {row['total_reviews']}")

        # Sentiment Meter
        st.markdown("<div class='section-title'>🎛 Sentiment Meter</div>", unsafe_allow_html=True)
        st.progress((row["avg_sentiment"] + 1) / 2)

        # Sentiment Breakdown
        product_reviews = raw_df[raw_df["product_title"] == row["product_title"]]

        pos = (product_reviews["sentiment"] == "Positive").sum()
        neu = (product_reviews["sentiment"] == "Neutral").sum()
        neg = (product_reviews["sentiment"] == "Negative").sum()

        sentiment_data = pd.DataFrame({
            "Sentiment": ["Positive", "Neutral", "Negative"],
            "Count": [pos, neu, neg]
        })

       # Sentiment Breakdown Section
        st.markdown("<div class='section-title'>📊 Sentiment Breakdown</div>", unsafe_allow_html=True)

        fig_pie = px.pie(sentiment_data, values="Count", names="Sentiment")
        st.plotly_chart(fig_pie, use_container_width=True, key=f"pie_{row['product_title']}")

        st.markdown("</div>", unsafe_allow_html=True)


# --------------------------------------------------
# CHATBOT
# --------------------------------------------------
st.markdown("---")
st.subheader("🤖 Smart Recommendation Assistant")

user_q = st.text_input(
    "Ask something like: ",
    placeholder="Best phone? Best book? Best clothing item?"
)

if user_q:
    # Pick best product in the chosen domain
    if domain == "All":
        best = product_df.sort_values("avg_rating", ascending=False).iloc[0]
    else:
        best = product_df[product_df["domain"] == domain].sort_values("avg_rating", ascending=False).iloc[0]

    st.success(
        f"### Recommended Product\n"
        f"**{best['product_title']}**\n"
        f"- ⭐ Rating: {round(best['avg_rating'], 2)}\n"
        f"- 🛒 Category: {best['domain']}"
    )
