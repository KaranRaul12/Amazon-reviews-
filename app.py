import streamlit as st
import pandas as pd
import plotly.express as px

# --------------------------------------------------
# STREAMLIT PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Amazon Review Intelligence",
    page_icon="🛒",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS (Modern clean UI)
# --------------------------------------------------
st.markdown("""
<style>
body { background-color: #f6f7fb; }

h1 {
    text-align: center;
    color: #1f2c38;
}

.card {
    background: white;
    padding: 22px;
    border-radius: 14px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.07);
    margin-bottom: 22px;
}

.section {
    font-size: 18px;
    font-weight: 600;
    margin-top: 15px;
    color: #2c3e50;
}

.section-title {
    font-size: 20px;
    font-weight: 700;
    margin-top: 15px;
    margin-bottom: 10px;
    color: #273746;
}

.small {
    color: #6c757d;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("master_mixed_dataset.csv")

df = load_data()

# --------------------------------------------------
# SENTIMENT LABELS
# --------------------------------------------------
def label_sentiment(r):
    if r >= 4:
        return "Positive"
    elif r == 3:
        return "Neutral"
    else:
        return "Negative"

df["sentiment"] = df["rating"].apply(label_sentiment)
sentiment_map = {"Positive": 1, "Neutral": 0, "Negative": -1}
df["sentiment_score"] = df["sentiment"].map(sentiment_map)

# --------------------------------------------------
# AGGREGATE TO PRODUCT LEVEL
# --------------------------------------------------
product_df = df.groupby(["product_title", "domain"]).agg(
    avg_rating=("rating", "mean"),
    review_count=("rating", "count"),
    avg_sentiment_score=("sentiment_score", "mean")
).reset_index()

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown("<h1>🛒 Amazon Review Intelligence</h1>", unsafe_allow_html=True)
st.markdown("<p class='small' style='text-align:center;'>Cross-Domain Sentiment • Review Summaries • Buying Guide • AI Assistant</p>", unsafe_allow_html=True)
st.markdown("---")

# --------------------------------------------------
# SEARCH + CATEGORY FILTER
# --------------------------------------------------
c1, c2 = st.columns([3, 1])

with c1:
    query = st.text_input(
        "🔍 Search product",
        placeholder="Search phones, books, fashion, etc."
    )

with c2:
    domain_filter = st.selectbox("Category", ["All", "Books", "Electronics", "Clothing"])

filtered = product_df.copy()

if domain_filter != "All":
    filtered = filtered[filtered["domain"] == domain_filter]

if query:
    filtered = filtered[
        filtered["product_title"].str.contains(query, case=False, na=False)
    ]

# --------------------------------------------------
# SEARCH RESULTS
# --------------------------------------------------
st.subheader("🔎 Search Results")

if filtered.empty:
    st.warning("No matching products found.")
else:
    for idx, row in filtered.iterrows():

        st.markdown("<div class='card'>", unsafe_allow_html=True)

        # Title + Rating
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"### {row['product_title']}")
            st.caption(f"**Category:** {row['domain']}")
        with col2:
            st.metric("⭐ Avg Rating", round(row["avg_rating"], 2))

        st.caption(f"Total Reviews: {row['review_count']}")

        # Summary
        st.markdown("<div class='section-title'>📘 Review Summary</div>", unsafe_allow_html=True)
        st.write("Users shared a variety of opinions indicating balanced sentiment across key aspects of the product.")

        # Sentiment Meter
        st.markdown("<div class='section-title'>🎛 Sentiment Meter</div>", unsafe_allow_html=True)
        sentiment_val = (row["avg_sentiment_score"] + 1) / 2
        st.progress(sentiment_val)

        # Buying Guide
        st.markdown("<div class='section-title'>🎯 Buying Recommendation</div>", unsafe_allow_html=True)
        if row["avg_rating"] >= 4:
            st.success("Must Buy ✔")
        elif row["avg_rating"] <= 2.5:
            st.error("Avoid ❌")
        else:
            st.warning("Think Again ⚠️")

        # SENTIMENT DATA PER PRODUCT
        product_reviews = df[df["product_title"] == row["product_title"]]

        pos = (product_reviews["sentiment"] == "Positive").sum()
        neu = (product_reviews["sentiment"] == "Neutral").sum()
        neg = (product_reviews["sentiment"] == "Negative").sum()

        sentiment_df = pd.DataFrame({
            "Sentiment": ["Positive", "Neutral", "Negative"],
            "Count": [pos, neu, neg]
        })

        # Pie Chart
        st.markdown("<div class='section-title'>📊 Sentiment Breakdown</div>", unsafe_allow_html=True)
        fig_pie = px.pie(
            sentiment_df,
            names="Sentiment",
            values="Count",
            color="Sentiment",
            color_discrete_map={
                "Positive": "#2ecc71",
                "Neutral": "#f1c40f",
                "Negative": "#e74c3c"
            }
        )
        st.plotly_chart(fig_pie, use_container_width=True, key=f"pie_{idx}")

        # Bar Chart
        fig_bar = px.bar(
            sentiment_df,
            x="Sentiment",
            y="Count",
            text="Count",
            color="Sentiment",
            color_discrete_map={
                "Positive": "#2ecc71",
                "Neutral": "#f1c40f",
                "Negative": "#e74c3c"
            }
        )
        st.plotly_chart(fig_bar, use_container_width=True, key=f"bar_{idx}")

        st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------
# SMART ASSISTANT
# --------------------------------------------------
st.markdown("---")
st.subheader("🤖 Smart Recommendation Assistant")

query = st.text_input(
    "Ask anything…",
    placeholder="Suggest me a good phone / fiction book / budget clothing"
)

if query:
    q = query.lower()

    if "phone" in q or "mobile" in q:
        subset = product_df[product_df["domain"] == "Electronics"]
    elif "book" in q or "novel" in q:
        subset = product_df[product_df["domain"] == "Books"]
    elif "cloth" in q or "shirt" in q or "dress" in q or "fashion" in q:
        subset = product_df[product_df["domain"] == "Clothing"]
    else:
        subset = product_df

    best = subset.sort_values("avg_rating", ascending=False).iloc[0]

    st.success(
        f"### ✅ Recommended Product\n\n"
        f"**{best['product_title']}**\n"
        f"- ⭐ Rating: {round(best['avg_rating'], 2)}\n"
        f"- 🛒 Category: {best['domain']}"
    )
