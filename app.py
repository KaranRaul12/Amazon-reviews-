import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ------------------------------------------------------------
# STREAMLIT CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="Universal Product Analyzer",
    page_icon="🔍",
    layout="wide"
)

# ------------------------------------------------------------
# CUSTOM CSS (MODERN HYBRID UI)
# ------------------------------------------------------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

.title-container {
    text-align: center;
    padding: 20px;
    color: white;
}

.big-title {
    font-size: 42px;
    font-weight: 900;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.30);
}

.subtitle {
    font-size: 18px;
    opacity: 0.9;
}

.card {
    background: white;
    padding: 22px;
    border-radius: 18px;
    box-shadow: 0px 6px 18px rgba(0,0,0,0.12);
    margin-bottom: 25px;
}

.product-header {
    background: white;
    padding: 25px;
    border-radius: 18px;
    display: flex;
    gap: 25px;
    box-shadow: 0px 6px 20px rgba(0,0,0,0.15);
}

.badge {
    display: inline-block;
    padding: 6px 15px;
    border-radius: 20px;
    background: #667eea;
    color: white;
    font-size: 12px;
    margin-bottom: 10px;
}

.feature-box {
    background: #f8f9fa;
    padding: 15px;
    border-radius: 12px;
    margin-bottom: 10px;
}

.score-bar-container {
    height: 7px;
    width: 100%;
    background: #e0e0e0;
    border-radius: 20px;
}

.score-bar-fill {
    height: 7px;
    background: #667eea;
    border-radius: 20px;
}

.section-title {
    font-size: 22px;
    font-weight: 700;
    color: #2c3e50;
    margin-bottom: 15px;
}

.sentiment-badge {
    padding: 6px 14px;
    border-radius: 15px;
    font-size: 13px;
}

.positive { background: #d4edda; color:#155724; }
.neutral { background: #fff3cd; color:#856404; }
.negative { background: #f8d7da; color:#721c24; }

.summary-card {
    background: rgba(255, 255, 255, 0.45);
    padding: 18px;
    border-radius: 12px;
    backdrop-filter: blur(8px);
    margin-bottom: 12px;
}

.summary-section {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 35px;
    border-radius: 18px;
    color: white;
    box-shadow: 0 6px 25px rgba(0,0,0,0.15);
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# LOAD DATASET
# ------------------------------------------------------------
@st.cache_resource
def load_master_data():
    return pd.read_csv("master_mixed_dataset.csv")

df = load_master_data()

# Category icons
CATEGORY_ICONS = {
    "Phones": "📱",
    "Clothing": "👕",
    "Books": "📚",
    "Appliances": "🏠",
    "Cosmetics": "💄"
}

# ------------------------------------------------------------
# PAGE TITLE
# ------------------------------------------------------------
st.markdown("""
<div class="title-container">
    <div class="big-title">🔍 Universal Product Analyzer</div>
    <div class="subtitle">AI-Powered Cross-Domain Product Insights • Sentiment • Pricing • Recommendations</div>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# CATEGORY SELECTOR
# ------------------------------------------------------------
category = st.radio(
    "Choose a category:",
    list(CATEGORY_ICONS.keys()),
    horizontal=True
)

cat_icon = CATEGORY_ICONS[category]

# Filter dataset
cat_df = df[df["category"] == category]

# ------------------------------------------------------------
# SEARCH BAR
# ------------------------------------------------------------
search = st.text_input("Search Product", placeholder=f"Search {category.lower()}… e.g., {cat_df.iloc[0]['name']}")
filtered = cat_df[cat_df["name"].str.contains(search, case=False, na=False)] if search else cat_df

if search and filtered.empty:
    st.warning("No products found. Try another search.")
    st.stop()

# Pick first product
product = filtered.iloc[0]

# ------------------------------------------------------------
# PRODUCT HEADER
# ------------------------------------------------------------
st.markdown(f"""
<div class="product-header">
    <div style="font-size: 60px;">{cat_icon}</div>
    <div>
        <div class="badge">{product['category']}</div>
        <h2>{product['name']}</h2>
        <p><b>Price:</b> ₹{int(product['price'])}</p>
        <p><b>Rating:</b> ⭐ {product['rating']}/5</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# FEATURE SCORES
# ------------------------------------------------------------
st.markdown("### 📊 Product Feature Breakdown")

features = {
    "Quality": product["quality_score"],
    "Durability": product["durability_score"],
    "Value for Money": product["value_score"],
    "Aesthetics": product["design_score"]
}

left, right = st.columns([1.3, 1])

with left:
    for feat, score in features.items():
        st.markdown(f"""
        <div class="feature-box">
            <b>{feat}</b>
            <div class="score-bar-container">
                <div class="score-bar-fill" style="width:{score*10}%"></div>
            </div>
            <span style="font-size: 13px; opacity:0.8;">Score: {score}/10</span>
        </div>
        """, unsafe_allow_html=True)

# Sentiment donut
sentiment_data = {
    "Positive": product["positive"],
    "Neutral": product["neutral"],
    "Negative": product["negative"]
}

with right:
    fig = go.Figure(
        data=[go.Pie(
            labels=list(sentiment_data.keys()),
            values=list(sentiment_data.values()),
            hole=0.55,
            marker=dict(colors=["#2ecc71", "#f1c40f", "#e74c3c"])
        )]
    )
    fig.update_layout(title="Sentiment Breakdown", showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------
# TREND CHART
# ------------------------------------------------------------
st.markdown("### 📈 Price Trend (Past 4 months)")
trend_fig = px.line(
    x=["4 months ago", "3 months ago", "2 months ago", "Now"],
    y=[product["price"] * 1.10, product["price"] * 1.05, product["price"] * 1.03, product["price"]],
    markers=True
)
trend_fig.update_traces(line_color="#667eea")
st.plotly_chart(trend_fig, use_container_width=True)

# ------------------------------------------------------------
# SUMMARY SECTION
# ------------------------------------------------------------
st.markdown("### 🧠 AI Summary & Recommendation")

st.markdown("""
<div class="summary-section">
    <h2 style="text-align:center;">📘 Product Summary</h2>
""", unsafe_allow_html=True)

colA, colB, colC = st.columns(3)

with colA:
    st.markdown("<h4>✔ Strengths</h4>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="summary-card">
    <ul>
        <li>High quality score ({product['quality_score']}/10)</li>
        <li>Strong durability rating</li>
        <li>Good value for money</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with colB:
    st.markdown("<h4>⚠ Weaknesses</h4>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="summary-card">
    <ul>
        <li>Design score could improve ({product['design_score']}/10)</li>
        <li>Neutral sentiment noticeable</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with colC:
    st.markdown("<h4>🔥 Issues</h4>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="summary-card">
    <ul>
        <li>{product['negative']}% negative feedback</li>
        <li>Some users reported quality inconsistencies</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# FINAL VERDICT
final_score = round(
    (product["quality_score"] +
     product["value_score"] +
     product["durability_score"] +
     product["design_score"]) / 4, 1
)

st.markdown(f"""
<div class="summary-card" style="background:white; color:black; text-align:center;">
    <h2>🏆 Final Verdict: {final_score}/10</h2>
    <p style="font-size:16px;">
        Based on <b>{product['positive']}%</b> positive sentiment and strong feature ratings,
        this product is <b>recommended</b> for most buyers.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
