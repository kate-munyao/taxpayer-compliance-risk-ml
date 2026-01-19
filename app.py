import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go

import os
from compliance_fraud_pipeline import main as generate_data


# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="Taxpayer Compliance Risk Dashboard",
    page_icon="🚨",
    layout="wide"
)

DB_NAME = "taxpayer_compliance.db"


if not os.path.exists(DB_NAME):
    st.warning("⚠️ Database not found. Generating synthetic data...")
    generate_data()


# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql("SELECT * FROM taxpayers", conn)
    conn.close()
    return df

@st.cache_data
def load_feature_importance():
    try:
        return pd.read_excel(
            "synthetic_taxpayer_compliance_data.xlsx",
            sheet_name='Feature_Importance'
        )
    except:
        return None

df = load_data()
feat_imp = load_feature_importance()

# -----------------------------
# SIDEBAR FILTERS
# -----------------------------
st.sidebar.title("🔧 Filters")

sector_filter = st.sidebar.multiselect(
    "Select Sector",
    options=sorted(df["sector"].unique()),
    default=df["sector"].unique()
)

risk_filter = st.sidebar.multiselect(
    "Select Risk Band",
    options=["Low", "Medium", "High"],
    default=["Low", "Medium", "High"]
)

show_anomalies_only = st.sidebar.checkbox("Show Anomalies Only", value=False)

st.sidebar.markdown("---")

# Search by Taxpayer ID
st.sidebar.subheader("🔎 Search Taxpayer")
search_id = st.sidebar.number_input(
    "Enter Taxpayer ID",
    min_value=1,
    max_value=int(df['taxpayer_id'].max()),
    value=1,
    step=1
)

search_button = st.sidebar.button("🔍 Search")

# Apply filters
filtered_df = df[
    (df["sector"].isin(sector_filter)) &
    (df["risk_band"].isin(risk_filter))
]

if show_anomalies_only:
    filtered_df = filtered_df[filtered_df["anomaly_flag"] == 1]

st.sidebar.markdown("---")
st.sidebar.metric("📊 Filtered Records", f"{len(filtered_df):,}")

# -----------------------------
# HEADER
# -----------------------------
st.title("🚨 Taxpayer Compliance Risk & Fraud Detection Dashboard")
st.caption("ML-powered anomaly detection using Isolation Forest & Risk Scoring")
st.markdown("---")

# -----------------------------
# KEY METRICS
# -----------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Taxpayers",
        f"{len(filtered_df):,}",
        delta=f"{len(filtered_df)/len(df)*100:.1f}% of total"
    )

with col2:
    high_risk_count = (filtered_df["risk_band"] == "High").sum()
    st.metric(
        "High Risk",
        high_risk_count,
        delta=f"{high_risk_count/len(filtered_df)*100:.1f}%" if len(filtered_df) > 0 else "0%"
    )

with col3:
    anomaly_count = filtered_df["anomaly_flag"].sum()
    st.metric(
        "Anomalies Detected",
        anomaly_count,
        delta=f"{anomaly_count/len(filtered_df)*100:.1f}%" if len(filtered_df) > 0 else "0%"
    )

with col4:
    overdue_count = filtered_df["overdue_flag"].sum()
    st.metric(
        "Overdue Filers",
        overdue_count,
        delta=f"{overdue_count/len(filtered_df)*100:.1f}%" if len(filtered_df) > 0 else "0%"
    )

st.markdown("---")

# -----------------------------
# SEARCH RESULT (if searched)
# -----------------------------
if search_button:
    searched_taxpayer = df[df['taxpayer_id'] == search_id]
    
    if not searched_taxpayer.empty:
        st.subheader(f"🔍 Search Result: Taxpayer #{search_id}")
        
        # Display key info in columns
        col1, col2, col3, col4, col5 = st.columns(5)
        
        taxpayer = searched_taxpayer.iloc[0]
        
        with col1:
            st.metric("Sector", taxpayer['sector'])
        with col2:
            risk_color = {
                "High": "🔴",
                "Medium": "🟡",
                "Low": "🟢"
            }
            st.metric("Risk Band", 
                     f"{risk_color.get(taxpayer['risk_band'], '')} {taxpayer['risk_band']}")
        with col3:
            st.metric("Risk Score", f"{taxpayer['risk_score']:.2f}")
        with col4:
            anomaly_status = "⚠️ Yes" if taxpayer['anomaly_flag'] == 1 else "✅ No"
            st.metric("Anomaly", anomaly_status)
        with col5:
            overdue_status = "🚨 Yes" if taxpayer['overdue_flag'] == 1 else "✅ No"
            st.metric("Overdue", overdue_status)
        
        # Detailed information
        st.markdown("**Detailed Information**")
        detail_cols = st.columns(3)
        
        with detail_cols[0]:
            st.write(f"**Average Payment:** ${taxpayer['avg_payment']:,.2f}")
            st.write(f"**Payment Std Dev:** ${taxpayer['payment_std']:,.2f}")
            st.write(f"**Payment Consistency:** {taxpayer['payment_consistency']:.2f}")
        
        with detail_cols[1]:
            st.write(f"**Late Filing Rate:** {taxpayer['late_filing_rate']:.2%}")
            st.write(f"**Missed Payments:** {taxpayer['missed_payments']:.0f}")
            st.write(f"**Filing Frequency:** {taxpayer['filing_frequency']} per year")
        
        with detail_cols[2]:
            st.write(f"**Days Since Filing:** {taxpayer['days_since_filing']:.0f}")
            st.write(f"**Revenue Volatility:** {taxpayer['revenue_volatility']:.2f}")
            st.write(f"**Risk Pressure:** {taxpayer['risk_pressure']:.2f}")
        
        st.markdown("---")
    else:
        st.warning(f"⚠️ No taxpayer found with ID: {search_id}")
        st.markdown("---")

# -----------------------------
# ROW 1: RISK DISTRIBUTION & ANOMALY MAP
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Risk Band Distribution")
    
    risk_dist = filtered_df["risk_band"].value_counts().reset_index()
    risk_dist.columns = ["Risk Band", "Count"]
    
    # Color mapping for consistent theming
    color_map = {
        "High": "#FF4B4B",
        "Medium": "#FFA500",
        "Low": "#00CC66"
    }
    
    fig1 = px.pie(
        risk_dist,
        values="Count",
        names="Risk Band",
        title="Risk Level Breakdown",
        color="Risk Band",
        color_discrete_map=color_map,
        hole=0.3  # Donut chart
    )
    fig1.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("🔍 Anomaly Detection Map")
    
    fig_scatter = px.scatter(
        filtered_df,
        x='log_avg_payment',
        y='risk_pressure',
        color='anomaly_flag',
        size='risk_score',
        hover_data=['taxpayer_id', 'sector', 'risk_band'],
        title='Payment Patterns vs Risk Pressure',
        color_discrete_map={0: '#1f77b4', 1: '#ff0000'},
        labels={
            'anomaly_flag': 'Anomaly',
            'log_avg_payment': 'Log Avg Payment',
            'risk_pressure': 'Risk Pressure'
        }
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# -----------------------------
# ROW 2: SECTOR ANALYSIS
# -----------------------------
st.markdown("---")
st.subheader("🏭 Sector Risk Analysis")

sector_summary = (
    filtered_df
    .groupby("sector")
    .agg(
        total_taxpayers=("taxpayer_id", "count"),
        avg_risk_score=("risk_score", "mean"),
        anomalies=("anomaly_flag", "sum"),
        overdue=("overdue_flag", "sum")
    )
    .reset_index()
    .sort_values("avg_risk_score", ascending=False)
)

col1, col2 = st.columns([2, 1])

with col1:
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=sector_summary["sector"],
        y=sector_summary["avg_risk_score"],
        marker_color='indianred',
        text=sector_summary["avg_risk_score"].round(2),
        textposition='auto',
    ))
    fig2.update_layout(
        title="Average Risk Score by Sector",
        xaxis_title="Sector",
        yaxis_title="Average Risk Score",
        height=400
    )
    st.plotly_chart(fig2, use_container_width=True)

with col2:
    st.markdown("**Sector Metrics**")
    st.dataframe(
        sector_summary.style.format({
            'avg_risk_score': '{:.2f}',
            'total_taxpayers': '{:,}',
            'anomalies': '{:,}',
            'overdue': '{:,}'
        }),
        hide_index=True,
        use_container_width=True
    )

# -----------------------------
# ROW 3: RISK SCORE DISTRIBUTION
# -----------------------------
st.markdown("---")
st.subheader("📊 Risk Score Distribution")

col1, col2 = st.columns([2, 1])

with col1:
    fig_hist = px.histogram(
        filtered_df,
        x='risk_score',
        nbins=30,
        title='Risk Score Distribution',
        color_discrete_sequence=['steelblue'],
        labels={'risk_score': 'Risk Score', 'count': 'Number of Taxpayers'}
    )
    fig_hist.update_layout(
        showlegend=False,
        height=400,
        bargap=0.1
    )
    # Add mean line
    mean_risk = filtered_df['risk_score'].mean()
    fig_hist.add_vline(
        x=mean_risk,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {mean_risk:.2f}",
        annotation_position="top"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with col2:
    st.markdown("**Distribution Statistics**")
    
    stats_df = pd.DataFrame({
        'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
        'Value': [
            f"{filtered_df['risk_score'].mean():.2f}",
            f"{filtered_df['risk_score'].median():.2f}",
            f"{filtered_df['risk_score'].std():.2f}",
            f"{filtered_df['risk_score'].min():.2f}",
            f"{filtered_df['risk_score'].max():.2f}"
        ]
    })
    
    st.dataframe(stats_df, hide_index=True, use_container_width=True)
    
    st.info("""
    **Interpretation:**
    - Most taxpayers cluster around the mean
    - Long tail indicates outliers
    - Red line shows average risk
    """)

# -----------------------------
# ROW 4: FEATURE IMPORTANCE (if available)
# -----------------------------
if feat_imp is not None:
    st.markdown("---")
    st.subheader("📈 Model Explainability: Feature Importance")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig_feat = px.bar(
            feat_imp,
            y='feature',
            x='importance',
            orientation='h',
            title='Key Drivers of High-Risk Classification',
            color='importance',
            color_continuous_scale='Reds'
        )
        fig_feat.update_layout(
            showlegend=False,
            height=400,
            xaxis_title="Importance Score",
            yaxis_title="Feature"
        )
        st.plotly_chart(fig_feat, use_container_width=True)
    
    with col2:
        st.markdown("**Interpretation**")
        st.info("""
        **Feature Importance** shows which factors most strongly predict high-risk taxpayers:
        
        - Higher scores = stronger predictive power
        - Top features drive risk classification
        - Used by Random Forest model
        """)

# -----------------------------
# ROW 5: TOP HIGH-RISK TAXPAYERS
# -----------------------------
st.markdown("---")
st.subheader("🔴 Top 10 High-Risk Taxpayers")

high_risk_df = filtered_df.nlargest(10, 'risk_score')[[
    'taxpayer_id', 'sector', 'risk_score', 'risk_band',
    'anomaly_flag', 'overdue_flag', 'missed_payments', 'days_since_filing'
]]

# Style the dataframe
def highlight_anomalies(row):
    if row['anomaly_flag'] == 1:
      return [
    'background-color: #FFCDD2; color: black; font-weight: bold'
] * len(row)

    return [''] * len(row)

styled_high_risk = high_risk_df.style.apply(highlight_anomalies, axis=1).format({
    'risk_score': '{:.2f}',
    'days_since_filing': '{:.0f}'
})

st.dataframe(styled_high_risk, use_container_width=True, hide_index=True)
st.caption("💡 Rows highlighted in pink indicate flagged anomalies")

# -----------------------------
# ROW 6: DETAILED DATA VIEW
# -----------------------------
st.markdown("---")
st.subheader("📋 Detailed Taxpayer Data")

display_cols = [
    'taxpayer_id', 'sector', 'risk_score', 'risk_band',
    'anomaly_flag', 'overdue_flag', 'missed_payments',
    'late_filing_rate', 'days_since_filing'
]

st.dataframe(
    filtered_df[display_cols].sort_values('risk_score', ascending=False).style.format({
        'risk_score': '{:.2f}',
        'late_filing_rate': '{:.2%}',
        'days_since_filing': '{:.0f}'
    }),
    use_container_width=True,
    hide_index=True,
    height=400
)

# -----------------------------
# DOWNLOAD SECTION
# -----------------------------
st.markdown("---")
st.subheader("⬇️ Export Data")

col1, col2 = st.columns(2)

with col1:
    csv = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Filtered Data (CSV)",
        data=csv,
        file_name="filtered_taxpayer_risk_data.csv",
        mime="text/csv"
    )

with col2:
    high_risk_csv = high_risk_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="🚨 Download Top 10 High-Risk (CSV)",
        data=high_risk_csv,
        file_name="top_10_high_risk.csv",
        mime="text/csv"
    )

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("Built with Python, Scikit-learn, Streamlit & Plotly | Data: Synthetic")
st.caption("ML Models: Isolation Forest (Anomaly Detection) + Random Forest (Feature Importance)")