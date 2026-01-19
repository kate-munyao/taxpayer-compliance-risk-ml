import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
import sqlite3
import matplotlib.pyplot as plt
import logging

# -----------------------------
# LOGGING SETUP
# -----------------------------
logging.basicConfig(
    filename='compliance_pipeline.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# -----------------------------
# 1. CONFIGURATION
# -----------------------------
np.random.seed(42)
N_TAXPAYERS = 1500
OUTPUT_EXCEL = "synthetic_taxpayer_compliance_data.xlsx"
DB_NAME = "taxpayer_compliance.db"

logging.info("=" * 50)
logging.info("Starting Taxpayer Compliance Pipeline")
logging.info(f"Generating {N_TAXPAYERS} synthetic records")

# -----------------------------
# 2. SYNTHETIC DATA GENERATION
# -----------------------------
sectors = [
    "Retail",
    "Manufacturing",
    "Transport",
    "Hospitality",
    "Professional"
]

sector_risk_map = {
    "Retail": 1.2,
    "Manufacturing": 0.9,
    "Transport": 1.1,
    "Hospitality": 1.3,
    "Professional": 0.8
}

df = pd.DataFrame({
    "taxpayer_id": range(1, N_TAXPAYERS + 1),
    "sector": np.random.choice(sectors, N_TAXPAYERS),
    "avg_payment": np.random.lognormal(mean=10, sigma=0.6, size=N_TAXPAYERS),
    "payment_std": np.random.lognormal(mean=9, sigma=0.7, size=N_TAXPAYERS),
    "late_filing_rate": np.random.beta(2, 6, size=N_TAXPAYERS),
    "missed_payments": np.random.poisson(1.5, size=N_TAXPAYERS),
    "filing_frequency": np.random.choice([4, 12], size=N_TAXPAYERS)
})

# Add temporal data
df['last_filing_date'] = pd.to_datetime('2024-01-01') + pd.to_timedelta(
    np.random.randint(0, 365, N_TAXPAYERS), unit='D'
)
df['days_since_filing'] = (pd.Timestamp.now() - df['last_filing_date']).dt.days
df['overdue_flag'] = (df['days_since_filing'] > 120).astype(int)

df["sector_risk"] = df["sector"].map(sector_risk_map)
df["revenue_volatility"] = df["payment_std"] / (df["avg_payment"] + 1)

logging.info("Synthetic data generation complete")

# -----------------------------
# 3. FEATURE ENGINEERING
# -----------------------------
df["payment_consistency"] = df["avg_payment"] / (df["payment_std"] + 1)
df["non_compliance_score"] = (
    (df["late_filing_rate"] * 0.4) +
    ((df["missed_payments"] / 12) * 0.3) +
    (df["revenue_volatility"] * 0.3)
)
df["risk_pressure"] = df["non_compliance_score"] * df["sector_risk"]
df["log_avg_payment"] = np.log1p(df["avg_payment"])

logging.info("Feature engineering complete")

# -----------------------------
# 4. ANOMALY / FRAUD DETECTION
# -----------------------------
features_for_anomaly = [
    "log_avg_payment",
    "payment_consistency",
    "late_filing_rate",
    "missed_payments",
    "risk_pressure"
]

X = df[features_for_anomaly]

iso_model = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42
)

df["anomaly_flag"] = iso_model.fit_predict(X)
df["anomaly_score"] = iso_model.decision_function(X)
df["anomaly_flag"] = df["anomaly_flag"].map({1: 0, -1: 1})

anomaly_count = df["anomaly_flag"].sum()
logging.info(f"Anomaly detection complete: {anomaly_count} anomalies detected")

# -----------------------------
# 5. COMPLIANCE RISK BANDING
# -----------------------------
df["risk_score"] = (df["risk_pressure"] * 100).clip(0, 100)

def assign_risk_band(score):
    if score < 30:
        return "Low"
    elif score < 70:
        return "Medium"
    else:
        return "High"

df["risk_band"] = df["risk_score"].apply(assign_risk_band)

high_risk_count = (df['risk_band'] == 'High').sum()
logging.info(f"Risk banding complete: {high_risk_count} high-risk taxpayers identified")

# -----------------------------
# 6. FEATURE IMPORTANCE ANALYSIS
# -----------------------------
df['is_high_risk'] = (df['risk_band'] == 'High').astype(int)
X_train = df[features_for_anomaly]
y_train = df['is_high_risk']

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

feat_imp = pd.DataFrame({
    'feature': features_for_anomaly,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feat_imp['feature'], feat_imp['importance'], color='steelblue')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.title('Feature Importance for High-Risk Prediction')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300)

logging.info("Feature importance analysis complete")
print("\n✅ Feature importance chart saved: 'feature_importance.png'")

# -----------------------------
# 7. SQL DATABASE STORAGE
# -----------------------------
conn = sqlite3.connect(DB_NAME)
df.to_sql("taxpayers", conn, if_exists="replace", index=False)

logging.info(f"Data stored in SQLite database: {DB_NAME}")
print(f"✅ Database created: {DB_NAME}")

# Sample SQL queries
high_risk_query = """
SELECT taxpayer_id, sector, risk_score, anomaly_flag, overdue_flag
FROM taxpayers
WHERE risk_band = 'High'
ORDER BY risk_score DESC
LIMIT 10
"""

sector_summary_query = """
SELECT 
    sector,
    COUNT(*) as total_taxpayers,
    AVG(risk_score) as avg_risk_score,
    SUM(anomaly_flag) as anomalies_detected,
    SUM(overdue_flag) as overdue_filers
FROM taxpayers
GROUP BY sector
ORDER BY avg_risk_score DESC
"""

high_risk_df = pd.read_sql_query(high_risk_query, conn)
sector_summary_df = pd.read_sql_query(sector_summary_query, conn)

conn.close()

# -----------------------------
# 8. EXPORT TO EXCEL
# -----------------------------
with pd.ExcelWriter(OUTPUT_EXCEL, engine='openpyxl') as writer:
    df.to_excel(writer, sheet_name='All_Taxpayers', index=False)
    high_risk_df.to_excel(writer, sheet_name='Top_10_High_Risk', index=False)
    sector_summary_df.to_excel(writer, sheet_name='Sector_Summary', index=False)
    feat_imp.to_excel(writer, sheet_name='Feature_Importance', index=False)

logging.info(f"Excel export complete: {OUTPUT_EXCEL}")

# -----------------------------
# 9. SUMMARY OUTPUT
# -----------------------------
print("\n" + "="*60)
print("🎯 TAXPAYER COMPLIANCE PIPELINE - EXECUTION SUMMARY")
print("="*60)

print("\n📊 Risk Band Distribution:")
print(df["risk_band"].value_counts().to_string())

print(f"\n🚨 Anomalies Detected: {anomaly_count}")
print(f"⚠️  High-Risk Taxpayers: {high_risk_count}")
print(f"📅 Overdue Filers (>120 days): {df['overdue_flag'].sum()}")

print("\n🏭 Sector Risk Summary:")
print(sector_summary_df.to_string(index=False))

print(f"\n📁 Outputs Generated:")
print(f"   • Excel: {OUTPUT_EXCEL}")
print(f"   • Database: {DB_NAME}")
print(f"   • Chart: feature_importance.png")
print(f"   • Log: compliance_pipeline.log")

print("\n" + "="*60)
print("✅ Pipeline completed successfully!")
print("="*60)

logging.info("Pipeline execution complete")
logging.info("=" * 50)