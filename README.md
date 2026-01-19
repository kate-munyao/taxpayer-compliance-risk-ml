# Taxpayer Compliance Risk Scoring System

An end-to-end machine learning pipeline for identifying high-risk and anomalous taxpayers using synthetic compliance data.

## Project Overview
This project simulates a compliance risk assessment system similar to those used by tax authorities. It applies machine learning to:
- Score taxpayer compliance risk
- Detect anomalies
- Classify taxpayers into risk bands
- Analyze sector-level risk patterns

##  Features
- Risk scoring using Scikit-learn
- Anomaly detection
- Sector-based analytics
- Feature importance visualization
- SQLite database backend
- Excel output for operational review

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- SQLite
- Streamlit (dashboard)
- Docker (optional deployment)

##  How to Run
```bash
pip install -r requirements.txt
python compliance_fraud_pipeline.py
