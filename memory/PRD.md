# SkyCity Auckland — Profit Optimization Platform
**Created:** 2026-02-18  
**Stack:** Python 3 · Streamlit · scikit-learn · XGBoost · Plotly  
**Serving:** Port 3000 via Supervisor (replaces React frontend)

---

## Problem Statement
SkyCity Auckland Restaurants & Bars lacks predictive intelligence for channel-mix decisions. 
This platform provides ML-driven profit forecasting and prescriptive optimization across 
In-Store, Uber Eats, DoorDash, and Self-Delivery channels.

---

## Dataset
- **File:** `/app/SkyCity Auckland Restaurants & Bars - SkyCity Auckland Restaurants & Bars.csv`
- **Records:** 1,696 unique restaurants
- **Cuisine types:** Burgers, Chicken Dishes, Chinese, Indian, Japanese, Kebabs/Mediterranean, Pizza, Thai
- **Segments:** Cafe, QSR, Ghost Kitchen, Full-service
- **Subregions:** North Shore, South Auckland, West Auckland, CBD

---

## Architecture
```
/app/
  app.py                  # Main Streamlit application (all pages)
  .streamlit/config.toml  # Streamlit config (port 3000, light theme)
  SkyCity Auckland ....csv  # Source dataset
  frontend/
    package.json          # yarn start -> runs streamlit on port 3000
  backend/
    server.py             # Minimal FastAPI health check on port 8001
  memory/PRD.md           # This file
```

---

## Core Requirements (Static)

### Pages / Modules
1. **Overview** – KPI cards, cuisine revenue, segment profit, channel mix donut, subregion comparison, key observations
2. **Exploratory Analysis** – 4 tabs: Distribution & Revenue | Cost Analysis | Channel Dynamics | Correlation Matrix
3. **Predictive Models** – Train 4 models (Linear Regression, Random Forest, Gradient Boosting, XGBoost), compare RMSE/R²/MAE, feature importance, predicted vs actual, residuals
4. **What-If Simulator** – Channel mix sliders, cost/ops sliders, restaurant profile, real-time 4-model prediction, channel breakdown, commission sensitivity sweep
5. **Optimization Panel** – 4 tabs: Channel Efficiency | Commission Analysis | Self-Delivery Threshold | Recommendations

### ML Features (19 features)
- Channel shares, commission, delivery cost, radius, growth factor, AOV, orders, COGS rate, OPEX rate
- Interaction terms: Commission×UE_share, DeliveryCost×SD_share, GrowthAdj_Orders, CostToRevenue
- Encoded: CuisineType, Segment, Subregion

---

## Implementation History

### 2026-02-18 — MVP
- Built full Streamlit app (`app.py`) with 5 pages, 19 ML features
- All 4 ML models working: LR R²=0.89, RF R²=0.97, GB R²=0.98, XGB R²=0.98
- XGBoost best model with RMSE=$806, R²=0.9798, MAE=$585
- Streamlit configured to run on port 3000 replacing React frontend
- All charts use Plotly with light/clean enterprise theme

---

## Prioritized Backlog

### P0 (Core — Done)
- [x] Dataset loading & feature engineering
- [x] Overview page with KPIs
- [x] EDA with 4 tabs
- [x] 4 ML models with comparison
- [x] What-If Simulator with sliders
- [x] Optimization Panel with recommendations

### P1 (Next)
- [ ] Persist trained models (save/load with joblib) to avoid re-training
- [ ] Export scenario results to CSV/PDF
- [ ] Cross-validation scores for models
- [ ] Confidence intervals on predictions
- [ ] Multi-scenario comparison (save A vs B)

### P2 (Future)
- [ ] Time-series forecasting (if monthly data becomes available)
- [ ] Individual restaurant drill-down page
- [ ] Monte Carlo simulation for risk analysis
- [ ] Custom commission negotiation tool
- [ ] PDF executive report generator
