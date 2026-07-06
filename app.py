import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnGuard — Customer Retention Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #0d0d0d; color: #e8e8e8; }
  [data-testid="stSidebar"] { background-color: #111111; border-right: 1px solid #1e1e1e; }
  .main .block-container { padding: 2rem 2.5rem; max-width: 1200px; }
  .metric-card { background: #141414; border: 1px solid #1e1e1e; border-radius: 12px; padding: 1.2rem 1.5rem; text-align: center; }
  .metric-value { font-size: 2rem; font-weight: 700; }
  .metric-label { font-size: 0.75rem; color: #555; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }
  .risk-high { background: #1a0808; border: 1px solid #ef4444; border-radius: 10px; padding: 1rem 1.5rem; }
  .risk-medium { background: #1a1408; border: 1px solid #eab308; border-radius: 10px; padding: 1rem 1.5rem; }
  .risk-low { background: #081a08; border: 1px solid #22c55e; border-radius: 10px; padding: 1rem 1.5rem; }
  .section-label { font-size: 0.72rem; font-weight: 600; color: #555; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.8rem; }
  .insight-card { background: #141414; border: 1px solid #1e1e1e; border-left: 3px solid #7c6bff; border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 0.6rem; font-size: 0.88rem; }
  .recommendation-card { background: #0f1a0f; border: 1px solid #22c55e; border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 0.6rem; font-size: 0.88rem; }
  .stButton > button { background: #7c6bff !important; color: #fff !important; border: none !important; border-radius: 8px !important; font-weight: 600 !important; }
  #MainMenu { visibility: hidden; } footer { visibility: hidden; } header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── Load Models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    with open('xgboost_churn_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('shap_explainer.pkl', 'rb') as f:
        explainer = pickle.load(f)
    with open('feature_cols.pkl', 'rb') as f:
        feature_cols = pickle.load(f)
    with open('tenure_churn.json', 'r') as f:
        tenure_churn = json.load(f)
    return model, scaler, explainer, feature_cols, tenure_churn

model, scaler, explainer, feature_cols, tenure_churn = load_models()

# ─── Encoding Maps ─────────────────────────────────────────────────────────────
login_device_map = {'Mobile Phone': 1, 'Computer': 0, 'Tablet': 2}
payment_map = {'Debit Card': 1, 'UPI': 5, 'Credit Card': 0, 'Cash on Delivery': 2, 'E wallet': 3, 'CC': 4}
gender_map = {'Male': 1, 'Female': 0}
order_cat_map = {'Laptop & Accessory': 2, 'Mobile Phone': 3, 'Fashion': 1, 'Grocery': 4, 'Others': 5, 'Mobile': 6}
marital_map = {'Single': 2, 'Married': 1, 'Divorced': 0}

# ─── Helper Functions ──────────────────────────────────────────────────────────
def get_risk_level(prob):
    if prob >= 0.6:
        return "HIGH RISK", "#ef4444", "risk-high"
    elif prob >= 0.3:
        return "MEDIUM RISK", "#eab308", "risk-medium"
    else:
        return "LOW RISK", "#22c55e", "risk-low"

def get_recommendations(input_data, prob):
    recs = []
    if input_data['Tenure'] <= 6:
        recs.append("🎯 New customer — Launch personalized onboarding campaign within 48 hours")
    if input_data['Complain'] == 1:
        recs.append("🚨 Complaint filed — Escalate to customer success team immediately")
    if input_data['HourSpendOnApp'] < 2:
        recs.append("📱 Low app engagement — Send push notification with personalized offers")
    if input_data['DaySinceLastOrder'] > 7:
        recs.append("⏰ Inactive customer — Trigger re-engagement email with discount code")
    if input_data['CouponUsed'] > 2:
        recs.append("🏷️ High coupon dependency — Shift to loyalty points program")
    if input_data['SatisfactionScore'] <= 2:
        recs.append("⭐ Low satisfaction — Schedule proactive customer feedback call")
    if input_data['CashbackAmount'] > 200:
        recs.append("💰 High cashback user — Offer exclusive membership tier benefits")
    if prob >= 0.6 and not recs:
        recs.append("🔴 High risk detected — Assign dedicated account manager")
    if not recs:
        recs.append("✅ Customer is healthy — Continue standard engagement")
    return recs

def predict_churn(input_dict):
    input_df = pd.DataFrame([input_dict])
    input_scaled = scaler.transform(input_df[feature_cols])
    prob = model.predict_proba(input_scaled)[0][1]
    shap_vals = explainer.shap_values(input_scaled)
    return prob, shap_vals, input_scaled

def revenue_at_risk(n_customers, avg_order_value=163, orders_per_month=2):
    monthly = n_customers * avg_order_value * orders_per_month
    annual = monthly * 12
    return monthly, annual

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ ChurnGuard")
    st.markdown("*Customer Retention Intelligence*")
    st.markdown("---")
    tab_choice = st.radio(
        "Navigation",
        ["🎯 Single Prediction", "📊 Batch Prediction", "📈 Business Insights", "🧠 Model Explainability"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown('<div class="section-label">Model Performance</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.82rem; color:#888;">
    🏆 XGBoost Classifier<br>
    📊 ROC-AUC: <b style="color:#7c6bff;">99.77%</b><br>
    🎯 Accuracy: <b style="color:#7c6bff;">98%</b><br>
    🔍 Churn Recall: <b style="color:#7c6bff;">94%</b>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SINGLE PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
if tab_choice == "🎯 Single Prediction":
    st.markdown("## 🎯 Single Customer Churn Prediction")
    st.markdown("Enter customer details to predict churn probability and get retention recommendations.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="section-label">Customer Profile</div>', unsafe_allow_html=True)
        tenure = st.slider("Tenure (months)", 0, 33, 6)
        gender = st.selectbox("Gender", ['Male', 'Female'])
        marital = st.selectbox("Marital Status", ['Single', 'Married', 'Divorced'])
        city_tier = st.selectbox("City Tier", [1, 2, 3])
        n_address = st.slider("Number of Addresses", 1, 22, 3)

    with col2:
        st.markdown('<div class="section-label">Behavioral Data</div>', unsafe_allow_html=True)
        login_device = st.selectbox("Preferred Login Device", ['Mobile Phone', 'Computer', 'Tablet'])
        hour_app = st.slider("Hours Spent on App (daily)", 0.5, 5.0, 3.0, 0.5)
        n_devices = st.slider("Number of Devices Registered", 1, 6, 3)
        order_cat = st.selectbox("Preferred Order Category", ['Laptop & Accessory', 'Mobile Phone', 'Fashion', 'Grocery', 'Others'])
        payment = st.selectbox("Preferred Payment Mode", ['Debit Card', 'UPI', 'Credit Card', 'Cash on Delivery', 'E wallet'])

    with col3:
        st.markdown('<div class="section-label">Transaction Data</div>', unsafe_allow_html=True)
        satisfaction = st.slider("Satisfaction Score (1-5)", 1, 5, 3)
        complain = st.selectbox("Complaint Filed?", ['No', 'Yes'])
        order_hike = st.slider("Order Amount Hike from Last Year (%)", 11, 26, 15)
        coupon_used = st.slider("Coupons Used (last month)", 0, 4, 1)
        order_count = st.slider("Order Count (last month)", 1, 6, 2)
        days_last = st.slider("Days Since Last Order", 0, 15, 3)
        cashback = st.slider("Cashback Amount (₹)", 50, 300, 163)
        warehouse = st.slider("Warehouse to Home Distance (km)", 5, 37, 14)

    st.markdown("---")

    if st.button("🔮 Predict Churn Risk", use_container_width=True):
        input_dict = {
            'Tenure': tenure,
            'PreferredLoginDevice': login_device_map[login_device],
            'CityTier': city_tier,
            'WarehouseToHome': warehouse,
            'PreferredPaymentMode': payment_map[payment],
            'Gender': gender_map[gender],
            'HourSpendOnApp': hour_app,
            'NumberOfDeviceRegistered': n_devices,
            'PreferedOrderCat': order_cat_map.get(order_cat, 2),
            'SatisfactionScore': satisfaction,
            'MaritalStatus': marital_map[marital],
            'NumberOfAddress': n_address,
            'Complain': 1 if complain == 'Yes' else 0,
            'OrderAmountHikeFromlastYear': order_hike,
            'CouponUsed': coupon_used,
            'OrderCount': order_count,
            'DaySinceLastOrder': days_last,
            'CashbackAmount': cashback
        }

        prob, shap_vals, input_scaled = predict_churn(input_dict)
        risk_label, risk_color, risk_class = get_risk_level(prob)
        recs = get_recommendations(input_dict, prob)

        # ── Results ──
        r1, r2, r3, r4 = st.columns(4)
        with r1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:{risk_color}">{prob*100:.1f}%</div>
                <div class="metric-label">Churn Probability</div>
            </div>""", unsafe_allow_html=True)
        with r2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:{risk_color}">{risk_label}</div>
                <div class="metric-label">Risk Level</div>
            </div>""", unsafe_allow_html=True)
        with r3:
            monthly, annual = revenue_at_risk(1, cashback, order_count)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:#eab308">₹{monthly:,.0f}</div>
                <div class="metric-label">Monthly Revenue at Risk</div>
            </div>""", unsafe_allow_html=True)
        with r4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:#7c6bff">{tenure}m</div>
                <div class="metric-label">Customer Tenure</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Churn Probability Gauge ──
        col_gauge, col_recs = st.columns([1, 1])

        with col_gauge:
            st.markdown('<div class="section-label">Churn Probability Gauge</div>', unsafe_allow_html=True)
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Churn Risk %", 'font': {'color': '#e8e8e8', 'size': 14}},
                number={'suffix': "%", 'font': {'color': risk_color, 'size': 36}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': '#555'},
                    'bar': {'color': risk_color},
                    'bgcolor': '#141414',
                    'bordercolor': '#1e1e1e',
                    'steps': [
                        {'range': [0, 30], 'color': '#0f1a0f'},
                        {'range': [30, 60], 'color': '#1a1408'},
                        {'range': [60, 100], 'color': '#1a0808'},
                    ],
                    'threshold': {'line': {'color': risk_color, 'width': 4}, 'value': prob * 100}
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor='#0d0d0d', plot_bgcolor='#0d0d0d',
                font={'color': '#e8e8e8'}, height=280, margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_recs:
            st.markdown('<div class="section-label">Retention Recommendations</div>', unsafe_allow_html=True)
            for rec in recs:
                st.markdown(f'<div class="recommendation-card">{rec}</div>', unsafe_allow_html=True)

        # ── SHAP Explanation ──
        st.markdown("---")
        st.markdown('<div class="section-label">Why Will This Customer Churn? (SHAP Explanation)</div>', unsafe_allow_html=True)

        shap_df = pd.DataFrame({
            'Feature': feature_cols,
            'SHAP Value': shap_vals[0]
        }).sort_values('SHAP Value', key=abs, ascending=False).head(10)

        colors = ['#ef4444' if v > 0 else '#22c55e' for v in shap_df['SHAP Value']]
        fig_shap = go.Figure(go.Bar(
            x=shap_df['SHAP Value'],
            y=shap_df['Feature'],
            orientation='h',
            marker_color=colors
        ))
        fig_shap.update_layout(
            title="Top 10 Factors Driving Churn (Red = Increases Risk, Green = Decreases Risk)",
            paper_bgcolor='#0d0d0d', plot_bgcolor='#141414',
            font={'color': '#e8e8e8', 'size': 11},
            xaxis=dict(gridcolor='#1e1e1e', title='SHAP Value'),
            yaxis=dict(gridcolor='#1e1e1e'),
            height=350, margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_shap, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — BATCH PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
elif tab_choice == "📊 Batch Prediction":
    st.markdown("## 📊 Batch Customer Churn Prediction")
    st.markdown("Upload a CSV file with customer data to predict churn for all customers at once.")
    st.markdown("---")

    st.info("📋 CSV must have columns: Tenure, PreferredLoginDevice, CityTier, WarehouseToHome, PreferredPaymentMode, Gender, HourSpendOnApp, NumberOfDeviceRegistered, PreferedOrderCat, SatisfactionScore, MaritalStatus, NumberOfAddress, Complain, OrderAmountHikeFromlastYear, CouponUsed, OrderCount, DaySinceLastOrder, CashbackAmount")

    uploaded = st.file_uploader("Upload Customer CSV", type=['csv'])

    if uploaded:
        batch_df = pd.read_csv(uploaded)
        st.markdown(f"**{len(batch_df)} customers loaded**")
        st.dataframe(batch_df.head(5), use_container_width=True)

        if st.button("🔮 Predict All Customers", use_container_width=True):
            try:
                batch_scaled = scaler.transform(batch_df[feature_cols])
                probs = model.predict_proba(batch_scaled)[:, 1]
                batch_df['ChurnProbability'] = (probs * 100).round(1)
                batch_df['RiskLevel'] = pd.cut(probs,
                    bins=[0, 0.3, 0.6, 1.0],
                    labels=['Low Risk', 'Medium Risk', 'High Risk'])

                # Summary metrics
                high = (batch_df['RiskLevel'] == 'High Risk').sum()
                medium = (batch_df['RiskLevel'] == 'Medium Risk').sum()
                low = (batch_df['RiskLevel'] == 'Low Risk').sum()
                monthly_risk, annual_risk = revenue_at_risk(high)

                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#ef4444">{high}</div><div class="metric-label">High Risk Customers</div></div>', unsafe_allow_html=True)
                with m2:
                    st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#eab308">{medium}</div><div class="metric-label">Medium Risk</div></div>', unsafe_allow_html=True)
                with m3:
                    st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#22c55e">{low}</div><div class="metric-label">Low Risk</div></div>', unsafe_allow_html=True)
                with m4:
                    st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#eab308">₹{monthly_risk:,.0f}</div><div class="metric-label">Monthly Revenue at Risk</div></div>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Risk distribution chart
                fig_dist = px.histogram(batch_df, x='ChurnProbability', nbins=20,
                    title='Churn Probability Distribution',
                    color_discrete_sequence=['#7c6bff'])
                fig_dist.update_layout(paper_bgcolor='#0d0d0d', plot_bgcolor='#141414',
                    font={'color': '#e8e8e8'}, height=300)
                st.plotly_chart(fig_dist, use_container_width=True)

                # High risk customers table
                st.markdown('<div class="section-label">High Risk Customers — Take Action Now</div>', unsafe_allow_html=True)
                high_risk_df = batch_df[batch_df['RiskLevel'] == 'High Risk'].sort_values('ChurnProbability', ascending=False)
                st.dataframe(high_risk_df, use_container_width=True, hide_index=True)

                # Download
                csv_out = batch_df.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Download Full Results CSV", csv_out,
                    "churn_predictions.csv", "text/csv", use_container_width=True)

            except Exception as e:
                st.error(f"Error: {e} — Make sure CSV has all required columns")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — BUSINESS INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
elif tab_choice == "📈 Business Insights":
    st.markdown("## 📈 Business Insights & Revenue Impact")
    st.markdown("---")

    # ── Revenue Calculator ──
    st.markdown("### 💰 Revenue at Risk Calculator")
    col_calc1, col_calc2, col_calc3 = st.columns(3)
    with col_calc1:
        n_at_risk = st.number_input("Number of High-Risk Customers", 1, 10000, 50)
    with col_calc2:
        avg_order = st.number_input("Average Order Value (₹)", 50, 1000, 163)
    with col_calc3:
        orders_pm = st.number_input("Orders per Month", 1, 20, 2)

    monthly_rev = n_at_risk * avg_order * orders_pm
    annual_rev = monthly_rev * 12
    retention_cost = n_at_risk * 200
    roi = ((monthly_rev - retention_cost) / retention_cost) * 100

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#ef4444">₹{monthly_rev:,.0f}</div><div class="metric-label">Monthly Revenue at Risk</div></div>', unsafe_allow_html=True)
    with r2:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#ef4444">₹{annual_rev:,.0f}</div><div class="metric-label">Annual Revenue at Risk</div></div>', unsafe_allow_html=True)
    with r3:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#eab308">₹{retention_cost:,.0f}</div><div class="metric-label">Est. Retention Campaign Cost</div></div>', unsafe_allow_html=True)
    with r4:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#22c55e">{roi:.0f}%</div><div class="metric-label">ROI if Retained</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Tenure Group Churn ──
    st.markdown("### 📉 Churn Rate by Customer Tenure")
    tenure_df = pd.DataFrame({
        'Tenure Group': list(tenure_churn.keys()),
        'Churn Rate (%)': list(tenure_churn.values())
    })

    fig_tenure = px.bar(tenure_df, x='Tenure Group', y='Churn Rate (%)',
        color='Churn Rate (%)',
        color_continuous_scale=['#22c55e', '#eab308', '#ef4444'],
        title='Churn Rate by Tenure Group — First 6 Months are Critical!')
    fig_tenure.update_layout(paper_bgcolor='#0d0d0d', plot_bgcolor='#141414',
        font={'color': '#e8e8e8'}, height=350)
    st.plotly_chart(fig_tenure, use_container_width=True)

    st.markdown("""
    <div class="insight-card">
    💡 <b>Key Finding:</b> Customers with 0-6 months tenure have <b>25.9% churn rate</b> vs only <b>1.3%</b> for 20+ month customers.
    The first 6 months is the <b>critical retention window</b> — invest heavily in onboarding during this period.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-card">
    💡 <b>Business Action:</b> If you can move a customer past the 6-month mark, their churn probability drops by <b>16%</b>.
    A ₹500 investment in new customer onboarding can save ₹5,000+ in lifetime value.
    </div>
    """, unsafe_allow_html=True)

    # ── Churn Drivers Summary ──
    st.markdown("### 🔑 Top Churn Drivers & Business Actions")
    drivers = {
        "Tenure < 6 months": ("25.9% churn rate", "Invest in onboarding — welcome calls, tutorials, first-order incentives"),
        "Complaint Filed": ("High churn signal", "Resolve within 24hrs — assign dedicated support agent"),
        "Low App Engagement": ("< 2 hrs/day = risk", "Send personalized push notifications — show recently viewed items"),
        "Days Since Last Order > 7": ("Inactivity = churn", "Trigger automated re-engagement emails with discount"),
        "High Coupon Dependency": ("Discount-driven users", "Shift to loyalty points — reduce discount bleeding"),
    }

    for driver, (stat, action) in drivers.items():
        col_d1, col_d2 = st.columns([1, 2])
        with col_d1:
            st.markdown(f'<div class="risk-high"><b>{driver}</b><br><small style="color:#ef4444">{stat}</small></div>', unsafe_allow_html=True)
        with col_d2:
            st.markdown(f'<div class="recommendation-card">✅ {action}</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MODEL EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════════════════════
elif tab_choice == "🧠 Model Explainability":
    st.markdown("## 🧠 Model Explainability & Performance")
    st.markdown("---")

    # Model comparison
    st.markdown("### 📊 Model Comparison")
    model_data = {
        'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'],
        'ROC-AUC': [0.8251, 0.9952, 0.9977],
        'Accuracy': [0.73, 0.98, 0.98],
        'Churn Recall': [0.77, 0.92, 0.94]
    }
    model_df = pd.DataFrame(model_data)
    st.dataframe(model_df, use_container_width=True, hide_index=True)

    fig_model = px.bar(model_df, x='Model', y=['ROC-AUC', 'Accuracy', 'Churn Recall'],
        barmode='group', title='Model Performance Comparison',
        color_discrete_sequence=['#7c6bff', '#22c55e', '#eab308'])
    fig_model.update_layout(paper_bgcolor='#0d0d0d', plot_bgcolor='#141414',
        font={'color': '#e8e8e8'}, height=350)
    st.plotly_chart(fig_model, use_container_width=True)

    st.markdown("""
    <div class="insight-card">
    🏆 <b>Why XGBoost?</b> XGBoost handles class imbalance better, captures non-linear relationships between features,
    and provides native feature importance. With 99.77% ROC-AUC and 94% churn recall, it correctly identifies
    <b>179 out of 190 actual churners</b> in the test set — missing only 11.
    </div>
    """, unsafe_allow_html=True)

    # SHAP Global
    st.markdown("### 🔍 Global Feature Importance (SHAP)")
    feature_importance = {
        'Tenure': 2.45, 'CashbackAmount': 0.85, 'Complain': 0.72,
        'NumberOfAddress': 0.68, 'DaySinceLastOrder': 0.61, 'WarehouseToHome': 0.52,
        'CityTier': 0.48, 'CouponUsed': 0.45, 'OrderAmountHikeFromlastYear': 0.42,
        'OrderCount': 0.38
    }
    shap_df = pd.DataFrame(list(feature_importance.items()), columns=['Feature', 'SHAP Value'])
    fig_shap = px.bar(shap_df.sort_values('SHAP Value'),
        x='SHAP Value', y='Feature', orientation='h',
        title='Mean SHAP Values — Global Feature Importance',
        color='SHAP Value', color_continuous_scale=['#22c55e', '#eab308', '#ef4444'])
    fig_shap.update_layout(paper_bgcolor='#0d0d0d', plot_bgcolor='#141414',
        font={'color': '#e8e8e8'}, height=400)
    st.plotly_chart(fig_shap, use_container_width=True)

    st.markdown("""
    <div class="insight-card">
    💡 <b>Tenure dominates</b> with a SHAP value of 2.45 — far above all other features.
    This confirms that customer lifecycle stage is the #1 predictor of churn.
    Cashback amount and complaints are secondary signals that compound the risk.
    </div>
    """, unsafe_allow_html=True)