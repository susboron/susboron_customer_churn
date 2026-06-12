import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go


# 1. UI
st.set_page_config(
    page_title="Revenue Risk Engine",
    page_icon="conversation.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .metric-container {
        background-color: #1e2430;
        padding: 24px;
        border-radius: 12px;
        border-left: 5px solid #d8a6a6;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .metric-container.critical { border-left-color: #a00000; }
    .metric-container.success { border-left-color: #52796f; }
    .metric-title {
        font-size: 24px;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #8a99ad;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .metric-value { font-size: 32px; font-weight: 700; color: #ffffff; }
    .metric-subtitle { font-size: 16px; color: #a0aec0; margin-top: 4px; }
    </style>
""", unsafe_allow_html=True)




# 2. loading data and the artifact created
@st.cache_resource
def load_model_resources():
    with open('churn_model_artifacts.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_customer_database():
    df = pd.read_csv('live_customer_database.csv')
    card_names = {1: 'Star (Tier 1)', 2: 'Nova (Tier 2)', 3: 'Aurora (Tier 3)'}
    df['Loyalty_Card_Tier'] = df['Loyalty Card'].map(card_names)
    df['Enrollment_Strategy'] = df['Enrollment Type_Standard'].map(
        {True: 'Standard Enrollment', False: 'Promo/Partner Enrollment'}
    )
    return df

try:
    resources = load_model_resources()
    model = resources['model']
    feature_cols = resources['features']
    df = load_customer_database()
except Exception as e:
    st.error("Loading error")
    st.stop()





# 3. sidebar

st.sidebar.image("conversation.png", width=60)
st.sidebar.markdown("### **SIDEBAR**\n\n",)

st.sidebar.markdown("### **Segment Filters**")
selected_tier = st.sidebar.multiselect(
    "Filter by Loyalty Tier",
    options=list(df['Loyalty_Card_Tier'].unique()),
    default=list(df['Loyalty_Card_Tier'].unique())
)

selected_onboarding = st.sidebar.multiselect(
    "Filter by Source of Enrollment",
    options=list(df['Enrollment_Strategy'].unique()),
    default=list(df['Enrollment_Strategy'].unique())
)

risk_threshold = st.sidebar.slider(
    "Minimum Churn Risk Threshold",
    min_value=0.0, max_value=1.0, value=0.15, step=0.05
)

filtered_df = df[
    (df['Loyalty_Card_Tier'].isin(selected_tier)) & 
    (df['Enrollment_Strategy'].isin(selected_onboarding)) &
    (df['Churn Probability'] >= risk_threshold)
]






# 4. main tab

st.title("Airline Predictive Retention System")

tab_dashboard, tab_diagnostics, tab_simulator = st.tabs([
    "Main Console", 
    "Diagnostics", 
    "Retention Simulator"
])




# 4.1: Main Console
with tab_dashboard:
    total_at_risk = len(filtered_df)
    potential_revenue_loss = filtered_df['CLV'].sum()
    avg_momentum = filtered_df['sixmonth_avg_flights'].mean() / (filtered_df['Avg_Monthly_Flights'].mean() + 1e-5)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="metric-container critical"><div class="metric-title">Customers At Risk</div><div class="metric-value">{total_at_risk:,}</div><div class="metric-subtitle">Active accounts above {risk_threshold*100:.0f}% risk</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-container"><div class="metric-title">Exposed Pipeline Value</div><div class="metric-value">${potential_revenue_loss:,.2f}</div><div class="metric-subtitle">Sum of historical CLV currently at risk</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-container success"><div class="metric-title">Cohort Momentum</div><div class="metric-value">{avg_momentum:.2f}x</div><div class="metric-subtitle">Recent 6-month flight activity vs average</div></div>', unsafe_allow_html=True)

    left_chart, right_chart = st.columns([3, 2])
    with left_chart:
        st.markdown("### **Value Disconnect Map** (Salary vs. CLV)")
        fig = px.scatter(
            filtered_df, x="Salary", y="CLV", color="Churn Probability", size="Lifetime_Flights",
            color_continuous_scale=["#52796f", "#d8a6a6", "#a00000"], labels={"Churn Probability": "Risk Score"}, template="plotly_dark"
        )
        fig.update_layout(margin=dict(l=20, r=20, t=10, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
    with right_chart:
        st.markdown("### **Risk Concentration by Tier**")
        if not filtered_df.empty:
            fig_pie = px.pie(filtered_df, names='Loyalty_Card_Tier', values='CLV', hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu_r, template="plotly_dark")
            fig_pie.update_layout(margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No records match the current filters.")

    st.markdown("---")
    st.markdown("## Retention Info")
    if not filtered_df.empty:
        def assign_action(row):
            if row['Loyalty Card'] == 3 and row['Enrollment_Strategy'] == 'Promo/Partner Enrollment':
                return 'Experiential Upgrade (Lounge / Priority Access)'
            elif row['Loyalty Card'] in [1, 2] and row['Enrollment_Strategy'] == 'Promo/Partner Enrollment':
                return 'Points Promo (Zero-Fee Ticket Processing)'
            else:
                return 'Standard Engagement Email'

        action_df = filtered_df.copy()
        action_df['Targeted_Action'] = action_df.apply(assign_action, axis=1)
        formatted_display = action_df[['Salary', 'Loyalty_Card_Tier', 'Enrollment_Strategy', 'CLV', 'Churn Probability', 'Targeted_Action']].sort_values(by='Churn Probability', ascending=False)
        st.dataframe(formatted_display.style.background_gradient(subset=['Churn Probability'], cmap='YlOrRd').format({'Salary': '${:,.0f}', 'CLV': '${:,.2f}', 'Churn Probability': '{:.2%}'}), use_container_width=True)
        
        st.download_button(label="Export Retention Details", data=formatted_display.to_csv(index=True).encode('utf-8'), file_name="retention_targets.csv", mime="text/csv")




        
# 4.2: Diagnostics

with tab_diagnostics:
    st.markdown("## Feature Mechanics")
    st.markdown("I used a Random Forest Classifier trained on over 16,000 customer records.")
    
    diag_left, diag_right = st.columns([2, 2])
    
    with diag_left:
        st.markdown("### **Features Ranked by Importance**")
        importances = model.feature_importances_
        feat_imp_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importances}).sort_values(by='Importance', ascending=True).tail(10)
        
        fig_imp = px.bar(feat_imp_df, x='Importance', y='Feature', orientation='h', template='plotly_dark', color_discrete_sequence=['#d8a6a6'])
        fig_imp.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_imp, use_container_width=True)
        
    with diag_right:
        st.markdown("### Risk Simulator")
        st.markdown(" Modify values below to run a direct inference query against my model.")
        
        sim_avg_flights = st.number_input("Average Monthly Flights (Lifetime)", min_value=0.0, max_value=20.0, value=2.0)
        sim_recent_flights = st.number_input("Average Monthly Flights (Last 6 Months)", min_value=0.0, max_value=20.0, value=0.5)
        sim_lifetime_flights = st.number_input("Total Lifetime Flights Taken", min_value=0, max_value=500, value=40)
        sim_points_redeemed = st.number_input("Total Lifetime Points Redeemed", min_value=0, max_value=100000, value=500)
        
        # Create dummy vector matching original model structure
        input_data = np.zeros((1, len(feature_cols)))
        sim_df = pd.DataFrame(input_data, columns=feature_cols)
        
        # Overlay fields that map exactly to model inputs
        if 'Avg_Monthly_Flights' in sim_df.columns: sim_df['Avg_Monthly_Flights'] = sim_avg_flights
        if 'sixmonth_avg_flights' in sim_df.columns: sim_df['sixmonth_avg_flights'] = sim_recent_flights
        if 'Lifetime_Flights' in sim_df.columns: sim_df['Lifetime_Flights'] = sim_lifetime_flights
        if 'Lifetime_Points_Redeemed' in sim_df.columns: sim_df['Lifetime_Points_Redeemed'] = sim_points_redeemed
        
        if st.button("Compute Real-Time Inference Risk Profile"):
            risk_score = model.predict_proba(sim_df)[0][1]
            st.markdown(f"### Calculated Account Churn Probability: **{risk_score:.2%}**")
            if risk_score >= 0.5:
                st.error("HIGH FLIGHT RISK. Operational intervention required.")
            else:
                st.success("HEALTHY & STABLE. Retain baseline tracking.")


        


        
        
# 4.3: Retention Simulator
with tab_simulator:
    st.markdown("## Operational ROI Evaluation Desk")
    st.markdown("To estimate the ROI before implementing the system.")
    
    sim_col1, sim_col2 = st.columns([1, 2])
    
    with sim_col1:
        st.markdown("### **Campaign Cost Inputs**")
        est_conversion_rate = st.slider("Target Save Success Rate (%)", min_value=5, max_value=100, value=25, step=5) / 100.0
        campaign_cost_per_user = st.number_input("Average Cost to Execute Intervention ($ / user)", min_value=5.0, max_value=500.0, value=50.0)
        
        total_exposed_count = len(filtered_df)
        total_exposed_value = filtered_df['CLV'].sum()
        
    with sim_col2:
        st.markdown("### **Projected ROI Model**")
        
        total_execution_spend = total_exposed_count * campaign_cost_per_user
        projected_saved_users = int(total_exposed_count * est_conversion_rate)
        
        avg_clv_of_risk_cohort = filtered_df['CLV'].mean() if not filtered_df.empty else 0
        gross_recovered_pipeline = projected_saved_users * avg_clv_of_risk_cohort
        net_retained_value = gross_recovered_pipeline - total_execution_spend
        
        sim_m1, sim_m2, sim_m3 = st.columns(3)
        with sim_m1:
            st.metric("Total Operational Budget Req.", f"${total_execution_spend:,.2f}")
        with sim_m2:
            st.metric(" Accounts Saved", f"{projected_saved_users:,} members")
        with sim_m3:
            st.metric("Net Financial Return (ROI)", f"${net_retained_value:,.2f}")
            
        fig_roi = go.Figure(data=[
            go.Bar(name='Exposed Risk Pipeline', x=['Financial Comparison'], y=[total_exposed_value], marker_color='#a00000'),
            go.Bar(name='Net Recovered Value', x=['Financial Comparison'], y=[max(0, net_retained_value)], marker_color='#52796f')
        ])
        fig_roi.update_layout(
            barmode='group', 
            template='plotly_dark', 
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_roi, use_container_width=True)
        
        st.markdown(f"**Execution Spend Requirement:** Out of the exposed pipeline, this campaign requires an operational budget of **${total_execution_spend:,.2f}** to deploy.")