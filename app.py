import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import boto3
import io
from datetime import timedelta, date

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="OAK Analytics Dashboard", page_icon="📊", layout="wide")
st.title("OAK Performance & Financial Dashboard")

# --- 2. DATA LOADING & CACHING ---
@st.cache_data(ttl=3600)
def load_data_from_s3():
    """Loads Parquet files from S3 into Pandas DataFrames."""
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
            region_name=st.secrets["AWS_REGION"]
        )
        bucket = st.secrets["S3_BUCKET_NAME"]

        def fetch_parquet(key):
            try:
                obj = s3_client.get_object(Bucket=bucket, Key=key)
                return pd.read_parquet(io.BytesIO(obj['Body'].read()))
            except Exception as e:
                return pd.DataFrame()

        # Load existing pipeline data
        df_subs = fetch_parquet("stripe_subscriptions.parquet")
        df_payments = fetch_parquet("stripe_payments.parquet")

        # Load Ads and AWS data (assuming these are added to your ETL pipeline)
        df_ads = fetch_parquet("google_ads_data.parquet")
        df_aws = fetch_parquet("aws_cost_data.parquet")

        # Basic Preprocessing for dates
        if not df_subs.empty:
            if 'created_at' in df_subs.columns:
                df_subs['created_at'] = pd.to_datetime(df_subs['created_at'])

            # Safely check if canceled_at exists. If not, create it as empty (NaT)
            if 'canceled_at' in df_subs.columns:
                df_subs['canceled_at'] = pd.to_datetime(df_subs['canceled_at'])
            else:
                df_subs['canceled_at'] = pd.NaT

        if not df_payments.empty and 'created_at' in df_payments.columns:
            df_payments['created_at'] = pd.to_datetime(df_payments['created_at'])

        return df_subs, df_payments, df_ads, df_aws

    except Exception as e:
        st.error(f"Error connecting to AWS: {e}. Please check your Streamlit secrets.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

df_subs, df_payments, df_ads, df_aws = load_data_from_s3()

if df_subs.empty:
    st.warning("No subscription data found. Please ensure the ETL pipeline has run.")
    st.stop()

# --- 3. SIDEBAR / GLOBAL FILTERS ---
st.sidebar.header("Global Filters")
min_date = df_subs['created_at'].min().date() if not df_subs.empty else date(2023, 1, 1)
max_date = date.today()

start_date = st.sidebar.date_input("Start Date", min_date)
end_date = st.sidebar.date_input("End Date", max_date)
time_grouping = st.sidebar.selectbox("Time Grouping", ["Daily", "Weekly", "Monthly"])

# Map grouping to Pandas frequency strings
freq_map = {"Daily": "D", "Weekly": "W-MON", "Monthly": "ME"}
freq = freq_map[time_grouping]

# Filter base dataframes
mask_subs = (df_subs['created_at'].dt.date >= start_date) & (df_subs['created_at'].dt.date <= end_date)
df_subs_filtered = df_subs.loc[mask_subs].copy()

if not df_payments.empty:
    mask_pay = (df_payments['created_at'].dt.date >= start_date) & (df_payments['created_at'].dt.date <= end_date)
    df_payments_filtered = df_payments.loc[mask_pay].copy()
else:
    df_payments_filtered = pd.DataFrame()

# --- 4. HELPER FUNCTIONS FOR HISTORICAL LOGIC ---
def get_historical_status_counts(df, start, end, freq_str):
    """Reconstructs historical active/trialing users point-in-time."""
    date_range = pd.date_range(start=start, end=end, freq=freq_str)
    history = []

    # Normalize dates (removes the time portion, keeping it as a safe Pandas Timestamp)
    # This prevents the 'datetime64[ns] vs date' TypeError
    created_dates = df['created_at'].dt.normalize()
    canceled_dates = df['canceled_at'].dt.normalize()

    for dt in date_range:
        current_ts = dt.normalize()

        # Created on or before this date
        created_before = created_dates <= current_ts

        # Has not canceled, or canceled strictly after this date
        active_on_date = created_before & (canceled_dates.isna() | (canceled_dates > current_ts))

        snapshot = df[active_on_date]
        active_count = len(snapshot[snapshot['status'] == 'active'])
        trial_count = len(snapshot[snapshot['status'] == 'trialing'])

        history.append({
            'Date': dt.date(), # Convert to standard date only for the final chart display
            'Active Paid': active_count,
            'Trialing': trial_count,
            'Total': active_count + trial_count
        })

    return pd.DataFrame(history)

# --- 5. DASHBOARD LAYOUT (TABS) ---
tab1, tab2, tab3, tab4 = st.tabs(["Overview & Snapshots", "Trends & Projections", "Marketing & CAC", "Churn Analysis"])

# ==========================================
# TAB 1: OVERVIEW & SNAPSHOTS (Metrics 3-7, 9, 10)
# ==========================================
with tab1:
    st.subheader("Current Platform Snapshot")

    # Calculate snapshot metrics
    current_active = len(df_subs[df_subs['status'] == 'active'])
    current_trialing = len(df_subs[df_subs['status'] == 'trialing'])

    canceling_active = len(df_subs[(df_subs['status'] == 'active') & (df_subs['cancel_at_period_end'] == True)])
    canceling_trial = len(df_subs[(df_subs['status'] == 'trialing') & (df_subs['cancel_at_period_end'] == True)])

    net_revenue = df_payments_filtered[df_payments_filtered['status'] == 'succeeded']['net_usd'].sum() if not df_payments_filtered.empty else 0
    ad_spend = df_ads['spend'].sum() if not df_ads.empty else 0
    cloud_cost = df_aws['cost'].sum() if not df_aws.empty else 0

    # Row 1: Users
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Paid Subscribers", f"{current_active:,}")
    col2.metric("Current Trialing", f"{current_trialing:,}")
    col3.metric("Active (Canceling at End)", f"{canceling_active:,}")
    col4.metric("Trialing (Canceling at End)", f"{canceling_trial:,}")

    # Row 2: Financials
    st.markdown("---")
    st.subheader("Financial Holistic View")
    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
    f_col1.metric("Net Revenue", f"${net_revenue:,.2f}")
    f_col2.metric("Total Ad Spend", f"${ad_spend:,.2f}")
    f_col3.metric("Total Cloud Cost", f"${cloud_cost:,.2f}")
    f_col4.metric("Gross Margin (Rev - Ads - Cloud)", f"${(net_revenue - ad_spend - cloud_cost):,.2f}")

# ==========================================
# TAB 2: TRENDS & PROJECTIONS (Metrics 1, 2, 12)
# ==========================================
with tab2:
    st.subheader("Subscription Trends")

    # Cumulative Historical Trend
    hist_df = get_historical_status_counts(df_subs, start_date, end_date, freq)
    if not hist_df.empty:
        fig_hist = px.line(hist_df, x='Date', y=['Active Paid', 'Trialing'],
                           title="Cumulative Subscription Trend (Point-in-Time)",
                           labels={'value': 'Users', 'variable': 'Status'})
        st.plotly_chart(fig_hist, use_container_width=True)

    # New Subscriptions Trend (Bar Chart)
    new_subs = df_subs_filtered.set_index('created_at').resample(freq).size().reset_index(name='New Subscriptions')
    fig_new = px.bar(new_subs, x='created_at', y='New Subscriptions', title=f"New Subscriptions ({time_grouping})")
    st.plotly_chart(fig_new, use_container_width=True)

    st.markdown("---")
    st.subheader("Revenue Projections")
    # Projection Formula: yearly + pro avg. retention month * pro plan price + team avg. retention month * team plan price
    # Note: Requires pricing variables. Adjust these hardcoded values to match your actual Stripe catalog.
    YEARLY_PRICE = 100
    PRO_MONTHLY_PRICE = 15
    TEAM_MONTHLY_PRICE = 30

    # Calculate average retention in months for canceled users
    canceled_df = df_subs[pd.notnull(df_subs['canceled_at'])].copy()
    canceled_df['retention_days'] = (canceled_df['canceled_at'] - canceled_df['created_at']).dt.days
    canceled_df['retention_months'] = canceled_df['retention_days'] / 30

    # Group by plan (assuming 'product_name' dictates plan type)
    if 'product_name' in canceled_df.columns:
        avg_retention = canceled_df.groupby('product_name')['retention_months'].mean().fillna(1).to_dict()

        pro_retention = avg_retention.get('Pro Plan', 3) # Default 3 months if no data
        team_retention = avg_retention.get('Team Plan', 6) # Default 6 months if no data

        # Current active counts by plan
        active_pro = len(df_subs[(df_subs['status'] == 'active') & (df_subs['product_name'].str.contains('Pro', na=False))])
        active_team = len(df_subs[(df_subs['status'] == 'active') & (df_subs['product_name'].str.contains('Team', na=False))])
        active_yearly = len(df_subs[(df_subs['status'] == 'active') & (df_subs['product_name'].str.contains('Yearly', na=False))])

        projected_LTV_revenue = (active_yearly * YEARLY_PRICE) + \
                                (active_pro * pro_retention * PRO_MONTHLY_PRICE) + \
                                (active_team * team_retention * TEAM_MONTHLY_PRICE)

        st.info(f"**Estimated Projected Revenue (Based on Avg Retention):** ${projected_LTV_revenue:,.2f}")
        st.caption(f"Assumes Pro Plan retains for {pro_retention:.1f} months and Team Plan for {team_retention:.1f} months on average.")
    else:
        st.warning("Cannot calculate projections: 'product_name' column missing from subscription data.")

# ==========================================
# TAB 3: MARKETING & CAC (Metric 8, 11)
# ==========================================
with tab3:
    st.subheader("Customer Acquisition & Ad Performance")

    if not df_ads.empty:
        # 1. Ad spend of the period
        period_spend = df_ads[(pd.to_datetime(df_ads['date']).dt.date >= start_date) &
                              (pd.to_datetime(df_ads['date']).dt.date <= end_date)]['spend'].sum()
        st.metric("Ad Spend (Selected Period)", f"${period_spend:,.2f}")

        # 2 & 3. New Paid Subs & CAC with 7-Day Lag
        # Prepare Ads daily data
        ads_daily = df_ads.groupby('date')['spend'].sum().reset_index()
        ads_daily['date'] = pd.to_datetime(ads_daily['date'])
        ads_daily['date_lagged'] = ads_daily['date'] + timedelta(days=7) # The Lag

        # Prepare Subs daily data
        subs_daily = df_subs[df_subs['status'] == 'active'].set_index('created_at').resample('D').size().reset_index(name='new_paid_subs')
        subs_daily.rename(columns={'created_at': 'date'}, inplace=True)

        # Merge on Lag
        merged_lag = pd.merge(subs_daily, ads_daily, left_on='date', right_on='date_lagged', how='inner')
        merged_lag['CAC'] = merged_lag['spend'] / merged_lag['new_paid_subs']
        merged_lag['CAC'] = merged_lag['CAC'].replace([np.inf, -np.inf], 0).fillna(0)

        avg_cac = merged_lag['CAC'].mean()
        st.metric("Average CAC (7-Day Lag)", f"${avg_cac:,.2f}")

        # 4. Trend Chart showing relationship b/w Ad spend & subscription (7-day lag)
        fig_cac = go.Figure()
        fig_cac.add_trace(go.Bar(x=merged_lag['date_x'], y=merged_lag['spend'], name='Ad Spend (Lagged 7 Days)', opacity=0.5))
        fig_cac.add_trace(go.Scatter(x=merged_lag['date_x'], y=merged_lag['new_paid_subs'], name='New Paid Subs', yaxis='y2', mode='lines+markers'))

        fig_cac.update_layout(
            title="Ad Spend (7-Day Lag) vs. New Subscriptions",
            yaxis=dict(title='Ad Spend ($)'),
            yaxis2=dict(title='New Paid Subs', overlaying='y', side='right'),
            barmode='group'
        )
        st.plotly_chart(fig_cac, use_container_width=True)

    else:
        st.info("Google Ads data not yet available in the data lake.")

    st.markdown("---")
    st.subheader("Trial-to-Active Conversion Rate")
    # A rough estimation: total active users / (total active + total canceled that were on trial)
    # For a perfect conversion metric, Stripe Events API tracking customer progression is needed.
    total_active_ever = len(df_subs[df_subs['status'].isin(['active', 'canceled'])]) # Excludes current trialing
    if total_active_ever > 0:
        conversion_rate = (len(df_subs[df_subs['status'] == 'active']) / total_active_ever) * 100
        st.metric("Overall Trial-to-Active Success Rate", f"{conversion_rate:.1f}%")

# ==========================================
# TAB 4: CHURN ANALYSIS
# ==========================================
with tab4:
    st.subheader("Churning Situation")

    if not canceled_df.empty:
        # 1 & 2. Average time to cancel
        # If they canceled in <= 14 days, we assume it was a trial cancellation
        canceled_df['cancel_type'] = np.where(canceled_df['retention_days'] <= 14, 'Trial Cancel', 'Active Cancel')

        avg_time_trial = canceled_df[canceled_df['cancel_type'] == 'Trial Cancel']['retention_days'].mean()
        avg_time_active = canceled_df[canceled_df['cancel_type'] == 'Active Cancel']['retention_days'].mean()

        c1, c2, c3 = st.columns(3)
        c1.metric("Avg Time to Cancel (Trial)", f"{avg_time_trial:.1f} Days")
        c2.metric("Avg Time to Cancel (Active)", f"{avg_time_active:.1f} Days")

        # 3. Cancellation %
        total_subs_ever = len(df_subs)
        cancel_percent = (len(canceled_df) / total_subs_ever) * 100 if total_subs_ever > 0 else 0
        c3.metric("Overall Cancellation Rate", f"{cancel_percent:.1f}%")

        st.markdown("---")

        # Cancellation Reasons Trend & Pie
        if 'cancel_reason' in canceled_df.columns:
            reason_col1, reason_col2 = st.columns(2)

            with reason_col1:
                st.write("**Cancellation Reasons Breakdown**")
                reason_counts = canceled_df['cancel_reason'].value_counts().reset_index()
                reason_counts.columns = ['Reason', 'Count']
                fig_reasons = px.pie(reason_counts, values='Count', names='Reason', hole=0.4)
                st.plotly_chart(fig_reasons, use_container_width=True)

            with reason_col2:
                st.write("**Cancellation Trend Over Time**")
                cancel_trend = canceled_df.set_index('canceled_at').resample(freq).size().reset_index(name='Cancellations')
                fig_cancel_trend = px.line(cancel_trend, x='canceled_at', y='Cancellations')
                st.plotly_chart(fig_cancel_trend, use_container_width=True)
        else:
            st.info("No explicit 'cancel_reason' column found in the dataset.")

    else:
        st.success("No cancellations found in the selected data range!")
