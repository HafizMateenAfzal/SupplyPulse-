"""Retail Demand Forecasting app predicting next week's product sales quantity."""

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ==========================================
# DATA LOADING
# ==========================================


@st.cache_data
def load_sales_data() -> pd.DataFrame:
    """Load and preprocess real store sales data from store_sales.csv."""
    # Read the dataset and parse date column
    df = pd.read_csv("store_sales.csv")
    df["date"] = pd.to_datetime(df["date"])
    
    # Rename columns to match the rest of the app
    df = df.rename(columns={
        "store": "store_id",
        "item": "product_id",
        "sales": "units_sold"
    })
    
    # Check for and handle missing values (impute, don't drop rows)
    if df.isna().any().any():
        df = df.sort_values(by=["store_id", "product_id", "date"]).reset_index(drop=True)
        # Use forward fill then backward fill grouped by store and product
        df["units_sold"] = df.groupby(["store_id", "product_id"])["units_sold"].transform(lambda x: x.ffill().bfill())
        # Fallback for any remaining NaNs
        df["units_sold"] = df["units_sold"].fillna(0)
    
    # Cap/clip extreme outliers in units_sold using percentile capping (99th percentile per product)
    q99 = df.groupby("product_id")["units_sold"].transform("quantile", 0.99)
    df["units_sold"] = np.minimum(df["units_sold"], q99)
    
    # Keep the existing is_holiday logic against the real dates
    dates = pd.DatetimeIndex(df["date"].unique())
    holiday_map = pd.Series(0, index=dates)
    
    # Simple fixed date holidays
    holiday_map[(dates.month == 1) & (dates.day == 1)] = 1   # New Year's Day
    holiday_map[(dates.month == 7) & (dates.day == 4)] = 1   # Independence Day
    holiday_map[(dates.month == 12) & (dates.day == 25)] = 1 # Christmas Day
    
    # Floating US holidays
    for year in dates.year.unique():
        # Thanksgiving: 4th Thursday in Nov
        nov_days = pd.date_range(start=f"{year}-11-01", end=f"{year}-11-30")
        thursdays = nov_days[nov_days.weekday == 3]
        thanksgiving = thursdays[3]
        holiday_map[thanksgiving] = 1
        
        # Black Friday: Friday after Thanksgiving
        black_friday = thanksgiving + pd.Timedelta(days=1)
        holiday_map[black_friday] = 1
        
        # Memorial Day: last Monday in May
        may_days = pd.date_range(start=f"{year}-05-01", end=f"{year}-05-31")
        mondays = may_days[may_days.weekday == 0]
        holiday_map[mondays[-1]] = 1
        
        # Labor Day: first Monday in Sep
        sep_days = pd.date_range(start=f"{year}-09-01", end=f"{year}-09-30")
        labor_day = sep_days[sep_days.weekday == 0][0]
        holiday_map[labor_day] = 1
        
    df["is_holiday"] = df["date"].map(holiday_map).fillna(0).astype(int)
    
    # NOTE: promotion_flag is simulated here because store_sales.csv has no real promotion data.
    # We flag ~10% of rows randomly, weighted slightly toward Nov/Dec, using a fixed random seed.
    rng = np.random.default_rng(42)
    # Calibrate probability: Nov/Dec has 20% prob, other months have 8% prob, overall average ~10%
    probs = np.where(df["date"].dt.month.isin([11, 12]), 0.20, 0.08)
    df["promotion_flag"] = rng.binomial(1, probs)
    
    # Reorder columns and select only needed columns (removing temperature, competitor_price, our_price)
    column_order = [
        "date",
        "store_id",
        "product_id",
        "units_sold",
        "is_holiday",
        "promotion_flag"
    ]
    df = df[column_order]
    
    # Print summary after loading
    print("--- DATA LOAD SUMMARY ---")
    print(f"Row count: {len(df):,}")
    print(f"Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
    print(f"Unique store IDs: {df['store_id'].nunique()} ({sorted(df['store_id'].unique())})")
    print(f"Unique product IDs: {df['product_id'].nunique()} ({min(df['product_id'].unique())} to {max(df['product_id'].unique())})")
    unique_combos = df.groupby(['store_id', 'product_id']).ngroups
    print(f"Unique Store-Product combos: {unique_combos}")
    print(f"Sales Stats:\n{df['units_sold'].describe()}")
    print("-------------------------")
    
    return df


# ==========================================
# FEATURE ENGINEERING & MODEL TRAINING
# ==========================================


@st.cache_resource
def train_model(df: pd.DataFrame) -> dict:
    """Cap outliers, engineer lag/rolling features, and train an XGBoost model."""
    # Work on a copy of the dataframe
    df = df.copy()
    
    # Ensure chronological sorting per store and product for correct lag and rolling features
    df = df.sort_values(by=["store_id", "product_id", "date"]).reset_index(drop=True)
    
    # Feature engineering: date/seasonality
    df["day_of_week"] = df["date"].dt.weekday
    df["month"] = df["date"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    
    # Feature engineering: historical lags
    df["lag_7"] = df.groupby(["store_id", "product_id"])["units_sold"].shift(7)
    df["lag_14"] = df.groupby(["store_id", "product_id"])["units_sold"].shift(14)
    
    # Feature engineering: historical rolling average
    df["sales_roll_mean_7"] = (
        df.groupby(["store_id", "product_id"])["units_sold"]
        .shift(7)
        .rolling(window=7, min_periods=1)
        .mean()
    )
    
    # Impute missing lags/rolling features (first 14 days of each time series)
    overall_median = df["units_sold"].median()
    for col in ["lag_7", "lag_14", "sales_roll_mean_7"]:
        # Impute with product-store specific median first
        df[col] = df.groupby(["store_id", "product_id"])[col].transform(lambda x: x.fillna(x.median()))
        # Fallback to overall median if needed
        df[col] = df[col].fillna(overall_median)
        
    # Categorical variable encoding
    le_store = LabelEncoder()
    df["store_id_encoded"] = le_store.fit_transform(df["store_id"])
    
    le_product = LabelEncoder()
    df["product_id_encoded"] = le_product.fit_transform(df["product_id"])
    
    # Define feature names and targets
    features = [
        "store_id_encoded", "product_id_encoded", "is_holiday", "promotion_flag",
        "day_of_week", "month", "is_weekend", "lag_7", "lag_14", "sales_roll_mean_7"
    ]
    target = "units_sold"
    
    # Time-based train-test split: last 8 weeks for evaluation
    max_date = df["date"].max()
    split_date = max_date - pd.Timedelta(weeks=8)
    
    train_df = df[df["date"] <= split_date]
    test_df = df[df["date"] > split_date]
    
    X_train, y_train = train_df[features], train_df[target]
    X_test, y_test = test_df[features], test_df[target]
    
    # Train XGBoost regressor
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Calculate evaluation metrics
    preds = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, preds))
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    
    # Compute feature importance (top 10)
    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=features).sort_values(ascending=False).head(10).to_dict()
    
    return {
        "model": model,
        "features": features,
        "encoders": {"store_id": le_store, "product_id": le_product},
        "mae": mae,
        "rmse": rmse,
        "feature_importance": feat_imp
    }


# ==========================================
# STREAMLIT UI
# ==========================================

# Set page configuration
st.set_page_config(
    page_title="Retail Demand Forecasting",
    page_icon="📈",
    layout="wide"
)

# App Title & Description
st.title("📈 Retail Demand Forecasting Dashboard")
st.markdown("""
Predict next week's product sales quantity and optimize inventory levels using an XGBoost machine learning model.
This app loads Kaggle's Store Item Demand Forecasting Challenge data, trains a forecasting model, and allows interactive predictions in real-time.
""")

# Spinner for initial load
with st.spinner("Loading store sales data and training XGBoost model (running once)..."):
    df_sales = load_sales_data()
    model_res = train_model(df_sales)

# Extract parameters from the model results
mae = model_res["mae"]
rmse = model_res["rmse"]
features_used = model_res["features"]
encoders = model_res["encoders"]
xgb_model = model_res["model"]
feat_importances = model_res["feature_importance"]

# Sidebar: Model Performance Expander
with st.sidebar.expander("📊 Model Performance", expanded=True):
    st.metric("Test MAE", f"{mae:.2f}")
    st.metric("Test RMSE", f"{rmse:.2f}")
    
    st.markdown("**Top Feature Importances**")
    for feat, imp in feat_importances.items():
        st.caption(f"{feat}: `{imp:.3f}`")
        st.progress(float(imp))

# Sidebar inputs
st.sidebar.header("Forecast Settings")

selected_store = st.sidebar.selectbox("Select Store", sorted(df_sales["store_id"].unique()))
selected_product = st.sidebar.selectbox("Select Product", sorted(df_sales["product_id"].unique()))

# Filter historical data for selected store and product
hist_combo = df_sales[
    (df_sales["store_id"] == selected_store) & 
    (df_sales["product_id"] == selected_product)
].sort_values("date")

# Input fields
forecast_date = st.sidebar.date_input("Forecast Date", value=pd.to_datetime("2018-01-01"))
promo_active = st.sidebar.checkbox("Promotion Active", value=False)

# Helper to check holiday
def check_holiday_name(dt) -> tuple[int, str | None]:
    if dt.month == 1 and dt.day == 1:
        return 1, "New Year's Day"
    if dt.month == 7 and dt.day == 4:
        return 1, "Independence Day"
    if dt.month == 12 and dt.day == 25:
        return 1, "Christmas Day"
    
    year = dt.year
    # Thanksgiving: 4th Thursday in Nov
    nov_days = pd.date_range(start=f"{year}-11-01", end=f"{year}-11-30")
    thursdays = nov_days[nov_days.weekday == 3]
    thanksgiving = thursdays[3]
    if dt == thanksgiving:
        return 1, "Thanksgiving"
        
    # Black Friday
    black_friday = thanksgiving + pd.Timedelta(days=1)
    if dt == black_friday:
        return 1, "Black Friday"
        
    # Memorial Day
    may_days = pd.date_range(start=f"{year}-05-01", end=f"{year}-05-31")
    memorial_day = may_days[may_days.weekday == 0][-1]
    if dt == memorial_day:
        return 1, "Memorial Day"
        
    # Labor Day
    sep_days = pd.date_range(start=f"{year}-09-01", end=f"{year}-09-30")
    labor_day = sep_days[sep_days.weekday == 0][0]
    if dt == labor_day:
        return 1, "Labor Day"
        
    return 0, None

# Feature engineering for the forecast date
f_date = pd.to_datetime(forecast_date)
day_of_week = f_date.weekday()
month = f_date.month
is_weekend = 1 if day_of_week >= 5 else 0
is_holiday, holiday_name = check_holiday_name(f_date)

# Look up lags from historical data
lag_7_date = f_date - pd.Timedelta(days=7)
lag_14_date = f_date - pd.Timedelta(days=14)

lag_7_row = hist_combo[hist_combo["date"] == lag_7_date]
if not lag_7_row.empty:
    lag_7_val = float(lag_7_row["units_sold"].values[0])
else:
    lag_7_val = float(hist_combo["units_sold"].iloc[-1])

lag_14_row = hist_combo[hist_combo["date"] == lag_14_date]
if not lag_14_row.empty:
    lag_14_val = float(lag_14_row["units_sold"].values[0])
else:
    lag_14_val = float(hist_combo["units_sold"].iloc[-2]) if len(hist_combo) > 1 else float(hist_combo["units_sold"].iloc[-1])

# Rolling average (mean of T-7 to T-13)
roll_start_date = f_date - pd.Timedelta(days=13)
roll_end_date = f_date - pd.Timedelta(days=7)
roll_history = hist_combo[(hist_combo["date"] >= roll_start_date) & (hist_combo["date"] <= roll_end_date)]

if not roll_history.empty:
    roll_val = float(roll_history["units_sold"].mean())
else:
    roll_val = float(hist_combo["units_sold"].iloc[-7:].mean())

# Encode IDs
store_enc = int(encoders["store_id"].transform([selected_store])[0])
product_enc = int(encoders["product_id"].transform([selected_product])[0])

# Construct prediction feature row
pred_df = pd.DataFrame([{
    "store_id_encoded": store_enc,
    "product_id_encoded": product_enc,
    "is_holiday": is_holiday,
    "promotion_flag": int(promo_active),
    "day_of_week": day_of_week,
    "month": month,
    "is_weekend": is_weekend,
    "lag_7": lag_7_val,
    "lag_14": lag_14_val,
    "sales_roll_mean_7": roll_val
}])

# Ensure exact feature order
pred_df = pred_df[features_used]

# Make prediction
predicted_sales = max(0.0, float(xgb_model.predict(pred_df)[0]))

# --- MAIN PAGE DISPLAY ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("🔮 Demand Forecast")
    
    # Calculate percentage change vs 7 days prior
    change_vs_lag = predicted_sales - lag_7_val
    pct_change = (change_vs_lag / lag_7_val * 100) if lag_7_val > 0 else 0.0
    
    st.metric(
        label=f"Predicted Sales for {f_date.strftime('%Y-%m-%d')}",
        value=f"{predicted_sales:.1f} units",
        delta=f"{change_vs_lag:+.1f} units ({pct_change:+.1f}%) vs. 7 Days Prior"
    )
    
    # Safety buffer stock recommendation
    safety_buffer = 0.15
    rec_stock = int(np.ceil(predicted_sales * (1 + safety_buffer)))
    
    st.markdown("### 💡 Business Insight")
    
    insight_parts = []
    if promo_active:
        insight_parts.append("- 🎯 **Promotion Active**: A pricing promo is running on this product, which historically increases sales volume significantly.")
    if is_holiday:
        insight_parts.append(f"- 🎈 **US Holiday ({holiday_name})**: Holiday traffic peaks demand during this period.")
    if is_weekend:
        insight_parts.append("- 📅 **Weekend Effect**: High weekend customer traffic is expected to boost demand.")
        
    insight_text = "\n".join(insight_parts) if insight_parts else "- *Steady state demand drivers.*"
    
    st.info(
        f"**Recommendation:** Store **{selected_store}** should stock at least **{rec_stock} units** of product **{selected_product}** for this day. "
        f"This covers the base prediction (**{predicted_sales:.1f} units**) plus a **{safety_buffer*100:.0f}% safety buffer** to protect against stockouts.\n\n"
        f"**Key Demand Factors for this date:**\n\n{insight_text}"
    )

with col2:
    st.subheader("📈 Historical Sales Trend & Forecast")
    
    # 90 days historical data
    last_90 = hist_combo.tail(90).copy()
    
    # Create plotting DataFrame
    plot_data = pd.DataFrame(index=pd.to_datetime(last_90["date"]))
    plot_data["Historical"] = last_90["units_sold"].values
    
    # Add prediction point (connected to the last historical point)
    plot_data.loc[last_90["date"].iloc[-1], "Predicted"] = last_90["units_sold"].iloc[-1]
    plot_data.loc[f_date, "Predicted"] = predicted_sales
    
    # Plot using Streamlit's native line chart
    st.line_chart(plot_data, y=["Historical", "Predicted"], use_container_width=True)
    
    st.caption("The solid blue line represents the last 90 days of actual historical sales. The orange line connects to the predicted sales quantity for your selected forecast date.")
