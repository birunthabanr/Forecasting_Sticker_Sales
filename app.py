import pandas as pd
import joblib
import streamlit as st
from datetime import datetime, date
import numpy as np

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("stacking_model.joblib")

def feature_engineering(df):
    """Apply feature engineering to the input data"""
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['quarter'] = df['date'].dt.quarter.astype('float64')
    df['month'] = df['date'].dt.month
    df['week'] = df['date'].dt.isocalendar().week
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype('float64')
    df['hour'] = df['date'].dt.hour.astype('float64')
    df['minute'] = df['date'].dt.minute.astype('float64')
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['sine_day'] = np.sin(2 * np.pi * df['day'] / 365)
    df['cos_day'] = np.cos(2 * np.pi * df['day'] / 365)
    df['sine_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
    df['sine_year'] = np.sin(2 * np.pi * df['year']/7)
    df['cos_year'] = np.cos(2 * np.pi * df['year']/7)
    df['group'] = (df['year'] - 2010) * 48 + df['month'] * 4 + df['day'] // 7
    return df

stacking_model = load_model()

# Page configuration
st.set_page_config(
    page_title="Sticker Sales Forecasting",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main title and description
st.title("🏪 Sticker Sales Forecasting Dashboard")
st.markdown("Predict sales using advanced machine learning models - Just enter the basic information!")

# Sidebar for inputs
st.sidebar.header("📋 Required Input Parameters")

with st.sidebar:
    st.subheader("Basic Information")
    
    # Basic inputs only
    # id_ = st.number_input("Transaction ID", value=1, min_value=1, step=1)
    
    # Date input with datetime picker
    selected_date = st.date_input(
        "Select Date",
        value=date.today(),
        min_value=date(2020, 1, 1),
        max_value=date(2030, 12, 31)
    )
    
    # Time input
    selected_time = st.time_input("Select Time", value=datetime.now().time())
    
    # Combine date and time
    datetime_input = datetime.combine(selected_date, selected_time)
    
    country = st.selectbox(
        "Country", 
        options=["Canada", "Finland", "Italy", "Kenya", "Norway","Singapore"],
        index=0
    )
    
    store = st.selectbox(
        "Store", 
        options=["Discount Stickers", "Stickers for Less", "Premium Sticker Mart"],
        index=0
    )
    
    product = st.selectbox(
        "Product", 
        options=["Holographics Goose", "Kaggle", "Kaggle Tiers", "Kerneler", "Kerneler Dark Mode"],
        index=0
    )

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Input Summary & Computed Features")
    
    # Create the input dataframe
    input_df = pd.DataFrame({
        # 'id': [id_],
        'date': [datetime_input],
        'country': [country],
        'store': [store],
        'product': [product]
    })
    
    # Apply feature engineering
    processed_df = feature_engineering(input_df.copy())
    
    # Display basic inputs
    with st.expander("📋 Basic Inputs", expanded=True):
        # st.write(f"**Transaction ID:** {id_}")
        st.write(f"**Date & Time:** {datetime_input.strftime('%Y-%m-%d %H:%M:%S')}")
        st.write(f"**Country:** {country}")
        st.write(f"**Store:** {store}")
        st.write(f"**Product:** {product}")
    
    # Display computed temporal features
    with st.expander("📅 Computed Temporal Features", expanded=True):
        col_a, col_b = st.columns(2)
        with col_a:
            st.write(f"**Year:** {processed_df['year'].iloc[0]}")
            st.write(f"**Quarter:** {processed_df['quarter'].iloc[0]}")
            st.write(f"**Month:** {processed_df['month'].iloc[0]}")
            st.write(f"**Week:** {processed_df['week'].iloc[0]}")
            st.write(f"**Day:** {processed_df['day'].iloc[0]}")
        with col_b:
            st.write(f"**Day of Week:** {processed_df['day_of_week'].iloc[0]} ({'Weekend' if processed_df['is_weekend'].iloc[0] else 'Weekday'})")
            st.write(f"**Hour:** {processed_df['hour'].iloc[0]}")
            st.write(f"**Minute:** {processed_df['minute'].iloc[0]}")
            st.write(f"**Week of Year:** {processed_df['week_of_year'].iloc[0]}")
            st.write(f"**Group:** {processed_df['group'].iloc[0]}")
    
    # Display cyclic features
    with st.expander("🔄 Computed Cyclic Features"):
        col_c, col_d = st.columns(2)
        with col_c:
            st.write(f"**Sine Day:** {processed_df['sine_day'].iloc[0]:.4f}")
            st.write(f"**Cosine Day:** {processed_df['cos_day'].iloc[0]:.4f}")
            st.write(f"**Sine Month:** {processed_df['sine_month'].iloc[0]:.4f}")
        with col_d:
            st.write(f"**Cosine Month:** {processed_df['cos_month'].iloc[0]:.4f}")
            st.write(f"**Sine Year:** {processed_df['sine_year'].iloc[0]:.4f}")
            st.write(f"**Cosine Year:** {processed_df['cos_year'].iloc[0]:.4f}")

with col2:
    st.subheader("🔮 Prediction")
    
    # Prediction button
    if st.button("🚀 Generate Prediction", type="primary", use_container_width=True):
        with st.spinner("Processing features and making prediction..."):
            try:
                # Prepare data for prediction (select only the required columns for the model)
                model_features = [
                     'country', 'store', 'product', 'year', 'quarter', 'month', 
                    'week', 'day', 'day_of_week', 'week_of_year', 'hour', 'minute', 
                    'is_weekend', 'sine_day', 'cos_day', 'sine_month', 'cos_month', 
                    'sine_year', 'cos_year', 'group'
                ]
                
                # Create prediction dataframe with correct data types
                prediction_data = processed_df[model_features].astype({
                    # 'id': 'int64',
                    'year': 'int32',
                    'quarter': 'float64',
                    'month': 'int32',
                    'week': 'UInt32',
                    'day': 'int32',
                    'day_of_week': 'int32',
                    'week_of_year': 'float64',
                    'hour': 'float64',
                    'minute': 'float64',
                    'is_weekend': 'int64',
                    'sine_day': 'float64',
                    'cos_day': 'float64',
                    'sine_month': 'float64',
                    'cos_month': 'float64',
                    'sine_year': 'float64',
                    'cos_year': 'float64',
                    'group': 'int32'
                })
                
                # Convert categorical columns to match training data format
                # You might need to adjust these based on how your model was trained
                prediction_data['country'] = prediction_data['country'].astype('category').cat.codes
                prediction_data['store'] = prediction_data['store'].astype('category').cat.codes  
                prediction_data['product'] = prediction_data['product'].astype('category').cat.codes
                
                prediction = stacking_model.predict(prediction_data)
                
                # Display prediction
                st.success("Prediction Complete!")
                st.metric(
                    label="Predicted Sales", 
                    value=f"{np.expm1(prediction[0]):,.2f} units",
                    delta=None
                )
                
                # Additional insights
                with st.expander("📈 Prediction Details"):
                    st.write(f"**Model Type:** Stacking Ensemble")
                    st.write(f"**Prediction Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    st.write(f"**Features Used:** {len(model_features)} variables")
                    st.write(f"**Input Date:** {datetime_input.strftime('%A, %B %d, %Y at %H:%M')}")
                    
                    # Show feature importance or contribution (if available)
                    st.write("**Key Features:**")
                    st.write("- Temporal patterns (day, month, year cycles)")
                    st.write("- Store and product identifiers")
                    st.write("- Weekend/weekday classification")
                    
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.write("**Debug Info:**")
                st.write("Please check if your model expects the same feature format.")
    
    # Show current data summary
    st.info(f"📅 **Current Selection:**\n{datetime_input.strftime('%A, %B %d, %Y')}\n{datetime_input.strftime('%H:%M:%S')}")

# Feature engineering explanation
with st.expander("ℹ️ How Features Are Computed"):
    st.markdown("""
    **Temporal Features:**
    - Year, quarter, month, week, day extracted from date
    - Day of week (0=Monday, 6=Sunday)
    - Hour and minute from time
    - Weekend flag (Saturday=5, Sunday=6)
    
    **Cyclic Features:**
    - Sine/cosine transformations capture cyclical patterns
    - Day cycle: sin(2π × day / 365), cos(2π × day / 365)  
    - Month cycle: sin(2π × month / 12), cos(2π × month / 12)
    - Year cycle: sin(2π × year / 7), cos(2π × year / 7)
    
    **Group Feature:**
    - Calculated as: (year - 2010) × 48 + month × 4 + day ÷ 7
    - Creates unique time-based groupings for modeling
    """)

# Add footer
st.markdown("---")
st.markdown("*Built with Streamlit • Automated Feature Engineering • Powered by Machine Learning*")
