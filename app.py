import streamlit as st
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


def simulate_prediction(city, bhk, size_sqft, age_years):

    base_price = 100

    if bhk == 3: base_price *= 1.8
    elif bhk == 4: base_price *= 2.5
    else: base_price *= 1.2

    # Introduce small random noise
    np.random.seed(42) # Set seed for consistent simulation
    noise = np.random.uniform(-10, 10)


    predicted_price = base_price + noise
    return predicted_price


try:

    df = pd.read_csv("india_housing_prices.csv")
    CITIES = sorted(df['City'].unique().tolist())

    if 'Mumbai' not in CITIES:
        CITIES.insert(0, "Mumbai")
except FileNotFoundError:
    # Use a mock dataframe if the file is not found, to keep the app running
    df = pd.DataFrame({'City': ['Mumbai', 'Delhi'], 'BHK': [2, 3], 'Size_in_SqFt': [1000, 1500], 'Price_in_Lakhs': [100, 150], 'BHK': [2, 3]})
    CITIES = sorted(df['City'].unique().tolist())
    st.error("Data file 'india_housing_prices.csv' not found. Using mock data for demonstration.")


st.set_page_config(
    page_title="Real Estate Investment Advisor",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
.advisor-header {
  background-color: black;
    color: white;
    padding: 20px;
    border-radius: 8px;
    text-align: center;
    margin-bottom: 20px;
}
.advisor-header h1 {
    font-size: 2.5em;
    margin: 0;
    padding-bottom: 5px;
}
.advisor-header p {
    font-size: 1.1em;
    margin: 0;
    opacity: 0.8;
}
</style>
<div class="advisor-header">
    <h1>🏠 REAL ESTATE INVESTMENT ADVISOR</h1>
    <p>Professional Property Analysis & Investment Prediction System</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["⚡ Quick Predictor", "🏘️ Property Search", "📈 Market Insights", "🙋 About & Skills"])



with tab1:
    st.header("⚡ Quick Investment Predictor")
    st.markdown("Enter property details to estimate fair market value and investment potential.")

    with st.form("investment_form"):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            city_input = st.selectbox("City", CITIES, index=CITIES.index('Mumbai') if 'Mumbai' in CITIES else 0)

        with col2:
            bhk_input = st.selectbox("BHK (Bedrooms)", options=sorted(df['BHK'].unique().tolist()) if not df.empty else [1, 2, 3, 4, 5, 6], index=2 if 3 in df['BHK'].unique() else 0)

        with col3:
            size_input = st.number_input("Size (SqFt)", min_value=300, max_value=10000, value=1200, step=50)

        with col4:
            age_input = st.number_input("Age (Years)", min_value=0, max_value=50, value=5, step=1)

        
        price_col, _ = st.columns([2, 5])
        with price_col:
            
            price_input = st.number_input("Current Asking Price (₹ Lakhs)", min_value=1.0, value=150.0, step=1.0)

        submitted = st.form_submit_button("Analyze Investment", type="primary")


    
    if submitted:
        st.subheader("Investment Analysis Results")

       
        predicted_price_lakhs = simulate_prediction(city_input, bhk_input, size_input, age_input)

        st.metric(label="Predicted Fair Market Value",
                  value=f"₹ {predicted_price_lakhs:,.2f} Lakhs",
                  delta=f"{(predicted_price_lakhs - price_input):,.2f} Lakhs"
                  )

        st.write("---")

    
        price_difference = predicted_price_lakhs - price_input

        if price_difference > (predicted_price_lakhs * 0.10): # Over 10% undervalued
            st.success(f"💰 **Strong Buy Signal!**")
            st.markdown(f"The property is estimated to be **{abs(price_difference):.2f} Lakhs** undervalued. This suggests a high potential for immediate equity gain.")

        elif price_difference > (predicted_price_lakhs * 0.02): # Over 2% undervalued
            st.info(f"👍 **Good Investment.**")
            st.markdown(f"The property is reasonably priced and offers a potential upside of **{abs(price_difference):.2f} Lakhs** on its fair value.")

        elif price_difference < -(predicted_price_lakhs * 0.10): # Over 10% overvalued
            st.error(f"⚠️ **High Risk/Overpriced!**")
            st.markdown(f"The property is estimated to be **{abs(price_difference):.2f} Lakhs** over its fair value. Exercise caution or negotiate heavily.")

        else:
            st.warning("⚖️ **Fairly Valued.**")
            st.markdown("The current asking price aligns closely with the predicted fair market value. It's a standard purchase.")

with tab2:
  st.header("🏘️ Property Search & Filter")

  st.markdown("Use powerful filters to explore properties from the dataset.")

    
  col1, col2, col3, col4 = st.columns(4)

  with col1:
      city_filter = st.selectbox("City", ["All"] + CITIES)

  with col2:
        bhk_filter = st.multiselect("BHK", sorted(df["BHK"].unique()), default=sorted(df['BHK'].unique()))

  with col3:
        min_size = st.number_input("Min Size (Sq Ft)", min_value=300, value=500)

  with col4:
        max_size = st.number_input("Max Size (Sq Ft)", min_value=300, value=3000)

  price_min = int(df['Price_in_Lakhs'].min())
  price_max = int(df['Price_in_Lakhs'].max())

  price_range = st.slider("Select Price Range (Lakhs)", price_min, price_max, (price_min, price_max))
  filtered = df.copy()

  if city_filter != "All":
        filtered = filtered[filtered["City"] == city_filter]

  filtered = filtered[
        (filtered["BHK"].isin(bhk_filter)) &
        (filtered["Size_in_SqFt"] >= min_size) &
        (filtered["Size_in_SqFt"] <= max_size) &
        (filtered["Price_in_Lakhs"] >= price_range[0]) &
        (filtered["Price_in_Lakhs"] <= price_range[1])
    ]

  st.subheader("🔎 Filtered Properties")
  st.write(f"Total Results: **{len(filtered)}**")

  st.dataframe(filtered)

   

with tab3:
    st.header("📈 Market Insights & EDA")

    st.markdown("Explore market trends using visual insights.")

    # 1. Price distribution
    st.subheader("📌 Price Distribution")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(df["Price_in_Lakhs"], kde=True, ax=ax)
    st.pyplot(fig)

    # 2. BHK vs Price Boxplot
    st.subheader("📌 BHK-wise Price Comparison")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=df, x="BHK", y="Price_in_Lakhs", ax=ax)
    st.pyplot(fig)

    # 3. Average City Price Bar Chart
    st.subheader("📌 Average Property Price by City")
    city_price = df.groupby("City")["Price_in_Lakhs"].mean().sort_values()

    fig, ax = plt.subplots(figsize=(7, 4))
    city_price.plot(kind='bar', ax=ax)
    ax.set_ylabel("Average Price (Lakhs)")
    st.pyplot(fig)

    # 4. Correlation Heatmap
    st.subheader("📌 Feature Correlation Heatmap")
    numeric_df = df.select_dtypes(include=np.number)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)
    

with tab4:
  st.header("🙋 About the Project")

  st.markdown("""
### 🏠 Real Estate Investment Advisor – Project Overview  
This project is a Machine Learning–powered real estate analysis system built to help users evaluate property prices, analyze trends, and make informed investment decisions.  
It uses data-driven insights to understand how different features such as **BHK, size, age of property, and city** influence housing prices in India.

The application is fully developed using **Streamlit**, which makes it interactive and user-friendly.  
This system allows users to:
- Predict fair market value of a property  
- Compare the asking price with estimated value  
- View market insights through visualizations  
- Explore and filter properties from the dataset  
- Understand price variations across cities  

---

### 🔧 Skills & Technologies Used in This Project

#### **1. Python Programming**
Used for data handling, analysis, and building the prediction logic.

#### **2. Data Analysis (Pandas & NumPy)**
- Cleaning the dataset  
- Handling missing values  
- Filtering, grouping, and statistical analysis  
- Feature selection  

#### **3. Exploratory Data Analysis (EDA)**
- Understanding patterns in housing prices  
- Visualizing distributions, correlations, and property trends  
- Matplotlib & Seaborn visualizations  

#### **4. Machine Learning Concepts**
- Feature engineering  
- Basic prediction logic (or ML model integration)  
- Understanding relationships between features and target variable  

#### **5. Streamlit (Web App Development)**
- Creating interactive user inputs  
- Displaying property filters  
- Integrating visual charts  
- Designing a clean and structured UI  

---

### 📌 Summary  
This project demonstrates how **data science + simple machine learning + interactive dashboards** can help users make better real estate decisions.  
It highlights your ability to work with datasets, apply EDA, build functional applications, and present insights in an easy-to-use interface.

""")

  st.success("This page describes the project and the skills involved.")
  
