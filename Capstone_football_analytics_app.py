# Filename: football_analytics_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Optional: Set up page config (title, layout, etc.)
st.set_page_config(
    page_title="Capstone Project: Forecasting Football Player Performance",
    layout="centered",
    initial_sidebar_state="expanded"
)

# LOADING... THE DATA
def load_data(file_path):
    """
    Loads data from a CSV file (or other format) into a pandas DataFrame.
    Replace or modify this function to match your data source.
    """
    df = pd.read_csv(file_path)
    return df

# CLEANING THE DATA
def preprocess_data(df):
    """
    Perform any data cleaning or feature engineering steps here.
    This might include handling missing values, creating new columns, etc.
    """
    # Example: dropping rows with too many missing values
    # df.dropna(subset=['some_important_column'], inplace=True)

    # Example: creating a new feature
    # df['Offensive_Contribution'] = df['Goals'] + df['Assists']

    # # Sidebar: Interactive Filters
    # st.sidebar.header("Filters")

    # # Filter by Player Positions (CSV has a 'Position_Cleaned' column)
    positions = st.sidebar.multiselect(
        "Select Player Position",
        options=df["Position_Cleaned"].unique(),
        default=df["Position_Cleaned"].unique()
    )
    # Additional filter: e.g., by Age if available
    ages = st.sidebar.multiselect(
        "Select Player Age",
        options=df["Age"].unique(),
        default=df["Age"].unique()
    )

    # Apply filters
    df = df[(df["Position_Cleaned"].isin(positions)) & (df["Age"].isin(ages))]
    
    return df

def exploratory_analysis(df):
    """
    Conduct any exploratory analysis, such as computing descriptive stats
    or generating figures. Return any figures or stats you want to display.
    """
    # Example descriptive statistics
    stats = df.describe()

    # @TODO:
    # Display Key Metrics
    st.subheader("Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_matches = df["MP"].sum() if "MP" in df.columns else "N/A"
        st.metric("Total Matches Played", total_matches)
    with col2:
        total_goals = df["Goals"].sum() if "Goals" in df.columns else "N/A"
        st.metric("Total Goals Scored", total_goals)
    with col3:
        total_assits = df["Assists"].sum() if "Assists" in df.columns else "N/A"
        st.metric("Total Assists", total_assits)
    with col4:
        avg_goals = round(df["G/Sh"].mean(), 2) if "G/Sh" in df.columns else "N/A"
        st.metric("Goals per Shots", avg_goals)
    
    # Example Seaborn figure
    fig, ax = plt.subplots(figsize=(6,4))
    sns.histplot(data=df, x='Goals', kde=True, ax=ax)
    ax.set_title("Distribution of Goals")
    
    return stats, fig

def model_training(df):
    """
    If your notebook includes a predictive model, set it up here.
    This function could train the model and return the trained model or predictions.
    """
    # Example placeholder: a dictionary with "predictions"
    # You would replace this with your actual model training code.
    results = {
        "example_prediction": [0.5, 1.0, 1.5]  # Dummy data
    }
    return results

# MAIN SECTION OF APP
def main():
    """
    Main function that orchestrates the Streamlit application.
    """
    st.title("Football Player Insights: A Data-Driven Performance Analysis")

    # Sidebar for user inputs or navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose a section:",
        ["About", "Data Upload", "Data Exploration", "Model Training"]
    )

    # Section 1: About
    if app_mode == "About":
        st.header("1. About This App")
        st.write("""
        Explore and visualize football player stats and attributes to identify top performers using data-driven analytics. 
        Adjust filters to match the dataset, perform EDA, and refine models for deeper insights. 
        For more details on Streamlit, visit [Streamlit Docs](https://docs.streamlit.io/).
        """)   

    # Section 2: Data Upload
    elif app_mode == "Data Upload":
        st.header("2. Data Upload")
        uploaded_file = st.file_uploader("Upload your dataset (CSV file):", type=["csv"])
        
        if uploaded_file is not None:
            df = load_data(uploaded_file)
            st.success("Data loaded successfully!")
            st.write("Preview of your data:")
            st.dataframe(df.head())
        else:
            st.info("Please upload a CSV file to proceed.")
    
    # Section 3: Data Exploration
    elif app_mode == "Data Exploration":
        st.header("3. Data Exploration")
        
        # For demonstration, we assume you have a default CSV to load
        # Replace 'data/football_data.csv' with a newly cleaned dataset 'actual path' or file
        df = load_data("data/cleaned_dataset.csv")
        df = preprocess_data(df)
        
        # # Display some EDA results
        stats, fig = exploratory_analysis(df)
        st.subheader("Descriptive Statistics")
        st.write(stats)
        
        st.subheader("Goals Distribution")
        st.pyplot(fig)

    # /**** VISUALISATIONS **** /

        # Display Key Metrics
        st.subheader("Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_matches = df["MP"].sum() if "MP" in df.columns else "N/A"
            st.metric("Total Matches Played", total_matches)
        with col2:
            total_goals = df["Goals"].sum() if "Goals" in df.columns else "N/A"
            st.metric("Total Goals", total_goals)
        with col3:
            total_assists = df["Assists"].sum() if "Assists" in df.columns else "N/A"
            st.metric("Total Assists", total_assists)
        with col4:
            avg_pts = round(df["Pts/MP"].mean(), 2) if "Pts/MP" in df.columns else "N/A"
            st.metric("Average Points per Match", avg_pts)

        # Data Overview
        st.subheader("Data Overview")
        st.dataframe(df.head())

        # Visualization 1: Scatter Plot of Goals vs. Assists by Age
        st.subheader("Goals vs. Assists by Forward Positions")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df, x="Goals", y="Assists", hue="Position_Cleaned", ax=ax)
        ax.set_xlabel("Goals")
        ax.set_ylabel("Assists")
        ax.set_title("Scatter Plot: Goals vs. Assists")
        st.pyplot(fig)

        # Visualization 2: Distribution of Expected Goals (xG)
        st.subheader("Distribution of Expected Goals (xG)")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        if "xG" in df.columns:
            sns.histplot(df["xG"].dropna(), bins=20, kde=True, ax=ax2)
            ax2.set_xlabel("Expected Goals (xG)")
            ax2.set_ylabel("Frequency")
            ax2.set_title("Distribution of Expected Goals (xG)")
            st.pyplot(fig2)
        else:
            st.info("Column 'xG' not found in the dataset.")
            
    # /**** END OF VISUALISATIONS ***/
    
    # Section 4: Model Training
    else: 
        app_mode == "Model Training"
        st.header("4. Model Training and Results")
        
        # Load and preprocess data
        df = load_data("data/cleaned_dataset.csv")
        df = preprocess_data(df)
        
        # Train or load your model
        results = model_training(df)
        
        st.write("Example model output or predictions:")
        st.write(results)
        
        # You can add interactive elements, metrics, or other visuals here
        # e.g., st.metric(label="Mean Absolute Error", value="0.123")


if __name__ == "__main__":
    main()
