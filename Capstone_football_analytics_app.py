# Filename: football_analytics_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time


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
        df = load_data("data/FPP.csv")
        df = preprocess_data(df)
        
        # --- BEGIN Model Visualisations Code ---#


        # Create a dummy dataset
        @st.cache_data
        def create_dummy_data():
            np.random.seed(42)
            n = 300
            data = {
                "Position_Cleaned": np.random.choice(["Forward", "Midfielder", "Defender", "Goalkeeper"], size=n),
                "Goals": np.random.randint(0, 10, size=n),
                "Shots": np.random.randint(10, 50, size=n),
                "Assists": np.random.randint(0, 5, size=n),
                "xG": np.random.uniform(0, 5, size=n),
                "xA": np.random.uniform(0, 3, size=n),
                "Passes": np.random.randint(20, 100, size=n),
                "Key Passes": np.random.randint(0, 20, size=n),
                "Tackles": np.random.randint(0, 10, size=n),
                "Interceptions": np.random.randint(0, 10, size=n),
                "Clearances": np.random.randint(0, 10, size=n),
                # Goalkeeper-specific attributes
                "Saves": np.random.randint(0, 10, size=n),
                "Clean Sheets": np.random.randint(0, 5, size=n),
                "Goals Conceded": np.random.randint(0, 5, size=n),
                "Save Percentage": np.random.uniform(50, 100, size=n),
                "Passes Completed": np.random.randint(20, 80, size=n)
            }
            return pd.DataFrame(data)

        df = create_dummy_data()

        # Sidebar: Dynamic selection of player position
        st.sidebar.header("Model & Visualization Options")
        position = st.sidebar.selectbox(
            "Select Player Position",
            options=["Forward", "Midfielder", "Defender", "Goalkeeper"]
        )

        # Filter the dataset based on selected position
        df_position = df[df["Position_Cleaned"] == position]

        # Define available attributes for each position
        if position == "Forward":
            available_attrs = ["Goals", "Shots", "Assists", "xG", "xA"]
        elif position == "Midfielder":
            available_attrs = ["Passes", "Key Passes", "Assists", "xA", "xG"]
        elif position == "Defender":
            available_attrs = ["Tackles", "Interceptions", "Clearances"]
        elif position == "Goalkeeper":
            available_attrs = ["Saves", "Clean Sheets", "Goals Conceded", "Save Percentage", "Passes Completed"]

        # Sidebar: Let the user choose which attributes to visualize
        selected_attrs = st.sidebar.multiselect(
            "Select Performance Attributes",
            options=available_attrs,
            default=available_attrs[:2]
        )

        st.title(f"Player Performance Analysis for {position}s")

        # Example 1: Scatter Plot
        if len(selected_attrs) >= 2:
            st.subheader(f"Scatter Plot: {selected_attrs[0]} vs. {selected_attrs[1]}")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(data=df_position, x=selected_attrs[0], y=selected_attrs[1], ax=ax)
            ax.set_xlabel(selected_attrs[0])
            ax.set_ylabel(selected_attrs[1])
            ax.set_title(f"{selected_attrs[0]} vs. {selected_attrs[1]} for {position}s")
            fig.tight_layout()
            st.pyplot(fig)
        else:
            st.info("Please select at least two attributes to display a scatter plot.")

        # Example 2: Histograms for each selected attribute
        for attr in selected_attrs:
            st.subheader(f"Distribution of {attr} for {position}s")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.histplot(df_position[attr].dropna(), bins=20, kde=True, ax=ax)
            ax.set_xlabel(attr)
            ax.set_ylabel("Frequency")
            ax.set_title(f"Distribution of {attr} for {position}s")
            fig.tight_layout()
            st.pyplot(fig)

        # --- END MODEL VISUALISATION CODE --#
       


if __name__ == "__main__":
    main()
