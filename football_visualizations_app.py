# Filename: football_visualizations_app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure the page
st.set_page_config(page_title="Football Analytics Visualizations Dashboard", layout="wide")

# Cache the data loading for faster performance
@st.cache_data
def load_data():
    # Replace the file path with the correct location if necessary
    df = pd.read_csv("2022-2023_Football_Player_Stats.csv")
    return df

# Load the data
df = load_data()

# Sidebar: Interactive Filters
st.sidebar.header("Filters")
# Filter by Country (CSV has a 'Nation' column)
countries = st.sidebar.multiselect(
    "Select Nation",
    options=df["Nation"].unique(),
    default=df["Nation"].unique()
)
# Additional filter: e.g., by Squad if available
squads = st.sidebar.multiselect(
    "Select Squad",
    options=df["Squad"].unique(),
    default=df["Squad"].unique()
)

# Apply filters
df_filtered = df[(df["Nation"].isin(countries)) & (df["Squad"].isin(squads))]

# Main Title
st.title("Football Analytics Capstone Dashboard")

# Display Key Metrics
st.subheader("Key Metrics")
col1, col2, col3, col4 = st.columns(4)
with col1:
    total_matches = df_filtered["MP"].sum() if "MP" in df_filtered.columns else "N/A"
    st.metric("Total Matches Played", total_matches)
with col2:
    total_goals_for = df_filtered["GF"].sum() if "GF" in df_filtered.columns else "N/A"
    st.metric("Total Goals For", total_goals_for)
with col3:
    total_goals_against = df_filtered["GA"].sum() if "GA" in df_filtered.columns else "N/A"
    st.metric("Total Goals Against", total_goals_against)
with col4:
    avg_pts = round(df_filtered["Pts/MP"].mean(), 2) if "Pts/MP" in df_filtered.columns else "N/A"
    st.metric("Average Points per Match", avg_pts)

# Data Overview
st.subheader("Data Overview")
st.dataframe(df_filtered.head())

# Visualization 1: Scatter Plot of Goals For vs. Goals Against by Squad
st.subheader("Goals For vs. Goals Against by Squad")
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=df_filtered, x="GF", y="GA", hue="Squad", ax=ax)
ax.set_xlabel("Goals For")
ax.set_ylabel("Goals Against")
ax.set_title("Scatter Plot: Goals For vs. Goals Against")
st.pyplot(fig)

# Visualization 2: Distribution of Expected Goals (xG)
st.subheader("Distribution of Expected Goals (xG)")
fig2, ax2 = plt.subplots(figsize=(10, 6))
if "xG" in df_filtered.columns:
    sns.histplot(df_filtered["xG"].dropna(), bins=20, kde=True, ax=ax2)
    ax2.set_xlabel("Expected Goals (xG)")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Distribution of Expected Goals (xG)")
    st.pyplot(fig2)
else:
    st.info("Column 'xG' not found in the dataset.")

# Visualization 3: Correlation Heatmap
st.subheader("Correlation Heatmap")
fig3, ax3 = plt.subplots(figsize=(12, 8))
# Compute the correlation matrix for numeric columns only
corr = df_filtered.select_dtypes(include="number").corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax3)
ax3.set_title("Correlation Heatmap")
st.pyplot(fig3)

# Additional interactive element: let the user select a metric to view its distribution
st.sidebar.header("Additional Analysis")
metric_options = [col for col in df_filtered.columns if df_filtered[col].dtype in ['float64', 'int64']]
selected_metric = st.sidebar.selectbox("Select a Metric for Distribution", options=metric_options)
if selected_metric:
    st.subheader(f"Distribution of {selected_metric}")
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    sns.histplot(df_filtered[selected_metric].dropna(), bins=20, kde=True, ax=ax4)
    ax4.set_xlabel(selected_metric)
    ax4.set_ylabel("Frequency")
    ax4.set_title(f"Distribution of {selected_metric}")
    st.pyplot(fig4)
