import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load dataset
df = pd.read_excel("synthetic_athlete_dataset.xlsx")

# List of ability columns
abilities = ["Speed", "Endurance", "Strength", "Agility", "ReactionTime"]

# Standardize abilities
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[abilities] = scaler.fit_transform(df[abilities])

# Sidebar page selection
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Raw Data"])

# Sidebar filters (common)
st.sidebar.header("Filter Athletes")
gender_options = ["All"] + df_scaled["Gender"].unique().tolist()
selected_gender = st.sidebar.selectbox("Select Gender", gender_options)

sport_options = ["All"] + df_scaled["Sport"].unique().tolist()
selected_sport = st.sidebar.selectbox("Select Sport", sport_options)

min_age, max_age = int(df_scaled["Age"].min()), int(df_scaled["Age"].max())
selected_age = st.sidebar.slider("Select Age Range", min_age, max_age, (min_age, max_age))

# Apply filters
filtered_df = df_scaled.copy()
if selected_gender != "All":
    filtered_df = filtered_df[filtered_df["Gender"] == selected_gender]
if selected_sport != "All":
    filtered_df = filtered_df[filtered_df["Sport"] == selected_sport]
filtered_df = filtered_df[(filtered_df["Age"] >= selected_age[0]) & (filtered_df["Age"] <= selected_age[1])]

# ---------------------------
# Page 1: Dashboard (plots)
# ---------------------------
if page == "Dashboard":
    st.title("Athlete Abilities Dashboard (Standardized)")

    # ---------------------------
    # 1. Bar plot
    # ---------------------------
    avg_values = filtered_df[abilities].mean()
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in avg_values]
    fig_bar, ax_bar = plt.subplots(figsize=(14,6))
    sns.barplot(x=avg_values.index, y=avg_values.values, palette=colors, ax=ax_bar)
    ax_bar.axhline(0, color="black", linestyle="--")
    ax_bar.set_ylabel("Z-score")
    fig_bar.tight_layout()
    st.pyplot(fig_bar)
    st.write("**Bar Plot:** Average abilities. Green = above avg, Red = below avg.")

    # ---------------------------
    # 2. Box plot
    # ---------------------------
    melted = filtered_df.melt(id_vars=["AthleteID"], value_vars=abilities, var_name="Ability", value_name="Z-Score")
    fig_box, ax_box = plt.subplots(figsize=(14,6))
    sns.boxplot(x="Ability", y="Z-Score", data=melted, palette="coolwarm", ax=ax_box)
    ax_box.axhline(0, color="black", linestyle="--")
    fig_box.tight_layout()
    st.pyplot(fig_box)
    st.write("**Box Plot:** Distribution of abilities, 0 = overall avg.")

    # ---------------------------
    # 3. Pie chart
    # ---------------------------
    counts = filtered_df["Gender"].value_counts()
    fig_pie, ax_pie = plt.subplots(figsize=(10,8))
    ax_pie.pie(counts, labels=counts.index, autopct="%1.1f%%", colors=sns.color_palette("Set2"))
    fig_pie.tight_layout()
    st.pyplot(fig_pie)
    st.write("**Pie Chart:** Gender distribution.")

    # ---------------------------
    # 4. Heatmap
    # ---------------------------
    corr = filtered_df[abilities].corr()
    fig_corr, ax_corr = plt.subplots(figsize=(14,6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax_corr)
    fig_corr.tight_layout()
    st.pyplot(fig_corr)
    st.write("**Heatmap:** Correlation between abilities.")

    # ---------------------------
    # 5. Radar chart
    # ---------------------------
    avg_values_radar = filtered_df[abilities].mean().values
    num_vars = len(abilities)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    avg_values_loop = np.concatenate((avg_values_radar, [avg_values_radar[0]]))
    angles_loop = angles + angles[:1]

    fig_radar, ax_radar = plt.subplots(figsize=(12,12), subplot_kw=dict(polar=True))
    ax_radar.plot(angles_loop, avg_values_loop, color="blue", linewidth=2)
    ax_radar.fill(angles_loop, avg_values_loop, color="skyblue", alpha=0.25)
    ax_radar.set_xticks(angles)
    ax_radar.set_xticklabels(abilities)
    ax_radar.axhline(0, color="grey", linestyle="--")
    fig_radar.tight_layout()
    st.pyplot(fig_radar)
    st.write("**Radar Chart:** Overall ability profile. Above 0 = above average, Below 0 = below average.")

# ---------------------------
# Page 2: Raw Data
# ---------------------------
elif page == "Raw Data":
    st.title("Raw Athlete Data")
    st.write("You can filter the raw data using the sidebar filters.")
    st.dataframe(filtered_df)
