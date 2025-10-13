import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(page_title="Athlete Ability", layout="wide", initial_sidebar_state="expanded")

# Load dataset
df = pd.read_excel("synthetic_athlete_dataset.xlsx")

# Ability columns
abilities = ["Speed", "Endurance", "Strength", "Agility", "ReactionTime"]

# Standardize abilities
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[abilities] = scaler.fit_transform(df[abilities])

# ---------------------------
# Sidebar Filters
# ---------------------------
st.sidebar.title("⚡ Athlete Dashboard Filters")
page = st.sidebar.radio("Navigate to", ["Athlete Ability", "Raw Data"])

st.sidebar.header("Filter Athletes")
gender_options = ["All"] + df_scaled["Gender"].unique().tolist()
selected_gender = st.sidebar.selectbox("Gender", gender_options)
sport_options = ["All"] + df_scaled["Sport"].unique().tolist()
selected_sport = st.sidebar.selectbox("Sport", sport_options)
min_age, max_age = int(df_scaled["Age"].min()), int(df_scaled["Age"].max())
selected_age = st.sidebar.slider("Age Range", min_age, max_age, (min_age, max_age))

# Apply filters
filtered_df = df_scaled.copy()
if selected_gender != "All":
    filtered_df = filtered_df[filtered_df["Gender"] == selected_gender]
if selected_sport != "All":
    filtered_df = filtered_df[filtered_df["Sport"] == selected_sport]
filtered_df = filtered_df[(filtered_df["Age"] >= selected_age[0]) & (filtered_df["Age"] <= selected_age[1])]

# ---------------------------
# Seaborn theme
# ---------------------------
sns.set_theme(style="whitegrid")

# ---------------------------
# Figure Sizes
# ---------------------------
fig_width = 7
fig_height = 4.5

# ---------------------------
# Athlete Ability Page
# ---------------------------
if page == "Athlete Ability":
    st.title("🏋️ Athlete Abilities Dashboard")
    st.markdown("Explore standardized abilities of athletes.")
    st.markdown("---")

    # --- Row 1: Bar & Box ---
    col1, col2 = st.columns(2)

    # Bar Plot
    with col1:
        avg_values = filtered_df[abilities].mean()
        colors = ["#2ca02c" if v >= 0 else "#d62728" for v in avg_values]
        fig_bar, ax_bar = plt.subplots(figsize=(fig_width, fig_height))
        sns.barplot(x=avg_values.index, y=avg_values.values, palette=colors, ax=ax_bar)
        ax_bar.axhline(0, color="black", linestyle="--")
        ax_bar.set_ylabel("Z-score")
        ax_bar.set_xlabel("Ability")
        fig_bar.tight_layout()
        st.subheader("📊 Average Abilities")
        st.pyplot(fig_bar)

    # Box Plot
    with col2:
        melted = filtered_df.melt(id_vars=["AthleteID"], value_vars=abilities, var_name="Ability", value_name="Z-Score")
        fig_box, ax_box = plt.subplots(figsize=(fig_width, fig_height))
        sns.boxplot(x="Ability", y="Z-Score", data=melted, palette="coolwarm", ax=ax_box)
        ax_box.axhline(0, color="black", linestyle="--")
        fig_box.tight_layout()
        st.subheader("📦 Ability Distribution")
        st.pyplot(fig_box)

    st.markdown("---")

    # --- Row 2: Radar & Heatmap ---
    col1, col2 = st.columns(2)

    # Radar Chart
    with col1:
        avg_values_radar = filtered_df[abilities].mean().values
        num_vars = len(abilities)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        avg_values_loop = np.concatenate((avg_values_radar, [avg_values_radar[0]]))
        angles_loop = angles + angles[:1]
        fig_radar, ax_radar = plt.subplots(figsize=(fig_width, fig_height), subplot_kw=dict(polar=True))
        ax_radar.plot(angles_loop, avg_values_loop, color="blue", linewidth=2)
        ax_radar.fill(angles_loop, avg_values_loop, color="skyblue", alpha=0.25)
        ax_radar.set_xticks(angles)
        ax_radar.set_xticklabels(abilities)
        ax_radar.set_yticklabels(np.round(ax_radar.get_yticks(), 2))
        ax_radar.axhline(0, color="grey", linestyle="--")
        fig_radar.tight_layout()
        st.subheader("📡 Overall Ability Profile")
        st.pyplot(fig_radar)

    # Heatmap
    with col2:
        corr = filtered_df[abilities].corr()
        fig_corr, ax_corr = plt.subplots(figsize=(fig_width, fig_height))
        sns.heatmap(corr, annot=True, cmap="coolwarm", linewidths=0.5, ax=ax_corr)
        ax_corr.set_title("Correlation Between Abilities")
        fig_corr.tight_layout()
        st.subheader("🔥 Ability Correlations")
        st.pyplot(fig_corr)

# ---------------------------
# Raw Data Page
# ---------------------------
elif page == "Raw Data":
    st.title("📝 Raw Athlete Data")
    st.write("You can filter the raw data using the sidebar filters below.")

     # Show table
    st.dataframe(df)

    # Smaller figure size for Raw Data plots
    fig_width_raw = 4
    fig_height_raw = 2.5
    label_fontsize = 5  # smaller labels

    # --- Row 1: Gender Pie & Age Bar ---
    col1, col2 = st.columns(2)

    # Gender Pie Chart
    with col1:
        counts = filtered_df["Gender"].value_counts()
        fig_pie, ax_pie = plt.subplots(figsize=(fig_width_raw, fig_height_raw))
        ax_pie.pie(counts, labels=counts.index, autopct="%1.1f%%",
               colors=sns.color_palette("Set2"), startangle=90,
               textprops={'fontsize': label_fontsize})
        #ax_pie.set_title("Gender Distribution", fontsize=12)
        fig_pie.tight_layout()
        st.subheader("🥧 Gender Distribution")
        st.pyplot(fig_pie)

# Age Bar Plot
    with col2:
       fig_age, ax_age = plt.subplots(figsize=(fig_width_raw, fig_height_raw))
       sns.histplot(filtered_df["Age"], bins=10, kde=False, color="#1f77b4", ax=ax_age)
       ax_age.set_xlabel("Age")
       ax_age.set_ylabel("Count")
       #ax_age.set_title("Age Distribution")
       fig_age.tight_layout()
       st.subheader("📊 Age Distribution")
       st.pyplot(fig_age)

   st.markdown("---")

   # --- Row 2: Sport Pie Chart aligned with first row ---
    col1, col2 = st.columns([1, 1])  # two equal columns
    with col1:
       fig_sport, ax_sport = plt.subplots(figsize=(fig_width_raw, fig_height_raw))
       sport_counts = filtered_df["Sport"].value_counts()
       ax_sport.pie(sport_counts, labels=sport_counts.index, autopct="%1.1f%%",
                 colors=sns.color_palette("Set2"), startangle=90,
                 textprops={'fontsize': label_fontsize})
       #ax_sport.set_title("Sport Distribution", fontsize=12)
      fig_sport.tight_layout()
      st.subheader("🏅 Sport Distribution")
      st.pyplot(fig_sport)



