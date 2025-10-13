import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np

# --- Load Dataset ---
df = pd.read_excel("synthetic_athlete_dataset.xlsx")

# --- Standardize Ability Columns ---
abilities = ["Speed", "Endurance", "Strength", "Agility", "ReactionTime"]
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[abilities] = scaler.fit_transform(df[abilities])

# --- Streamlit Sidebar ---
st.sidebar.title("⚡ Athlete Dashboard")
page = st.sidebar.radio("Navigate to", ["Dashboard", "Raw Data"])

st.sidebar.header("Filter Athletes")
gender_options = ["All"] + df_scaled["Gender"].unique().tolist()
selected_gender = st.sidebar.selectbox("Gender", gender_options)
sport_options = ["All"] + df_scaled["Sport"].unique().tolist()
selected_sport = st.sidebar.selectbox("Sport", sport_options)
min_age, max_age = int(df_scaled["Age"].min()), int(df_scaled["Age"].max())
selected_age = st.sidebar.slider("Age Range", min_age, max_age, (min_age, max_age))

# --- Apply Filters ---
filtered_df = df_scaled.copy()
if selected_gender != "All":
    filtered_df = filtered_df[filtered_df["Gender"] == selected_gender]
if selected_sport != "All":
    filtered_df = filtered_df[filtered_df["Sport"] == selected_sport]
filtered_df = filtered_df[(filtered_df["Age"] >= selected_age[0]) & (filtered_df["Age"] <= selected_age[1])]

# --- Set Seaborn Theme ---
sns.set_theme(style="whitegrid")

# --- Dashboard Page ---
if page == "Dashboard":
    st.title("🏋️ Athlete Abilities Dashboard", anchor=None)
    st.markdown("Explore the abilities of athletes in different sports and genders. All scores are standardized (Z-scores).")
    st.markdown("---")

    # --- Bar Plot ---
    avg_values = filtered_df[abilities].mean()
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in avg_values]
    fig_bar, ax_bar = plt.subplots(figsize=(14,7))
    sns.barplot(x=avg_values.index, y=avg_values.values, palette=colors, ax=ax_bar)
    ax_bar.axhline(0, color="black", linestyle="--")
    ax_bar.set_ylabel("Z-score", fontsize=12)
    ax_bar.set_xlabel("Ability", fontsize=12)
    ax_bar.tick_params(labelsize=11)
    fig_bar.tight_layout()
    st.subheader("📊 Average Abilities")
    st.pyplot(fig_bar)
    st.markdown("Green bars indicate above average performance; red bars indicate below average performance.")
    st.markdown("<br>", unsafe_allow_html=True)

    # --- Box Plot ---
    melted = filtered_df.melt(id_vars=["AthleteID"], value_vars=abilities, var_name="Ability", value_name="Z-Score")
    fig_box, ax_box = plt.subplots(figsize=(14,7))
    sns.boxplot(x="Ability", y="Z-Score", data=melted, palette="coolwarm", ax=ax_box)
    ax_box.axhline(0, color="black", linestyle="--")
    ax_box.set_xlabel("Ability", fontsize=12)
    ax_box.set_ylabel("Z-score", fontsize=12)
    ax_box.tick_params(labelsize=11)
    fig_box.tight_layout()
    st.subheader("📦 Ability Distribution")
    st.pyplot(fig_box)
    st.markdown("Box plots show the spread of abilities among filtered athletes (0 = overall average).")
    st.markdown("<br>", unsafe_allow_html=True)

    # --- Pie Chart ---
    counts = filtered_df["Gender"].value_counts()
    fig_pie, ax_pie = plt.subplots(figsize=(8,8))
    ax_pie.pie(counts, labels=counts.index, autopct="%1.1f%%", colors=sns.color_palette("Set2"), startangle=90)
    ax_pie.set_title("Gender Distribution", fontsize=14)
    fig_pie.tight_layout()
    st.subheader("🥧 Gender Distribution")
    st.pyplot(fig_pie)
    st.markdown("<br>", unsafe_allow_html=True)

    # --- Heatmap ---
    corr = filtered_df[abilities].corr()
    fig_corr, ax_corr = plt.subplots(figsize=(14,7))
    sns.heatmap(corr, annot=True, cmap="coolwarm", linewidths=0.5, ax=ax_corr)
    ax_corr.set_title("Correlation Between Abilities", fontsize=14)
    fig_corr.tight_layout()
    st.subheader("🔥 Ability Correlations")
    st.pyplot(fig_corr)
    st.markdown("<br>", unsafe_allow_html=True)

    # --- Radar Chart ---
    avg_values_radar = filtered_df[abilities].mean().values
    num_vars = len(abilities)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    avg_values_loop = np.concatenate((avg_values_radar, [avg_values_radar[0]]))
    angles_loop = angles + angles[:1]

    fig_radar, ax_radar = plt.subplots(figsize=(12,12), subplot_kw=dict(polar=True))
    ax_radar.plot(angles_loop, avg_values_loop, color="blue", linewidth=2)
    ax_radar.fill(angles_loop, avg_values_loop, color="skyblue", alpha=0.25)
    ax_radar.set_xticks(angles)
    ax_radar.set_xticklabels(abilities, fontsize=12)
    ax_radar.set_yticklabels(np.round(ax_radar.get_yticks(), 2), fontsize=11)
    ax_radar.axhline(0, color="grey", linestyle="--")
    fig_radar.tight_layout()
    st.subheader("📡 Overall Ability Profile")
    st.pyplot(fig_radar)
    st.markdown("Radar chart shows the overall performance per ability. Above 0 = above average, below 0 = below average.")

# --- Raw Data Page ---
elif page == "Raw Data":
    st.title("📝 Raw Athlete Data")
    st.write("You can filter the raw data using the sidebar filters below.")
    st.dataframe(filtered_df)
