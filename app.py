import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_excel("synthetic_athlete_dataset.xlsx")

# List of abilities
abilities = ["Speed", "Endurance", "Strength", "Agility", "ReactionTime"]

# Standardize abilities
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[abilities] = scaler.fit_transform(df[abilities])

# App title
st.title("Athlete Abilities Explorer (Standardized)")

# Sidebar filters
st.sidebar.header("Filter Athletes")
gender_options = ["All"] + df_scaled["Gender"].unique().tolist()
selected_gender = st.sidebar.selectbox("Select Gender", gender_options)

sport_options = ["All"] + df_scaled["Sport"].unique().tolist()
selected_sport = st.sidebar.selectbox("Select Sport", sport_options)

min_age, max_age = int(df_scaled["Age"].min()), int(df_scaled["Age"].max())
selected_age = st.sidebar.slider("Select Age Range", min_age, max_age, (min_age, max_age))

# Filter dataframe
filtered_df = df_scaled.copy()
if selected_gender != "All":
    filtered_df = filtered_df[filtered_df["Gender"] == selected_gender]
if selected_sport != "All":
    filtered_df = filtered_df[filtered_df["Sport"] == selected_sport]
filtered_df = filtered_df[(filtered_df["Age"] >= selected_age[0]) & (filtered_df["Age"] <= selected_age[1])]

# Show filtered data
st.subheader("Filtered Data")
st.dataframe(filtered_df)

# Visualization options
st.sidebar.header("Visualizations")
plot_type = st.sidebar.radio("Select Plot Type", ["Bar Plot", "Box Plot", "Pie Chart", "Correlation Heatmap", "Radar Chart"])

# ---------------------------
# Bar plot
# ---------------------------
if plot_type == "Bar Plot":
    st.subheader("Average Standardized Abilities")
    avg_values = filtered_df[abilities].mean()
    fig, ax = plt.subplots()
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in avg_values]  # Green if above avg, red if below
    sns.barplot(x=avg_values.index, y=avg_values.values, palette=colors, ax=ax)
    ax.axhline(0, color="black", linestyle="--")
    ax.set_ylabel("Standardized Score (Z-score)")
    st.pyplot(fig)

# ---------------------------
# Box plot
# ---------------------------
elif plot_type == "Box Plot":
    st.subheader("Abilities Distribution (Standardized)")
    melted = filtered_df.melt(id_vars=["AthleteID"], value_vars=abilities, var_name="Ability", value_name="Z-Score")
    fig, ax = plt.subplots()
    sns.boxplot(x="Ability", y="Z-Score", data=melted, palette="coolwarm", ax=ax)
    ax.axhline(0, color="black", linestyle="--")
    st.pyplot(fig)

# ---------------------------
# Pie chart
# ---------------------------
elif plot_type == "Pie Chart":
    st.subheader("Gender Distribution")
    if "Gender" in filtered_df.columns:
        counts = filtered_df["Gender"].value_counts()
        fig, ax = plt.subplots()
        ax.pie(counts, labels=counts.index, autopct="%1.1f%%", colors=sns.color_palette("Set2"))
        st.pyplot(fig)
    else:
        st.write("Pie chart not available for the selected filters.")

# ---------------------------
# Correlation heatmap
# ---------------------------
elif plot_type == "Correlation Heatmap":
    st.subheader("Abilities Correlation Heatmap")
    corr = filtered_df[abilities].corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# ---------------------------
# Radar chart
# ---------------------------
elif plot_type == "Radar Chart":
    st.subheader("Radar Chart: Average Standardized Abilities")
    import numpy as np

    avg_values = filtered_df[abilities].mean().values
    labels = abilities
    num_vars = len(labels)

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    avg_values = np.concatenate((avg_values, [avg_values[0]]))  # close the loop
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
    ax.plot(angles, avg_values, color="blue", linewidth=2)
    ax.fill(angles, avg_values, color="skyblue", alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.axhline(0, color="grey", linestyle="--")
    st.pyplot(fig)
