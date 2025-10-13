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

# App title
st.title("Athlete Abilities Dashboard (Standardized)")

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

# ---------------------------
# 1. Bar plot: average abilities
# ---------------------------
st.subheader("Average Standardized Abilities")
st.write("This bar plot shows the average performance of the selected athletes in each ability. "
         "Green bars indicate above-average abilities, red bars indicate below-average abilities (relative to all athletes).")

avg_values = filtered_df[abilities].mean()
colors = ["#2ca02c" if v >= 0 else "#d62728" for v in avg_values]

fig, ax = plt.subplots()
sns.barplot(x=avg_values.index, y=avg_values.values, palette=colors, ax=ax)
ax.axhline(0, color="black", linestyle="--")
ax.set_ylabel("Standardized Score (Z-score)")
st.pyplot(fig)

# ---------------------------
# 2. Box plot: distribution of abilities
# ---------------------------
st.subheader("Distribution of Abilities")
st.write("This box plot shows the distribution of standardized scores for each ability. "
         "The horizontal line at 0 represents the overall average.")

melted = filtered_df.melt(id_vars=["AthleteID"], value_vars=abilities, var_name="Ability", value_name="Z-Score")
fig, ax = plt.subplots()
sns.boxplot(x="Ability", y="Z-Score", data=melted, palette="coolwarm", ax=ax)
ax.axhline(0, color="black", linestyle="--")
st.pyplot(fig)

# ---------------------------
# 3. Pie chart: gender distribution
# ---------------------------
st.subheader("Gender Distribution")
st.write("This pie chart shows the proportion of male and female athletes in the selected dataset.")

if "Gender" in filtered_df.columns:
    counts = filtered_df["Gender"].value_counts()
    fig, ax = plt.subplots()
    ax.pie(counts, labels=counts.index, autopct="%1.1f%%", colors=sns.color_palette("Set2"))
    st.pyplot(fig)

# ---------------------------
# 4. Correlation heatmap
# ---------------------------
st.subheader("Correlation Between Abilities")
st.write("This heatmap shows how each ability correlates with the others. "
         "Values close to 1 or -1 indicate strong positive or negative correlation, respectively.")

corr = filtered_df[abilities].corr()
fig, ax = plt.subplots()
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# ---------------------------
# 5. Radar chart: overall ability profile
# ---------------------------
st.subheader("Radar Chart: Overall Ability Profile")
st.write("The radar chart shows the average standardized abilities for the selected athletes in a single view. "
         "Abilities above 0 are above average, below 0 are below average.")

avg_values = filtered_df[abilities].mean().values
labels = abilities
num_vars = len(labels)

angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
avg_values_loop = np.concatenate((avg_values, [avg_values[0]]))
angles_loop = angles + angles[:1]

fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
ax.plot(angles_loop, avg_values_loop, color="blue", linewidth=2)
ax.fill(angles_loop, avg_values_loop, color="skyblue", alpha=0.25)
ax.set_xticks(angles)
ax.set_xticklabels(labels)
ax.axhline(0, color="grey", linestyle="--")
st.pyplot(fig)
