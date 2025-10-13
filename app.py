# app.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_excel("synthetic_athlete_dataset.xlsx")

# App title
st.title("Athlete Abilities Explorer")

# Sidebar filters
st.sidebar.header("Filter Athletes")

# Gender filter
gender_options = ["All"] + df["Gender"].unique().tolist()
selected_gender = st.sidebar.selectbox("Select Gender", gender_options)

# Sport filter
sport_options = ["All"] + df["Sport"].unique().tolist()
selected_sport = st.sidebar.selectbox("Select Sport", sport_options)

# Age range filter
min_age, max_age = int(df["Age"].min()), int(df["Age"].max())
selected_age = st.sidebar.slider("Select Age Range", min_age, max_age, (min_age, max_age))

# Filter dataframe based on selections
filtered_df = df.copy()
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
plot_type = st.sidebar.radio("Select Plot Type", ["Bar Plot", "Box Plot", "Pie Chart", "Correlation Heatmap"])

abilities = ["Speed", "Endurance", "Strength", "Agility", "ReactionTime"]

# Bar plot
if plot_type == "Bar Plot":
    st.subheader("Average Abilities")
    avg_values = filtered_df[abilities].mean()
    fig, ax = plt.subplots()
    sns.barplot(x=avg_values.index, y=avg_values.values, palette="viridis", ax=ax)
    ax.set_ylabel("Average Value")
    st.pyplot(fig)

# Box plot
elif plot_type == "Box Plot":
    st.subheader("Abilities Distribution")
    melted = filtered_df.melt(id_vars=["AthleteID"], value_vars=abilities, var_name="Ability", value_name="Value")
    fig, ax = plt.subplots()
    sns.boxplot(x="Ability", y="Value", data=melted, palette="coolwarm", ax=ax)
    st.pyplot(fig)

# Pie chart
elif plot_type == "Pie Chart":
    st.subheader("Gender Distribution")
    if "Gender" in filtered_df.columns:
        gender_counts = filtered_df["Gender"].value_counts()
        fig, ax = plt.subplots()
        ax.pie(gender_counts, labels=gender_counts.index, autopct="%1.1f%%", colors=sns.color_palette("Set2"))
        st.pyplot(fig)
    else:
        st.write("Pie chart not available for the selected filters.")

# Correlation heatmap
elif plot_type == "Correlation Heatmap":
    st.subheader("Abilities Correlation Heatmap")
    corr = filtered_df[abilities].corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
