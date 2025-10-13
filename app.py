# ---------------------------
# 0. Imports
# ---------------------------
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# ---------------------------
# 1. Load dataset
# ---------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_excel("synthetic_athlete_dataset.xlsx")
    except FileNotFoundError:
        # Fallback: simulate small dataset
        np.random.seed(42)
        n_athletes = 50
        df = pd.DataFrame({
            "Athlete": [f"Athlete_{i+1}" for i in range(n_athletes)],
            "Speed": np.random.normal(25, 3, n_athletes),
            "Endurance": np.random.normal(70, 10, n_athletes),
            "Strength": np.random.normal(100, 15, n_athletes),
            "Agility": np.random.normal(50, 5, n_athletes),
            "ReactionTime": np.random.normal(0.3, 0.05, n_athletes)
        })
    return df

df_athletes = load_data()
features = ['Speed', 'Endurance', 'Strength', 'Agility', 'ReactionTime']

# ---------------------------
# 2. Standardize data
# ---------------------------
scaler = StandardScaler()
X = scaler.fit_transform(df_athletes[features])

# ---------------------------
# 3. Archetypal Analysis
# ---------------------------
def archetypal_analysis(X, n_archetypes=3, n_iter=100):
    n_samples, n_features = X.shape
    archetypes = X[np.random.choice(n_samples, n_archetypes, replace=False)]
    for _ in range(n_iter):
        alphas = np.linalg.lstsq(archetypes.T, X.T, rcond=None)[0].T
        alphas = np.clip(alphas, 0, 1)
        alphas = alphas / (alphas.sum(axis=1, keepdims=True) + 1e-8)
        archetypes = (alphas.T @ X) / (alphas.sum(axis=0)[:, None] + 1e-8)
    return archetypes, alphas

n_archetypes = 3
archetypes_scaled, _ = archetypal_analysis(X, n_archetypes=n_archetypes)
archetypes = scaler.inverse_transform(archetypes_scaled)

# ---------------------------
# 4. Assign unique descriptive archetype names
# ---------------------------
feature_names = features
used_labels = set()
archetype_labels = []

for arch in archetypes:
    sorted_idx = np.argsort(-arch)  # descending
    label = None
    for idx in sorted_idx:
        candidate_label = f"{feature_names[idx]} Archetype"
        if candidate_label not in used_labels:
            label = candidate_label
            used_labels.add(label)
            break
    if label is None:
        label = f"Archetype_{len(used_labels)+1}"
        used_labels.add(label)
    archetype_labels.append(label)

# ---------------------------
# 5. Streamlit Interface
# ---------------------------
st.set_page_config(page_title="Athlete Typology Analyzer", page_icon="🏃‍♂️", layout="centered")
st.title("🏋️ Athlete Typology Analyzer")
st.markdown("Enter athlete performance metrics to see your archetype composition and dominant traits.")

# Input sidebar
st.sidebar.header("Input Athlete Metrics")
speed = st.sidebar.number_input("Speed", value=25.0, min_value=10.0, max_value=40.0, step=0.1)
endurance = st.sidebar.number_input("Endurance", value=70.0, min_value=30.0, max_value=100.0, step=0.1)
strength = st.sidebar.number_input("Strength", value=100.0, min_value=50.0, max_value=200.0, step=0.1)
agility = st.sidebar.number_input("Agility", value=50.0, min_value=20.0, max_value=80.0, step=0.1)
reaction_time = st.sidebar.number_input("Reaction Time", value=0.3, min_value=0.1, max_value=0.6, step=0.01)

if st.sidebar.button("Analyze"):
    x_input = np.array([[speed, endurance, strength, agility, reaction_time]])
    x_scaled = scaler.transform(x_input)

    # Compute alphas
    alphas = np.linalg.lstsq(archetypes_scaled.T, x_scaled.T, rcond=None)[0].T
    alphas = np.clip(alphas, 0, 1)
    alphas = alphas / (alphas.sum(axis=1, keepdims=True) + 1e-8)
    alpha_percent = (alphas[0] * 100).round(2)

    # Safety: avoid all zeros
    if alpha_percent.sum() == 0:
        alpha_percent = np.array([33.3, 33.3, 33.4])

    # Normalize to sum 100
    alpha_percent = np.maximum(alpha_percent, 0.1)
    alpha_percent = alpha_percent / alpha_percent.sum() * 100

    # Display textual composition
    st.subheader("🏅 Archetype Composition")
    for label, pct in zip(archetype_labels, alpha_percent):
        st.write(f"**{label}:** {pct:.1f}%")

    # Pie chart
    fig, ax = plt.subplots()
    ax.pie(alpha_percent, labels=archetype_labels, autopct='%1.1f%%', startangle=90,
           colors=['#ff9999','#66b3ff','#99ff99'])
    ax.set_title("Archetype Composition")
    st.pyplot(fig)

    # Show dominant trait values for each archetype
    st.subheader("📊 Dominant Trait Values of Archetypes")
    dominant_values = []
    for arch, label in zip(archetypes, archetype_labels):
        dominant_idx = np.argmax(arch)
        dominant_trait = feature_names[dominant_idx]
        value = arch[dominant_idx]
        dominant_values.append({"Archetype": label, "Dominant Trait": dominant_trait, "Value": round(value, 2)})
    st.table(pd.DataFrame(dominant_values))

    st.markdown("---")
    st.markdown("*Computed using archetypal analysis on standardized athlete metrics.*")
