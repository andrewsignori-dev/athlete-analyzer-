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
    df = pd.read_excel("synthetic_athlete_dataset.xlsx")
    return df

df_athletes = load_data()
features = ['Speed', 'Endurance', 'Strength', 'Agility', 'ReactionTime']

# ---------------------------
# 2. Standardize data
# ---------------------------
scaler = StandardScaler()
X = scaler.fit_transform(df_athletes[features])

# ---------------------------
# 3. Archetypal analysis
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

# Compute archetypes
n_archetypes = 3
archetypes_scaled, _ = archetypal_analysis(X, n_archetypes=n_archetypes)
archetypes = scaler.inverse_transform(archetypes_scaled)

# Feature names
feature_names = ['Speed', 'Endurance', 'Strength', 'Agility', 'ReactionTime']

# Assign descriptive names robustly
used_labels = set()
archetype_labels = []

for arch in archetypes:
    # Sort trait indices by descending value
    sorted_idx = np.argsort(-arch)
    
    # Pick the first unused trait
    label = None
    for idx in sorted_idx:
        trait = feature_names[idx]
        candidate_label = f"{trait} Archetype"
        if candidate_label not in used_labels:
            label = candidate_label
            used_labels.add(candidate_label)
            break
    
    # If all labels are already used, just append "Archetype_X"
    if label is None:
        label = f"Archetype_{len(used_labels)+1}"
        used_labels.add(label)
    
    archetype_labels.append(label)


# ---------------------------
# 4. Streamlit interface
# ---------------------------
st.set_page_config(page_title="Athlete Typology Analyzer", page_icon="🏃‍♂️", layout="centered")

st.title("🏋️ Athlete Typology Analyzer")
st.markdown("Upload your test results to see which **archetype composition** best describes your athletic profile.")

st.sidebar.header("Input Athlete Metrics")
speed = st.sidebar.number_input("Speed", value=25.0, min_value=10.0, max_value=40.0, step=0.1)
endurance = st.sidebar.number_input("Endurance", value=70.0, min_value=30.0, max_value=100.0, step=0.1)
strength = st.sidebar.number_input("Strength", value=100.0, min_value=50.0, max_value=200.0, step=0.1)
agility = st.sidebar.number_input("Agility", value=50.0, min_value=20.0, max_value=80.0, step=0.1)
reaction_time = st.sidebar.number_input("Reaction Time", value=0.3, min_value=0.1, max_value=0.6, step=0.01)

if st.sidebar.button("Analyze"):
    x_input = np.array([[speed, endurance, strength, agility, reaction_time]])
    x_scaled = scaler.transform(x_input)

    alphas = np.linalg.lstsq(archetypes_scaled.T, x_scaled.T, rcond=None)[0].T
    alphas = np.clip(alphas, 0, 1)
    alphas = alphas / (alphas.sum(axis=1, keepdims=True) + 1e-8)
    alpha_percent = (alphas[0] * 100).round(2)

    st.subheader("🏅 Typology Composition")
    for label, pct in zip(archetype_labels, alpha_percent):
        st.write(f"**{label}:** {pct}%")

    # Pie chart
    fig, ax = plt.subplots()
    ax.pie(alpha_percent, labels=archetype_labels, autopct='%1.1f%%', startangle=90,
           colors=['#ff9999','#66b3ff','#99ff99'])
    ax.set_title("Archetype Composition")
    st.pyplot(fig)

    st.markdown("---")
    st.markdown("*Computed using archetypal analysis on standardized athlete metrics.*")
