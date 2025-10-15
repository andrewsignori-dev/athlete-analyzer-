import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
import statsmodels.api as sm
from scipy.special import logit

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(page_title="Athlete Ability", layout="centered", initial_sidebar_state="expanded")

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
page = st.sidebar.radio("Navigate to", ["Athlete Ability", "Injury Risk Model"])


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
# Small Figure Sizes
# ---------------------------
fig_width = 4.5
fig_height = 3
font_size = 6

# ---------------------------
# Athlete Ability Page
# ---------------------------
if page == "Athlete Ability":
    st.title("🏋️ Athlete Abilities Dashboard")
    st.markdown("Explore standardized abilities of athletes and evaluate a new athlete.")
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
        ax_bar.set_ylabel("Z-score", fontsize=font_size)
        ax_bar.set_xlabel("Ability", fontsize=font_size)
        ax_bar.tick_params(axis='x', labelsize=font_size)
        ax_bar.tick_params(axis='y', labelsize=font_size)
        fig_bar.tight_layout()
        st.subheader("📊 Average Abilities")
        st.pyplot(fig_bar)

    # Box Plot
    with col2:
        melted = filtered_df.melt(id_vars=["AthleteID"], value_vars=abilities, var_name="Ability", value_name="Z-Score")
        fig_box, ax_box = plt.subplots(figsize=(fig_width, fig_height))
        sns.boxplot(x="Ability", y="Z-Score", data=melted, palette="coolwarm", ax=ax_box)
        ax_box.axhline(0, color="black", linestyle="--")
        ax_box.tick_params(axis='x', labelsize=font_size)
        ax_box.tick_params(axis='y', labelsize=font_size)
        ax_box.set_xlabel("Ability", fontsize=font_size)
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
        ax_radar.plot(angles_loop, avg_values_loop, color="blue", linewidth=1.5)
        ax_radar.fill(angles_loop, avg_values_loop, color="skyblue", alpha=0.25)
        ax_radar.set_xticks(angles)
        ax_radar.set_xticklabels(abilities, fontsize=font_size-1)
        ax_radar.set_yticklabels(np.round(ax_radar.get_yticks(), 2), fontsize=font_size)
        ax_radar.axhline(0, color="grey", linestyle="--")
        fig_radar.tight_layout()
        st.subheader("📡 Overall Ability Profile")
        st.pyplot(fig_radar)

    # Heatmap
    with col2:
        corr = filtered_df[abilities].corr()
        fig_corr, ax_corr = plt.subplots(figsize=(fig_width, fig_height))
        sns.heatmap(corr, annot=True, cmap="coolwarm", linewidths=0.5, ax=ax_corr, annot_kws={"size": font_size})
        ax_corr.set_title("Correlation Between Abilities", fontsize=font_size)
        fig_corr.tight_layout()
        st.subheader("🔥 Ability Correlations")
        st.pyplot(fig_corr)

    # ---------------------------
    # Evaluate a New Athlete (Collapsible Section)
    # ---------------------------
    with st.expander("🏅 Evaluate a New Athlete"):
        st.markdown("Input ability levels for a new athlete and compare them to the reference group.")
        st.markdown("---")

        # Reference group selection
        ref_group = st.selectbox("Reference Group", ["All", "Gender", "Sport"])
        if ref_group == "All":
            ref_df = filtered_df.copy()
            legend_label = "Dataset Average"
        elif ref_group == "Gender":
            selected_ref_gender = st.selectbox("Select Gender", df["Gender"].unique())
            ref_df = filtered_df[filtered_df["Gender"] == selected_ref_gender]
            legend_label = f"{selected_ref_gender} Average"
        else:
            selected_ref_sport = st.selectbox("Select Sport", df["Sport"].unique())
            ref_df = filtered_df[filtered_df["Sport"] == selected_ref_sport]
            legend_label = f"{selected_ref_sport} Average"

        # Input sliders for new athlete
        athlete_input = {}
        for ability in abilities:
            min_val = float(df[ability].min())
            max_val = float(df[ability].max())
            mean_val = float(df[ability].mean())
            athlete_input[ability] = st.slider(f"{ability}", min_val, max_val, mean_val, 0.1)

        new_athlete_df = pd.DataFrame([athlete_input])
        new_athlete_scaled = pd.DataFrame(scaler.transform(new_athlete_df), columns=abilities)

        # Compute reference average
        avg_values_ref = ref_df[abilities].mean()
        new_values = new_athlete_scaled.iloc[0]

        # --- Summary Text ---
        st.markdown("### 🧾 Summary")
        differences = new_values - avg_values_ref
        above = differences[differences > 0].index.tolist()
        below = differences[differences < 0].index.tolist()
        summary_text = ""
        if above:
            summary_text += f"**Above average in:** {', '.join(above)}  \n"
        if below:
            summary_text += f"**Below average in:** {', '.join(below)}  \n"
        if not above and not below:
            summary_text = "This athlete's abilities are around the reference group average."
        st.markdown(summary_text)

        # Comparison Bar Chart
        st.subheader("📊 Comparison to Reference Average")
        fig_bar_eval, ax_bar_eval = plt.subplots(figsize=(3.5, 2))
        x = np.arange(len(abilities))
        width = 0.35
        ax_bar_eval.bar(x - width/2, avg_values_ref, width, label=legend_label, color='#888888')
        ax_bar_eval.bar(x + width/2, new_values, width, label='New Athlete', color='#1f77b4')
        ax_bar_eval.axhline(0, color="black", linestyle="--")
        ax_bar_eval.set_xticks(x)
        ax_bar_eval.set_xticklabels(abilities, fontsize=font_size)
        ax_bar_eval.set_ylabel("Z-score", fontsize=font_size)
        ax_bar_eval.tick_params(axis='y', labelsize=font_size)
        ax_bar_eval.legend(fontsize=font_size)
        fig_bar_eval.tight_layout()
        st.pyplot(fig_bar_eval)

        # Boxplot overlay
        st.subheader("📡 Percentile Placement")
        fig_box_overlay, ax_box_overlay = plt.subplots(figsize=(3.5, 2))
        melted_ref = ref_df.melt(id_vars=["AthleteID"], value_vars=abilities, var_name="Ability", value_name="Z-Score")
        sns.boxplot(x="Ability", y="Z-Score", data=melted_ref, palette="coolwarm", ax=ax_box_overlay)
        ax_box_overlay.axhline(0, color="black", linestyle="--")
        ax_box_overlay.tick_params(axis='x', labelsize=font_size)
        ax_box_overlay.tick_params(axis='y', labelsize=font_size)
        ax_box_overlay.set_ylabel("Z-score", fontsize=font_size)
        ax_box_overlay.set_xlabel("Ability", fontsize=font_size)

        # Overlay new athlete points
        for i, ability in enumerate(abilities):
            ax_box_overlay.scatter(i, new_values[ability], color="#1f77b4", s=50, zorder=10,
                                   label="New Athlete" if i == 0 else "")
        ax_box_overlay.legend(fontsize=font_size)
        fig_box_overlay.tight_layout()
        st.pyplot(fig_box_overlay)

    # ---------------------------
    # Raw Data (Collapsible Section)
    # ---------------------------
    with st.expander("📝 Show Raw Athlete Data (Dynamic)"):
        st.markdown("Filter and explore the raw data independently from the main dashboard.")
        st.markdown("---")

        # Filters
        gender_options_raw = ["All"] + df["Gender"].unique().tolist()
        selected_gender_raw = st.selectbox("Filter by Gender", gender_options_raw, key="gender_raw")
        sport_options_raw = ["All"] + df["Sport"].unique().tolist()
        selected_sport_raw = st.selectbox("Filter by Sport", sport_options_raw, key="sport_raw")
        min_age_raw, max_age_raw = int(df["Age"].min()), int(df["Age"].max())
        selected_age_raw = st.slider("Filter by Age Range", min_age_raw, max_age_raw, (min_age_raw, max_age_raw), key="age_raw")

        # Apply filters
        filtered_df_raw = df.copy()
        if selected_gender_raw != "All":
            filtered_df_raw = filtered_df_raw[filtered_df_raw["Gender"] == selected_gender_raw]
        if selected_sport_raw != "All":
            filtered_df_raw = filtered_df_raw[filtered_df_raw["Sport"] == selected_sport_raw]
        filtered_df_raw = filtered_df_raw[(filtered_df_raw["Age"] >= selected_age_raw[0]) & 
                                          (filtered_df_raw["Age"] <= selected_age_raw[1])]

        # Show filtered table
        st.dataframe(filtered_df_raw)

        # Compact plots
        fig_width_raw = 4
        fig_height_raw = 2
        label_fontsize = 6

        # --- Row 1: Gender Pie & Age Histogram ---
        col1, col2 = st.columns(2)

        with col1:
            counts = filtered_df_raw["Gender"].value_counts()
            fig_pie, ax_pie = plt.subplots(figsize=(fig_width_raw, fig_height_raw))
            ax_pie.pie(counts, labels=counts.index, autopct="%1.1f%%",
                       colors=sns.color_palette("Set2"), startangle=90,
                       textprops={'fontsize': label_fontsize})
            fig_pie.tight_layout()
            st.subheader("🥧 Gender Distribution")
            st.pyplot(fig_pie)

        with col2:
            fig_age, ax_age = plt.subplots(figsize=(fig_width_raw, fig_height_raw))
            sns.histplot(filtered_df_raw["Age"], bins=10, kde=False, color="#1f77b4", ax=ax_age)
            ax_age.set_xlabel("Age")
            ax_age.set_ylabel("Count")
            fig_age.tight_layout()
            st.subheader("📊 Age Distribution")
            st.pyplot(fig_age)

        st.markdown("---")

        # --- Row 2: Sport Pie Chart ---
        col1, col2 = st.columns(2)

        with col1:
            sport_counts = filtered_df_raw["Sport"].value_counts()
            fig_sport, ax_sport = plt.subplots(figsize=(fig_width_raw, fig_height_raw))
            ax_sport.pie(sport_counts, labels=sport_counts.index, autopct="%1.1f%%",
                         colors=sns.color_palette("Set2"), startangle=90,
                         textprops={'fontsize': label_fontsize})
            fig_sport.tight_layout()
            st.subheader("🏅 Sport Distribution")
            st.pyplot(fig_sport)

        with col2:
            st.write("")  # empty space for alignment

# ---------------------------
# Injury Risk Model Page
# ---------------------------
elif page == "Injury Risk Model":
    st.title("🏃 Injury Risk Model")
    st.markdown("Estimate the workload threshold (Kg) associated with a target injury probability")
    st.markdown("---")


    # Use the same Seaborn theme and figure sizing
    sns.set_theme(style="whitegrid")
    fig_width = 4.5
    fig_height = 3
    font_size = 6

    df_injury = pd.read_excel("synthetic_athlete_injury.xlsx")

    # Fit logistic model
    X = sm.add_constant(df_injury["Workload"])
    y = df_injury["Injury"]
    model = sm.Logit(y, X).fit(disp=False)
    beta0_hat, beta1_hat = model.params

    # Sidebar-like slider for target probability
    prob_target_percent = st.slider(
        "🎯 Target injury probability (%)",
        min_value=0,
        max_value=100,
        value=20,
        step=1,
        help="Select the target injury probability to compute the workload threshold (Kg)."
    )

    # Compute threshold and bootstrap CI
    X_target = prob_target_percent / 100
    w_star = (logit(X_target) - beta0_hat) / beta1_hat

    B = 200
    w_star_boot = []
    for _ in range(B):
        sample = df_injury.sample(n=len(df_injury), replace=True)
        Xb = sm.add_constant(sample["Workload"])
        yb = sample["Injury"]
        try:
            mb = sm.Logit(yb, Xb).fit(disp=False)
            b0, b1 = mb.params
            w_b = (logit(X_target) - b0) / b1
            w_star_boot.append(w_b)
        except:
            continue

    ci_lower, ci_upper = np.percentile(w_star_boot, [2.5, 97.5])

    # --- Row 1: Plot and Results ---
    col1, col2 = st.columns([2, 1])

    with col1:
        work_range = np.linspace(0, 30, 200)
        p_pred = 1 / (1 + np.exp(-(beta0_hat + beta1_hat * work_range)))

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.scatter(df_injury["Workload"], df_injury["Injury"],alpha=0.4, s=40, color="#1f77b4", label="Observed Data", edgecolor="w")
        ax.plot(work_range, p_pred, color="#d62728", linewidth=2, label="Predicted Probability")
        ax.fill_between(work_range, 0, p_pred, color="#d62728", alpha=0.1)
        ax.axvline(w_star, color="#2ca02c", linestyle="--", linewidth=2,label=f"Optimal Workload (w*) = {w_star:.2f}")
        ax.scatter(w_star, np.interp(w_star, work_range, p_pred), color="#2ca02c", s=80, zorder=5)
        ax.set_xlabel("Workload", fontsize=font_size+2, weight='bold')
        ax.set_ylabel("Injury Probability", fontsize=font_size+2, weight='bold')
        ax.tick_params(axis='x', labelsize=font_size)
        ax.tick_params(axis='y', labelsize=font_size)
        ax.grid(alpha=0.3, linestyle='--')
        ax.legend(fontsize=font_size, frameon=True, facecolor='white', edgecolor='black')
        fig.tight_layout()
        st.pyplot(fig)


    with col2:
        st.subheader("📋 Results")
        st.markdown(f"**Workload threshold (Kg)**: {w_star:.2f}")
        st.markdown(f"**95% CI**: [{ci_lower:.2f}, {ci_upper:.2f}]")
        st.markdown("---")
        st.markdown(f"For an injury probability of **{prob_target_percent}%**, the estimated workload threshold is **{w_star:.2f}**.")

    st.markdown("---")

    # --- Row 2: Dataset Preview ---
    with st.expander("🧮 Show simulated dataset"):
        st.dataframe(df_injury)




