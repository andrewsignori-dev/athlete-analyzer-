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
page = st.sidebar.radio("Navigate to", ["Athlete Ability", "Injury Risk Model", "Performance Prediction Model","Injury"])

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

    # --- In-page Filters ---
    st.subheader("🎯 Filter Athletes")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender_options = ["All"] + df_scaled["Gender"].unique().tolist()
        selected_gender = st.selectbox("Gender", gender_options)

    with col2:
        sport_options = ["All"] + df_scaled["Sport"].unique().tolist()
        selected_sport = st.selectbox("Sport", sport_options)

    with col3:
        min_age, max_age = int(df_scaled["Age"].min()), int(df_scaled["Age"].max())
        selected_age = st.slider("Age Range", min_age, max_age, (min_age, max_age))

    # --- Apply filters ---
    filtered_df = df_scaled.copy()
    if selected_gender != "All":
        filtered_df = filtered_df[filtered_df["Gender"] == selected_gender]
    if selected_sport != "All":
        filtered_df = filtered_df[filtered_df["Sport"] == selected_sport]
    filtered_df = filtered_df[
        (filtered_df["Age"] >= selected_age[0]) &
        (filtered_df["Age"] <= selected_age[1])
    ]

    st.markdown("---")

    # --- Display Filtered Results ---
    st.subheader("📊 Filtered Athlete Dataset")
    st.write(f"**Total Athletes:** {len(filtered_df)}")
    st.dataframe(filtered_df, use_container_width=True)

    # --- Optional Summary Section ---
    st.markdown("### 🧭 Summary Insights")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Average Age", f"{filtered_df['Age'].mean():.1f}")
        if "Gender" in filtered_df.columns:
            male_ratio = filtered_df["Gender"].eq("Male").mean() * 100
            st.metric("Male %", f"{male_ratio:.1f}%")

    with col2:
        if "Sport" in filtered_df.columns:
            st.metric("Unique Sports", f"{filtered_df['Sport'].nunique()}")
        st.metric("Total Records", f"{len(filtered_df)}")

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
# Injury Risk Model Page
# ---------------------------
elif page == "Injury Risk Model":
    st.title("🏃 Injury Risk Model")
    st.markdown("Estimate the workload threshold (Kg) associated with a target injury probability")
    st.markdown("---")

    # Load dataset
    df_injury = pd.read_excel("synthetic_athlete_injury.xlsx")

    # --- In-page Filters ---
    st.subheader("🎯 Filter Dataset Before Prediction")
    col1, col2 = st.columns(2)

    with col1:
        gender_options = ["All"] + sorted(df_injury["Gender"].dropna().unique().tolist())
        selected_gender = st.selectbox("Gender", gender_options)

    with col2:
        sport_options = ["All"] + sorted(df_injury["Sport"].dropna().unique().tolist())
        selected_sport = st.selectbox("Sport", sport_options)

    # Apply filters
    filtered_df = df_injury.copy()
    if selected_gender != "All":
        filtered_df = filtered_df[filtered_df["Gender"] == selected_gender]
    if selected_sport != "All":
        filtered_df = filtered_df[filtered_df["Sport"] == selected_sport]

    # If no data left after filtering, stop
    if filtered_df.empty:
        st.warning("⚠️ No data available for the selected filters. Please adjust your selection.")
        st.stop()

    st.markdown("---")

    # Seaborn theme and figure sizing
    sns.set_theme(style="whitegrid")
    fig_width = 4.5
    fig_height = 3
    font_size = 6

    # --- Fit logistic regression model ---
    X = sm.add_constant(filtered_df["Workload"])
    y = filtered_df["Injury"]
    model = sm.Logit(y, X).fit(disp=False)
    beta0_hat, beta1_hat = model.params

    # --- Injury probability slider ---
    prob_target_percent = st.slider(
        "🎯 Target injury probability (%)",
        min_value=0,
        max_value=100,
        value=20,
        step=1,
        help="Select the target injury probability to compute the workload threshold (Kg)."
    )

    # --- Compute threshold and bootstrap CI ---
    X_target = prob_target_percent / 100
    w_star = (logit(X_target) - beta0_hat) / beta1_hat

    B = 200
    w_star_boot = []
    for _ in range(B):
        sample = filtered_df.sample(n=len(filtered_df), replace=True)
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
        work_range = np.linspace(filtered_df["Workload"].min(), filtered_df["Workload"].max(), 200)
        p_pred = 1 / (1 + np.exp(-(beta0_hat + beta1_hat * work_range)))

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.scatter(filtered_df["Workload"], filtered_df["Injury"],
                   alpha=0.4, s=40, color="#1f77b4", label="Observed Data", edgecolor="w")
        ax.plot(work_range, p_pred, color="#d62728", linewidth=2, label="Predicted Probability")
        ax.fill_between(work_range, 0, p_pred, color="#d62728", alpha=0.1)
        ax.axvline(w_star, color="#2ca02c", linestyle="--", linewidth=2,
                   label=f"Optimal Workload (w*) = {w_star:.2f}")
        ax.scatter(w_star, np.interp(w_star, work_range, p_pred), color="#2ca02c", s=80, zorder=5)
        ax.set_xlabel("Workload (Kg)", fontsize=font_size+2, weight='bold')
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
        st.markdown(
            f"For an injury probability of **{prob_target_percent}%**, "
            f"the estimated workload threshold for **{selected_gender if selected_gender != 'All' else 'all genders'}** "
            f"and **{selected_sport if selected_sport != 'All' else 'all sports'}** athletes is **{w_star:.2f} Kg**."
        )

    st.markdown("---")

    # --- Dataset Preview & Filters + Plots ---
    with st.expander("🧮 Show Raw Athlete Data"):
        st.markdown("### Filter Dataset")
        gender_options_injury = ["All"] + df_injury["Gender"].unique().tolist()
        selected_gender_injury = st.selectbox("Gender", gender_options_injury, key="gender_injury")

        sport_options_injury = ["All"] + df_injury["Sport"].unique().tolist()
        selected_sport_injury = st.selectbox("Sport", sport_options_injury, key="sport_injury")

        # Apply filters
        filtered_injury_df = df_injury.copy()
        if selected_gender_injury != "All":
            filtered_injury_df = filtered_injury_df[filtered_injury_df["Gender"] == selected_gender_injury]
        if selected_sport_injury != "All":
            filtered_injury_df = filtered_injury_df[filtered_injury_df["Sport"] == selected_sport_injury]

        # Show filtered table
        st.dataframe(filtered_injury_df)

        # Create age groups
        filtered_injury_df["AgeGroup"] = pd.cut(
            filtered_injury_df["Age"], 
            bins=[15, 20, 25, 30, 35, 40], 
            labels=["16-20","21-25","26-30","31-35","36-40"]
        )

        # ---- Row 1 ----
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📊 Injury Rate Across Age Groups")
            age_rate = filtered_injury_df.groupby("AgeGroup")["Injury"].mean().reset_index()
            fig_age, ax_age = plt.subplots(figsize=(4.5, 3))
            sns.barplot(x="AgeGroup", y="Injury", data=age_rate, color="#d62728", ax=ax_age)
            ax_age.set_ylabel("Injury Rate")
            ax_age.set_xlabel("Age Group")
            ax_age.tick_params(axis='x', rotation=45)
            fig_age.tight_layout()
            st.pyplot(fig_age)

        with col2:
            st.markdown("### 🏅 Injury Rate by Sport")
            sport_rate = filtered_injury_df.groupby("Sport")["Injury"].mean().reset_index()
            fig_sport, ax_sport = plt.subplots(figsize=(4.5, 3))
            sns.barplot(x="Sport", y="Injury", data=sport_rate, palette="Set2", ax=ax_sport)
            ax_sport.set_ylabel("Injury Rate")
            ax_sport.set_xlabel("Sport")
            ax_sport.tick_params(axis='x', rotation=45)
            fig_sport.tight_layout()
            st.pyplot(fig_sport)

        # ---- Row 2 ----
        col3, col4 = st.columns(2)

        with col3:
            st.markdown("### ⚡ Workload Distribution by Injury Status")
            filtered_injury_df["Injury_str"] = filtered_injury_df["Injury"].map({0: "No", 1: "Yes"})
            fig_workload, ax_workload = plt.subplots(figsize=(4.5, 3))
            sns.boxplot(
                x="Injury_str",
                y="Workload",
                data=filtered_injury_df,
                palette={"No": "#1f77b4", "Yes": "#d62728"},
                ax=ax_workload
            )
            ax_workload.set_xlabel("Injury")
            ax_workload.set_ylabel("Workload")
            fig_workload.tight_layout()
            st.pyplot(fig_workload)

        with col4:
            st.markdown("### 🔥 Injury Heatmap: Age × Sport")
            heatmap_data = filtered_injury_df.pivot_table(
                index="AgeGroup", columns="Sport", values="Injury", aggfunc="mean"
            )
            fig_heat, ax_heat = plt.subplots(figsize=(4.5, 3))
            sns.heatmap(heatmap_data, annot=False, fmt=".2f", cmap="Reds", linewidths=0.5, ax=ax_heat)
            ax_heat.set_ylabel("Age Group")
            ax_heat.set_xlabel("Sport")
            ax_heat.set_yticklabels(ax_heat.get_yticklabels(), rotation=45)
            fig_heat.tight_layout()
            st.pyplot(fig_heat)

# -----------------------------------------------------------
# --- PERFORMANCE PREDICTION MODEL PAGE ---
# -----------------------------------------------------------
elif page == "Performance Prediction Model":
    st.title("📈 Performance Prediction Model")
    st.markdown("Estimate next week’s performance based on training volume and workload parameters.")
    st.markdown("---")

    # --- Fixed Model Parameters ---
    tau_f = 45 / 7     # fitness time constant
    tau_d = 10 / 7     # fatigue time constant
    P0 = 0              # base performance
    k1 = 1              # fitness coefficient
    k2 = 1.5            # fatigue coefficient

    # --- Load dataset ---
    df_performance = pd.read_excel("All_data.xlsx")

    # --- Data cleaning ---
    df_performance.columns = df_performance.columns.str.strip()
    df_performance["Date"] = pd.to_datetime(df_performance["Date"], errors="coerce")
    df_performance["Load (kg)"] = pd.to_numeric(df_performance["Load (kg)"], errors="coerce").fillna(0)
    df_performance["Set"] = pd.to_numeric(df_performance["Set"], errors="coerce").fillna(0)
    df_performance["Rep"] = pd.to_numeric(df_performance["Rep"], errors="coerce").fillna(0)

    # --- Body part classification ---
    lower_list = ['Squat', 'Front Squat', 'Olympic Lift', 'Deadlift', 'RDL', 'Hip Thrust', 'Run/Walk/Sprint']
    df_performance["BodyPart"] = np.where(df_performance["Family"].isin(lower_list), "Lower", "Upper")

    # --- Sidebar filters ---
    st.sidebar.markdown("### 🔍 Filter Options")

    area = st.selectbox("Select Area", sorted(df_performance["Area"].unique()))
    name = st.selectbox("Select Name", sorted(df_performance.loc[df_performance["Area"] == area, "Name"].unique()))
    body_part = st.selectbox("Select Body Part", ["Lower", "Upper"])

    # --- Family filter ---
    available_families = sorted(df_performance.loc[
        (df_performance["Area"] == area) &
        (df_performance["Name"] == name) &
        (df_performance["BodyPart"] == body_part),
        "Family"
    ].dropna().unique())
    family = st.selectbox("Select Family", ["All"] + available_families)

    # --- Filter dataset ---
    filtered_df = df_performance[
        (df_performance["Area"] == area) &
        (df_performance["Name"] == name) &
        (df_performance["BodyPart"] == body_part)
    ]
    if family != "All":
        filtered_df = filtered_df[filtered_df["Family"] == family]

    filtered_df = filtered_df.sort_values("Date")

    # --- Handle empty results ---
    if filtered_df.empty:
        st.warning("⚠️ No records found for the selected combination.")
    else:
        # --- Section header ---
        st.markdown(f"### 🏋️ Training Sessions — {name} ({body_part} Body{'' if family == 'All' else f' – {family}'})")
        
        # --- Compute workload per session ---
        filtered_df["workload"] = filtered_df["Set"] * filtered_df["Rep"] * filtered_df["Load (kg)"]
        daily_workload = filtered_df.groupby("Date")["workload"].sum().reset_index()

        # --- Compute fitness, fatigue, performance ---
        fitness, fatigue, performance = [], [], []
        for i, w in enumerate(daily_workload["workload"]):
            if i == 0:
                f_fit, f_fat = w, w
            else:
                f_fit = fitness[-1] * np.exp(-1 / tau_f) + w
                f_fat = fatigue[-1] * np.exp(-1 / tau_d) + w
            fitness.append(f_fit)
            fatigue.append(f_fat)
            performance.append(P0 + k1 * f_fit - k2 * f_fat)
        
        daily_workload["fitness"] = fitness
        daily_workload["fatigue"] = fatigue
        daily_workload["performance"] = performance

        # --- Plot performance evolution ---
        fig, ax = plt.subplots(figsize=(5, 3))
        title_suffix = f"{body_part} Body{'' if family == 'All' else f' – {family}'}"
        ax.plot(daily_workload["Date"], daily_workload["performance"], marker="o", color="blue")
        ax.set_title(f"Training Level Evolution – {title_suffix}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Fitness level")
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, linestyle="--", alpha=0.6)
        fig.tight_layout()
        st.pyplot(fig)

        # --- Next week prediction ---
        st.markdown(f"### 🔮 Predict Next Session Performance ({title_suffix})")
        set_val = st.number_input("Set", value=4, min_value=1)
        rep_val = st.number_input("Rep", value=8, min_value=1)
        load_val = st.number_input("Load (kg)", value=20.0, min_value=0.0)
        predict = st.button("Predict Next Training Level")
        
        if predict:
            w_new = set_val * rep_val * load_val
            fitness_prev = fitness[-1]
            fatigue_prev = fatigue[-1]
            fitness_new = fitness_prev * np.exp(-1 / tau_f) + w_new
            fatigue_new = fatigue_prev * np.exp(-1 / tau_d) + w_new
            performance_new = P0 + k1 * fitness_new - k2 * fatigue_new

            perf_change = (performance_new - performance[-1]) / performance[-1] * 100 if performance[-1] != 0 else 0
            
            # --- Display results ---
            col1, col2 = st.columns([2, 1])
            with col1:
                fig2, ax2 = plt.subplots(figsize=(5, 3))
                ax2.plot(daily_workload["Date"], daily_workload["performance"], marker="o", label="Observed")
                ax2.plot(
                    [daily_workload["Date"].iloc[-1], daily_workload["Date"].iloc[-1] + pd.Timedelta(days=7)],
                    [performance[-1], performance_new],
                    color="red", marker="o", linestyle="--", label="Predicted"
                )
                ax2.legend()
                ax2.set_title(f"Training level Forecast – {title_suffix}")
                ax2.set_xlabel("Date")
                ax2.set_ylabel("Fitness level")
                ax2.grid(True, linestyle="--", alpha=0.5)
                ax2.tick_params(axis='x', rotation=45)
                fig2.tight_layout()
                st.pyplot(fig2)

            with col2:
                st.metric(label="Change (%)", value=f"{perf_change:.2f}%")
            
        # --- Dataset Preview ---
        with st.expander("🧮 Show Raw Athlete Data"):
            st.dataframe(filtered_df, use_container_width=True)



elif page == "Injury":
    st.title("🏃 Workload Risk Analysis")
    st.markdown("Explore workload distributions and identify potential overload zones.")
    st.markdown("---")

    # --- Load dataset ---
    df_injury = pd.read_excel("All_data.xlsx")

    # --- Data cleaning ---
    df_injury.columns = df_injury.columns.str.strip()
    df_injury["Date"] = pd.to_datetime(df_injury["Date"], errors="coerce")
    df_injury["Load (kg)"] = pd.to_numeric(df_injury["Load (kg)"], errors="coerce").fillna(0)
    df_injury["Set"] = pd.to_numeric(df_injury["Set"], errors="coerce").fillna(0)
    df_injury["Rep"] = pd.to_numeric(df_injury["Rep"], errors="coerce").fillna(0)


    # --- Compute Workload ---
    df_injury["Workload"] = df_injury["Set"] * df_injury["Rep"] * df_injury["Load (kg)"]
    df_injury["Workload"] = pd.to_numeric(df_injury["Workload"].astype(str).str.replace(",", "."), errors="coerce")

    # Filters
    name_options = ["All"] + sorted(df_injury["Name"].dropna().unique().tolist())
    area_options = ["All"] + sorted(df_injury["Area"].dropna().unique().tolist())

    selected_name = st.selectbox("Name", name_options)
    selected_area = st.selectbox("Area", area_options)

    filtered_df = df_injury.copy()
    if selected_name != "All":
        filtered_df = filtered_df[filtered_df["Name"] == selected_name]
    if selected_area != "All":
        filtered_df = filtered_df[filtered_df["Area"] == selected_area]

    # Compute workload stats
    mu = filtered_df["Workload"].mean()
    sigma = filtered_df["Workload"].std()

    thresholds = [mu - sigma, mu, mu + sigma]
    labels = ["Low", "Moderate", "High", "Very High"]
    filtered_df["RiskZone"] = pd.cut(filtered_df["Workload"], bins=[-np.inf] + thresholds + [np.inf], labels=labels)

    st.subheader(f"📊 Workload Distribution – {selected_area if selected_area != 'All' else 'All Areas'}")

    fig, ax = plt.subplots(figsize=(5, 3))
    sns.histplot(filtered_df["Workload"], kde=True, bins=10, color="#1f77b4", ax=ax)
    for thr in thresholds:
        ax.axvline(thr, color="red", linestyle="--", alpha=0.7)
    ax.set_title("Workload Distribution and Risk Zones")
    ax.set_xlabel("Workload")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    st.markdown(f"**Mean workload:** {mu:.2f}")
    st.markdown(f"**Standard deviation:** {sigma:.2f}")
    st.markdown("**Risk zone thresholds:**")
    st.markdown(f"- Low: < {mu - sigma:.2f}")
    st.markdown(f"- Moderate: {mu - sigma:.2f} – {mu:.2f}")
    st.markdown(f"- High: {mu:.2f} – {mu + sigma:.2f}")
    st.markdown(f"- Very High: > {mu + sigma:.2f}")










