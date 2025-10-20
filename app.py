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
page = st.sidebar.radio("Navigate to", ["Athlete Ability", "Injury Risk Model", "Performance Prediction Model"])

# ---------------------------
# Seaborn theme
# ---------------------------
sns.set_theme(style="whitegrid")

# ---------------------------
# Small Figure Sizes & Style
# ---------------------------
fig_width = 4
fig_height = 3
font_size = 7
bar_colors = ["#2ca02c" if v >= 0 else "#d62728" for v in df_scaled[abilities].mean()]

sns.set_theme(style="whitegrid")

# ---------------------------
# Athlete Ability Page
# ---------------------------
if page == "Athlete Ability":
    st.title("🏋️ Athlete Abilities Dashboard")
    st.markdown("Explore standardized abilities of athletes and evaluate a new athlete.")
    st.markdown("---")

    # --- Filters ---
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
    st.subheader("🔎 Filter Data")
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

    # --- Row 1: Average Abilities Bar & Box Plot ---
    col1, col2 = st.columns([1,1])
    
    # Bar plot of mean abilities
    with col1:
        avg_values = filtered_df[abilities].mean()
        colors = ["#2ca02c" if v >= 0 else "#d62728" for v in avg_values]
        fig_bar, ax_bar = plt.subplots(figsize=(fig_width, fig_height))
        sns.barplot(x=avg_values.index, y=avg_values.values, palette=colors, ax=ax_bar)
        ax_bar.axhline(0, color="black", linestyle="--")
        ax_bar.set_ylabel("Z-score", fontsize=font_size)
        ax_bar.set_xlabel("Ability", fontsize=font_size)
        ax_bar.tick_params(axis='x', labelrotation=45, labelsize=font_size)
        ax_bar.tick_params(axis='y', labelsize=font_size)
        fig_bar.tight_layout()
        st.subheader("📊 Average Abilities")
        st.pyplot(fig_bar)

    # Boxplot distribution
    with col2:
        melted = filtered_df.melt(id_vars=["AthleteID"], value_vars=abilities, var_name="Ability", value_name="Z-Score")
        fig_box, ax_box = plt.subplots(figsize=(fig_width, fig_height))
        sns.boxplot(x="Ability", y="Z-Score", data=melted, palette="vlag", ax=ax_box)
        ax_box.axhline(0, color="black", linestyle="--")
        ax_box.tick_params(axis='x', labelrotation=45, labelsize=font_size)
        ax_box.tick_params(axis='y', labelsize=font_size)
        ax_box.set_xlabel("Ability", fontsize=font_size)
        fig_box.tight_layout()
        st.subheader("📦 Ability Distribution")
        st.pyplot(fig_box)

    st.markdown("---")

    # --- Row 2: Radar & Heatmap ---
    col1, col2 = st.columns([1,1])

    # Radar chart
    with col1:
        avg_values_radar = filtered_df[abilities].mean().values
        num_vars = len(abilities)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        avg_values_loop = np.concatenate((avg_values_radar, [avg_values_radar[0]]))
        angles_loop = angles + angles[:1]

        fig_radar, ax_radar = plt.subplots(figsize=(fig_width, fig_height), subplot_kw=dict(polar=True))
        ax_radar.plot(angles_loop, avg_values_loop, color="#1f77b4", linewidth=1.5)
        ax_radar.fill(angles_loop, avg_values_loop, color="#1f77b4", alpha=0.25)
        ax_radar.set_xticks(angles)
        ax_radar.set_xticklabels(abilities, fontsize=font_size-1)
        ax_radar.set_yticklabels(np.round(ax_radar.get_yticks(), 2), fontsize=font_size)
        ax_radar.axhline(0, color="grey", linestyle="--")
        fig_radar.tight_layout()
        st.subheader("📡 Overall Ability Profile")
        st.pyplot(fig_radar)

    # Heatmap of correlations
    with col2:
        corr = filtered_df[abilities].corr()
        fig_corr, ax_corr = plt.subplots(figsize=(fig_width, fig_height))
        sns.heatmap(corr, annot=True, cmap="coolwarm", linewidths=0.5, ax=ax_corr, annot_kws={"size": font_size})
        ax_corr.set_title("Correlation Between Abilities", fontsize=font_size)
        fig_corr.tight_layout()
        st.subheader("🔥 Ability Correlations")
        st.pyplot(fig_corr)

    st.markdown("---")

    # --- Evaluate a New Athlete (Expander Section) ---
    with st.expander("🏅 Evaluate a New Athlete"):
        st.markdown("Input ability levels and compare them to a reference group.")

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

        # Comparison bar chart
        avg_values_ref = ref_df[abilities].mean()
        new_values = new_athlete_scaled.iloc[0]

        st.subheader("📊 Comparison to Reference Average")
        fig_bar_eval, ax_bar_eval = plt.subplots(figsize=(fig_width, fig_height))
        x = np.arange(len(abilities))
        width = 0.35
        ax_bar_eval.bar(x - width/2, avg_values_ref, width, label=legend_label, color='#888888')
        ax_bar_eval.bar(x + width/2, new_values, width, label='New Athlete', color='#1f77b4')
        ax_bar_eval.axhline(0, color="black", linestyle="--")
        ax_bar_eval.set_xticks(x)
        ax_bar_eval.set_xticklabels(abilities, fontsize=font_size, rotation=45)
        ax_bar_eval.set_ylabel("Z-score", fontsize=font_size)
        ax_bar_eval.tick_params(axis='y', labelsize=font_size)
        ax_bar_eval.legend(fontsize=font_size)
        fig_bar_eval.tight_layout()
        st.pyplot(fig_bar_eval)

        # Boxplot overlay with athlete points
        st.subheader("📡 Percentile Placement")
        fig_box_overlay, ax_box_overlay = plt.subplots(figsize=(fig_width, fig_height))
        melted_ref = ref_df.melt(id_vars=["AthleteID"], value_vars=abilities, var_name="Ability", value_name="Z-Score")
        sns.boxplot(x="Ability", y="Z-Score", data=melted_ref, palette="vlag", ax=ax_box_overlay)
        ax_box_overlay.axhline(0, color="black", linestyle="--")
        ax_box_overlay.tick_params(axis='x', labelrotation=45, labelsize=font_size)
        ax_box_overlay.tick_params(axis='y', labelsize=font_size)
        ax_box_overlay.set_ylabel("Z-score", fontsize=font_size)
        ax_box_overlay.set_xlabel("Ability", fontsize=font_size)

        for i, ability in enumerate(abilities):
            ax_box_overlay.scatter(
                i, new_values[ability], color="#1f77b4", s=50, zorder=10,
                label="New Athlete" if i == 0 else ""
            )

        # Only show legend once
        handles, labels = ax_box_overlay.get_legend_handles_labels()
        if handles:
            ax_box_overlay.legend(handles=[handles[0]], labels=[labels[0]], fontsize=font_size)

        fig_box_overlay.tight_layout()
        st.pyplot(fig_box_overlay)


# ---------------------------
# Injury Risk Model Page
# ---------------------------
elif page == "Injury Risk Model":
    st.title("🏃 Individual Training Parameter Risk Analysis")
    st.markdown("Explore risk zones for Set, Rep, and Load (kg) individually.")
    st.markdown("---")

    # --- Load dataset ---
    df_injury = pd.read_excel("All_data.xlsx")

    # --- Data cleaning ---
    df_injury.columns = df_injury.columns.str.strip()
    df_injury["Date"] = pd.to_datetime(df_injury["Date"], errors="coerce")

    # Convert numeric columns safely
    for col in ["Set", "Rep", "Load (kg)"]:
        if col in df_injury.columns:
            df_injury[col] = (
                pd.to_numeric(df_injury[col].astype(str).str.replace(",", "."), errors="coerce")
                .fillna(1)
                .round(0)
            )
        else:
            df_injury[col] = 1

    # --- Dynamic Filters ---
    st.subheader("🔎 Filter Data")

    name_options = ["All"] + sorted(df_injury["Name"].dropna().unique().tolist())
    selected_name = st.selectbox("Select Name", name_options)

    temp_df = df_injury.copy()
    if selected_name != "All":
        temp_df = temp_df[temp_df["Name"] == selected_name]

    area_options = ["All"] + sorted(temp_df["Area"].dropna().unique().tolist())
    selected_area = st.selectbox("Select Area", area_options)

    temp_df2 = temp_df.copy()
    if selected_area != "All":
        temp_df2 = temp_df2[temp_df2["Area"] == selected_area]

    family_options = ["All"] + sorted(temp_df2["Family"].dropna().unique().tolist())
    selected_family = st.selectbox("Select Family", family_options)

    # --- Apply Filters ---
    filtered_df = df_injury.copy()
    if selected_name != "All":
        filtered_df = filtered_df[filtered_df["Name"] == selected_name]
    if selected_area != "All":
        filtered_df = filtered_df[filtered_df["Area"] == selected_area]
    if selected_family != "All":
        filtered_df = filtered_df[filtered_df["Family"] == selected_family]

    st.subheader(f"📊 Training Parameter Risk Zones – {selected_area if selected_area != 'All' else 'All Areas'}")

    # --- Function to compute zones (based on max value + 20–40%) ---
    def compute_zones(series):
        series = pd.to_numeric(series, errors="coerce").dropna()
        if series.empty:
            return pd.Series(["No data"] * len(series)), [0, 0, 0, 0]

        mu = series.mean()
        max_val = series.max()

        low = max(mu * 0.5, 0)
        moderate = mu
        high = max_val * 1.20
        very_high = max_val * 1.40

        thresholds = [round(low), round(moderate), round(high), round(very_high)]
        labels = ["Low", "Moderate", "High", "Very High"]

        thresholds = sorted(list(set(thresholds)))
        while len(thresholds) < 4:
            thresholds.append(thresholds[-1] + 1)

        try:
            zone_series = pd.cut(series, bins=[-np.inf] + thresholds + [np.inf], labels=labels)
        except ValueError:
            zone_series = pd.Series(["Undefined"] * len(series))

        return zone_series, thresholds

    # --- Compute results for each training parameter ---
    summary_rows = []
    for var, colname in [("Set", "Set"), ("Rep", "Rep"), ("Load (kg)", "Load (kg)")]:
        filtered_df[f"{colname}_Zone"], thr = compute_zones(filtered_df[colname])

        summary = filtered_df[colname].describe().round(0)
        mean, std, minv, maxv, count = summary["mean"], summary["std"], summary["min"], summary["max"], int(summary["count"])

        summary_rows.append({
            "Parameter": var,
            "Observations": count,
            "Mean": int(mean),
            "Std Dev": int(std),
            "Min": int(minv),
            "Max": int(maxv),
            "🟩 Low": f"< {thr[0]}",
            "🟨 Moderate": f"{thr[0]} – {thr[1]}",
            "🟧 High": f"{thr[1]} – {thr[2]}",
            "🟥 Very High": f"> {thr[2]}"
        })

    summary_df = pd.DataFrame(summary_rows)

    # --- Style table with HTML ---
    def colorize_zone(zone):
        if "Low" in zone:
            color = "#d4edda"  # light green
        elif "Moderate" in zone:
            color = "#fff3cd"  # light yellow
        elif "High" in zone:
            color = "#ffe5b4"  # light orange
        else:
            color = "#f8d7da"  # light red
        return f"background-color: {color}; border-radius: 6px; padding: 4px;"

    styled_html = summary_df.to_html(index=False, escape=False).replace(
        '🟩 Low', f'<span style="{colorize_zone("Low")}">🟩 Low</span>'
    ).replace(
        '🟨 Moderate', f'<span style="{colorize_zone("Moderate")}">🟨 Moderate</span>'
    ).replace(
        '🟧 High', f'<span style="{colorize_zone("High")}">🟧 High</span>'
    ).replace(
        '🟥 Very High', f'<span style="{colorize_zone("Very High")}">🟥 Very High</span>'
    )

    st.markdown("### 📈 Risk Threshold Overview")
    st.markdown(styled_html, unsafe_allow_html=True)

    st.markdown("""
    **Interpretation:**
    - 🟩 **Low:** Light or recovery training intensity.  
    - 🟨 **Moderate:** Normal and sustainable load.  
    - 🟧 **High:** Fatigue accumulation likely; monitor closely.  
    - 🟥 **Very High:** Potential overload or injury risk.  
    """)

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
    st.subheader("🔎 Filter Data")

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



















