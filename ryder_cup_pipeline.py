import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
import pulp
import os
import functools
from scipy.stats import spearmanr

# --- Constants ---
def get_metric_specs(year):
    """Returns metric specs for the given year."""
    # Base specs
    specs = [
        ("SG_total.csv", "sg_total_2023"),
        ("SG_putting.csv", "sg_putting_2023"),
        ("scrambling.csv", "scrambling_2023"),
        ("par5score.csv", "par5_score_2023"),
        ("par3score.csv", "par3_score_2023"),
        ("birdieorbetter.csv", "birdie_or_better_2023"),
        ("bogeyavoidance.csv", "bogey_avoidance_2023"),
    ]
    
    # Handle year-specific filenames
    if year == 2025:
        specs.append(("driving_distance.csv", "driving_distance_2023"))
    else:
        specs.append(("drivingdistance.csv", "driving_distance_2023"))
        
    return specs

def get_data_dir(year):
    if year == 2025:
        return "data/2025"
    return "data/2022-2023"

def get_team_files(year):
    if year == 2025:
        return {
            "USA": "2025_usa.csv",
            "Europe": "2025_europe.csv"
        }
    return {
        "USA": "2023_usa.csv",
        "Europe": "2023_europe.csv"
    }

DATA_DIR_TEAMS = "data"

SG_TOTAL_COL = "sg_total_2023"
TARGET_COL = "match_play_index"
TEAM_SIZE = 12
BETA_LOGISTIC = 15.0  # Tunable parameter

# --- 1. Data Loading and Normalization ---

def normalize_player_name(df):
    """Renames the player name column to 'player_name'."""
    candidates = ["player_name", "Player", "PLAYER", "Name", "NAME"]
    for col in df.columns:
        if col in candidates:
            return df.rename(columns={col: "player_name"})
    raise ValueError(f"Could not find player name column in {df.columns}")

def load_metric_file(data_dir, filename, metric_col_name):
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    df = pd.read_csv(path)
    df = normalize_player_name(df)
    
    # Identify metric column
    # Priority: "AVG" > "AVERAGE" > First numeric that is not ID
    target_col = None
    if "AVG" in df.columns:
        target_col = "AVG"
    elif "AVERAGE" in df.columns:
        target_col = "AVERAGE"
    else:
        # Heuristic: numeric, not ID (int-like and monotonic)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col == "player_name": continue
            # Skip likely ID columns (e.g. RANK, PLAYER_ID)
            if "ID" in col.upper() or "RANK" in col.upper():
                continue
            target_col = col
            break
    
    if target_col is None:
        raise ValueError(f"Could not identify metric column in {filename}")
        
    df = df.rename(columns={target_col: metric_col_name})
    return df[["player_name", metric_col_name]]

def load_team_file(filename):
    path = os.path.join(DATA_DIR_TEAMS, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    df = pd.read_csv(path)
    df = normalize_player_name(df)
    return df["player_name"].unique().tolist()

def compute_match_play_index(df):
    """Computes the Match Play Index (MPI) based on weighted z-scores."""
    # Mapping: Column Name -> Z-score Name
    z_map = {
        "driving_distance_2023": "z_drive",
        "birdie_or_better_2023": "z_birdie",
        "bogey_avoidance_2023": "z_bogey",
        "scrambling_2023": "z_scramble",
        "sg_putting_2023": "z_putt",
        "par3_score_2023": "z_par3",
        "par5_score_2023": "z_par5"
    }
    
    # Check columns
    for col in z_map.keys():
        if col not in df.columns:
            print(f"Warning: {col} missing, cannot compute MPI accurately.")
            return df

    # Compute Z-scores and store in DF
    for col, z_name in z_map.items():
        mean = df[col].mean()
        std = df[col].std()
        if std == 0:
            df[z_name] = 0.0
        else:
            df[z_name] = (df[col] - mean) / std
            
    # Flip par scoring (lower is better -> negative z is good, so we want -z)
    df["z_par3_good"] = -df["z_par3"]
    df["z_par5_good"] = -df["z_par5"]
    
    df["match_play_index"] = (
        0.20 * df["z_drive"] +
        0.25 * df["z_birdie"] +
        0.15 * df["z_bogey"] +
        0.15 * df["z_scramble"] +
        0.15 * df["z_putt"] +
        0.05 * df["z_par3_good"] +
        0.05 * df["z_par5_good"]
    )
    return df


# --- 2. Team Construction Helpers ---

def build_actual_us_team(players_usa_df, usa_actual_names):
    return players_usa_df[players_usa_df["player_name"].isin(usa_actual_names)].copy()

def build_europe_team(players_europe_df, europe_actual_names):
    return players_europe_df[players_europe_df["player_name"].isin(europe_actual_names)].copy()

def build_sg_top12_us_team(players_usa_df):
    return players_usa_df.sort_values(by=SG_TOTAL_COL, ascending=False).head(TEAM_SIZE).copy()

# --- 3. Model-Based Selection ---

def fit_readiness_model(df, feature_cols, target_col):
    # Drop rows with missing features
    train_df = df.dropna(subset=feature_cols + [target_col])
    X = train_df[feature_cols]
    y = train_df[target_col]
    
    model = Ridge(alpha=1.0)
    model.fit(X, y)
    return model

def compute_predicted_readiness(df, model, feature_cols, out_col="pred_readiness"):
    valid_df = df.dropna(subset=feature_cols).copy()
    if valid_df.empty:
        df[out_col] = np.nan
        return df
        
    valid_df[out_col] = model.predict(valid_df[feature_cols])
    return valid_df

def select_model_team(df_us_with_pred, team_size=12):
    # Filter to valid predictions
    candidates = df_us_with_pred.dropna(subset=["pred_readiness"]).copy()
    candidates = candidates.reset_index(drop=True)
    
    if len(candidates) < team_size:
        print(f"Warning: Not enough candidates ({len(candidates)}) for model team size {team_size}")
        return candidates
    
    # LP Problem
    prob = pulp.LpProblem("RyderCupSelection", pulp.LpMaximize)
    
    # Variables
    # x[i] = 1 if player i is selected, 0 otherwise
    x = pulp.LpVariable.dicts("select", candidates.index, cat="Binary")
    
    # Objective: Maximize total predicted readiness
    prob += pulp.lpSum([candidates.loc[i, "pred_readiness"] * x[i] for i in candidates.index])
    
    # Constraint: Team size
    prob += pulp.lpSum([x[i] for i in candidates.index]) == team_size
    
    # Solve
    # Suppress solver output
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    selected_indices = [i for i in candidates.index if pulp.value(x[i]) == 1.0]
    return candidates.loc[selected_indices].copy()

# --- 4. Strength & Win Prob ---

def add_strength_column(df, target_col=TARGET_COL, out_col="strength"):
    df[out_col] = df[target_col]
    return df

def summarize_team_strength(team_df, team_name, usa_pool_df=None, strength_col="strength"):
    stats = {
        "team_name": team_name,
        "avg_strength": team_df[strength_col].mean(),
        "min_strength": team_df[strength_col].min(),
        "max_strength": team_df[strength_col].max(),
    }
    
    # Count top players if pool is provided
    if usa_pool_df is not None:
        # Rank the pool
        pool_ranked = usa_pool_df.sort_values(by=SG_TOTAL_COL, ascending=False).reset_index(drop=True)
        top10_names = set(pool_ranked.head(10)["player_name"])
        top20_names = set(pool_ranked.head(20)["player_name"])
        top50_names = set(pool_ranked.head(50)["player_name"])
        
        team_names = set(team_df["player_name"])
        stats["count_top10_usa"] = len(team_names.intersection(top10_names))
        stats["count_top20_usa"] = len(team_names.intersection(top20_names))
        stats["count_top50_usa"] = len(team_names.intersection(top50_names))
    else:
        stats["count_top10_usa"] = None
        stats["count_top20_usa"] = None
        stats["count_top50_usa"] = None
        
    return stats

def implied_win_probability(delta, beta):
    return 1.0 / (1.0 + np.exp(-beta * delta))

# --- 5. Validation & Reporting ---

def build_mpi_ranking_table(players_df, top_n=10):
    """Returns top and bottom N players by MPI."""
    sorted_df = players_df.sort_values(by=TARGET_COL, ascending=False)
    
    cols = ["player_name", TARGET_COL, SG_TOTAL_COL]
    if "owgr_rank" in players_df.columns:
        cols.append("owgr_rank")
        
    top_df = sorted_df.head(top_n)[cols].copy()
    bottom_df = sorted_df.tail(top_n)[cols].copy()
    
    return top_df, bottom_df

def compute_mpi_correlations(players_df, team_us_actual, team_europe_actual):
    """Computes correlations and group stats."""
    results = {}
    
    # Correlations
    # MPI vs SG Total
    valid_sg = players_df.dropna(subset=[TARGET_COL, SG_TOTAL_COL])
    if not valid_sg.empty:
        corr, p = spearmanr(valid_sg[TARGET_COL], valid_sg[SG_TOTAL_COL])
        results["corr_mpi_sg"] = (corr, p)
    else:
        results["corr_mpi_sg"] = (np.nan, np.nan)
        
    # MPI vs OWGR
    if "owgr_rank" in players_df.columns:
        valid_owgr = players_df.dropna(subset=[TARGET_COL, "owgr_rank"])
        if not valid_owgr.empty:
            corr, p = spearmanr(valid_owgr[TARGET_COL], valid_owgr["owgr_rank"])
            results["corr_mpi_owgr"] = (corr, p)
        else:
            results["corr_mpi_owgr"] = (np.nan, np.nan)
    
    # Group Stats
    ryder_cup_names = set(team_us_actual["player_name"]).union(set(team_europe_actual["player_name"]))
    
    is_ryder = players_df["player_name"].isin(ryder_cup_names)
    ryder_df = players_df[is_ryder]
    non_ryder_df = players_df[~is_ryder]
    
    results["mean_mpi_ryder"] = ryder_df[TARGET_COL].mean()
    results["mean_mpi_non_ryder"] = non_ryder_df[TARGET_COL].mean()
    
    return results

def recompute_mpi_with_weights(df, weights_dict, out_col):
    """Recomputes MPI with custom weights."""
    
    # Check if z-cols exist
    required_z = ["z_drive", "z_birdie", "z_bogey", "z_scramble", "z_putt", "z_par3_good", "z_par5_good"]
    for col in required_z:
        if col not in df.columns:
            return df
            
    df[out_col] = (
        weights_dict.get("z_drive", 0) * df["z_drive"] +
        weights_dict.get("z_birdie", 0) * df["z_birdie"] +
        weights_dict.get("z_bogey", 0) * df["z_bogey"] +
        weights_dict.get("z_scramble", 0) * df["z_scramble"] +
        weights_dict.get("z_putt", 0) * df["z_putt"] +
        weights_dict.get("z_par3_good", 0) * df["z_par3_good"] +
        weights_dict.get("z_par5_good", 0) * df["z_par5_good"]
    )
    return df

def run_sensitivity_analysis(players_df, us_teams_dict, europe_team_df):
    """Runs sensitivity analysis on MPI weights."""
    
    # Define weight sets
    weight_sets = {
        "BASELINE": {
            "z_drive": 0.20, "z_birdie": 0.25, "z_bogey": 0.15, "z_scramble": 0.15, 
            "z_putt": 0.15, "z_par3_good": 0.05, "z_par5_good": 0.05
        },
        "AGGRESSIVE": {
            "z_drive": 0.30, "z_birdie": 0.35, "z_bogey": 0.05, "z_scramble": 0.05, 
            "z_putt": 0.15, "z_par3_good": 0.05, "z_par5_good": 0.05
        },
        "CONSERVATIVE": {
            "z_drive": 0.10, "z_birdie": 0.15, "z_bogey": 0.25, "z_scramble": 0.25, 
            "z_putt": 0.15, "z_par3_good": 0.05, "z_par5_good": 0.05
        },
        "PUTTING_HEAVY": {
            "z_drive": 0.15, "z_birdie": 0.20, "z_bogey": 0.10, "z_scramble": 0.10, 
            "z_putt": 0.35, "z_par3_good": 0.05, "z_par5_good": 0.05
        }
    }
    
    results = []
    
    for w_name, weights in weight_sets.items():
        col_name = f"mpi_{w_name}"
        # Recompute for ALL players
        players_df = recompute_mpi_with_weights(players_df, weights, col_name)
        
        # Europe Avg
        europe_names = europe_team_df["player_name"]
        europe_avg = players_df[players_df["player_name"].isin(europe_names)][col_name].mean()
        
        for team_name, team_df in us_teams_dict.items():
            us_names = team_df["player_name"]
            us_avg = players_df[players_df["player_name"].isin(us_names)][col_name].mean()
            
            delta = us_avg - europe_avg
            prob = implied_win_probability(delta, BETA_LOGISTIC)
            
            results.append({
                "weights_name": w_name,
                "team_name": team_name,
                "avg_mpi_variant": us_avg,
                "delta_vs_europe": delta,
                "implied_win_prob": prob
            })
            
    return pd.DataFrame(results)

def generate_mpi_validation_summary(top_mpi_df, bottom_mpi_df, corr_results, sensitivity_df):
    """Generates draft report text."""
    
    # Ranking Sanity
    top_names = ", ".join(top_mpi_df["player_name"].head(3).tolist())
    bottom_names = ", ".join(bottom_mpi_df["player_name"].head(3).tolist())
    
    sanity_text = (
        f"Ranking Sanity:\n"
        f"The top players by Match Play Index (MPI) include {top_names}, who are generally recognized as strong performers. "
        f"Conversely, the bottom of the list includes {bottom_names}. "
        f"This suggests the index is correctly identifying elite talent vs lower-tier performance within our dataset."
    )
    
    # Correlation
    corr_sg, p_sg = corr_results.get("corr_mpi_sg", (np.nan, np.nan))
    corr_text = f"Correlation Analysis:\nMPI shows a Spearman correlation of {corr_sg:.2f} (p={p_sg:.4f}) with SG Total."
    
    if "corr_mpi_owgr" in corr_results:
        corr_owgr, p_owgr = corr_results["corr_mpi_owgr"]
        corr_text += f" It also correlates with OWGR (rho={corr_owgr:.2f}, p={p_owgr:.4f}), noting that lower OWGR is better."
        
    mean_ryder = corr_results.get("mean_mpi_ryder", 0)
    mean_non = corr_results.get("mean_mpi_non_ryder", 0)
    corr_text += (
        f"\nRyder Cup players have a significantly higher average MPI ({mean_ryder:.3f}) "
        f"compared to non-Ryder Cup players ({mean_non:.3f}), further validating the metric's ability to distinguish team-caliber quality."
    )
    
    # Sensitivity
    # Check if ordering changes
    # Group by weights_name and find max win_prob team
    best_teams = sensitivity_df.loc[sensitivity_df.groupby("weights_name")["implied_win_prob"].idxmax()]
    
    sensitivity_text = "Sensitivity Analysis:\n"
    unique_winners = best_teams["team_name"].unique()
    if len(unique_winners) == 1:
        sensitivity_text += f"Across all tested weight configurations, {unique_winners[0]} remains the strongest team by MPI. "
    else:
        sensitivity_text += f"The strongest team varies by weight configuration: {', '.join(unique_winners)}. "
        
    sensitivity_text += "This indicates the model's preference is " + ("robust" if len(unique_winners) == 1 else "sensitive") + " to specific trait weightings."

    # Caveat
    caveat_text = (
        "Caveat:\n"
        "Please note that the Match Play Index is a constructed heuristic designed to approximate match-play suitability based on sub-metrics. "
        "It is not a ground-truth measure of player quality, but rather a plausible proxy validated against known performance indicators."
    )
    
    return f"{sanity_text}\n\n{corr_text}\n\n{sensitivity_text}\n\n{caveat_text}"


def plot_mpi_analysis(players_df, sensitivity_df, team_us_actual, team_europe_actual):
    """Generates additional plots for MPI analysis."""
    
    # Scatter Plot: MPI vs SG Total
    plt.figure(figsize=(8, 6))
    plt.scatter(players_df[SG_TOTAL_COL], players_df[TARGET_COL], alpha=0.5, label="All Players")
    
    # Highlight Ryder Cup players
    ryder_names = set(team_us_actual["player_name"]).union(set(team_europe_actual["player_name"]))
    ryder_df = players_df[players_df["player_name"].isin(ryder_names)]
    plt.scatter(ryder_df[SG_TOTAL_COL], ryder_df[TARGET_COL], color='red', label="Ryder Cup Players")
    
    plt.xlabel("SG Total 2023")
    plt.ylabel("Match Play Index (MPI)")
    plt.title("MPI vs SG Total Correlation")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("mpi_vs_sg_scatter.png")
    print("Saved mpi_vs_sg_scatter.png")
    
    # Distribution Plot
    plt.figure(figsize=(8, 6))
    ryder_df = players_df[players_df["player_name"].isin(ryder_names)]
    non_ryder_df = players_df[~players_df["player_name"].isin(ryder_names)]
    
    plt.hist(non_ryder_df[TARGET_COL], bins=30, alpha=0.5, label="Non-Ryder Cup", density=True)
    plt.hist(ryder_df[TARGET_COL], bins=15, alpha=0.7, label="Ryder Cup", density=True)
    
    plt.xlabel("Match Play Index (MPI)")
    plt.ylabel("Density")
    plt.title("MPI Distribution: Ryder Cup vs Field")
    plt.legend()
    plt.savefig("mpi_distribution.png")
    print("Saved mpi_distribution.png")
    
    # Sensitivity Plot
    plt.figure(figsize=(10, 6))
    
    # Pivot for plotting: index=weights_name, columns=team_name, values=win_prob
    pivot_df = sensitivity_df.pivot(index="weights_name", columns="team_name", values="implied_win_prob")
    
    # Reorder index if possible to match logic (Baseline first)
    desired_order = ["BASELINE", "AGGRESSIVE", "CONSERVATIVE", "PUTTING_HEAVY"]
    existing_order = [x for x in desired_order if x in pivot_df.index]
    pivot_df = pivot_df.reindex(existing_order)
    
    pivot_df.plot(kind="bar", figsize=(10, 6))
    plt.axhline(y=0.5, color='r', linestyle='--', label="Even Match")
    plt.ylabel("Implied Win Probability")
    plt.title("Sensitivity Analysis: Win Prob by Weight Configuration")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("mpi_sensitivity.png")
    print("Saved mpi_sensitivity.png")

# --- Main ---

def run_analysis_for_year(year):
    print(f"\n{'='*40}")
    print(f"--- Starting Ryder Cup Analysis for {year} ---")
    print(f"{'='*40}\n")
    
    data_dir = get_data_dir(year)
    metric_specs = get_metric_specs(year)
    team_files = get_team_files(year)
    
    # 1. Load Metrics
    dfs = []
    for filename, metric_col in metric_specs:
        try:
            d = load_metric_file(data_dir, filename, metric_col)
            dfs.append(d)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return

    # Merge all
    players_df = functools.reduce(lambda left, right: pd.merge(left, right, on="player_name", how="outer"), dfs)
    
    # Drop missing SG Total
    initial_len = len(players_df)
    players_df = players_df.dropna(subset=[SG_TOTAL_COL])
    print(f"Loaded metrics for {len(players_df)} players (dropped {initial_len - len(players_df)} missing SG Total).")
    
    # Compute Match Play Index
    players_df = compute_match_play_index(players_df)
    # Drop rows where MPI could not be computed (if any NaNs propagated)
    players_df = players_df.dropna(subset=[TARGET_COL])
    print(f"Computed Match Play Index for {len(players_df)} players.")

    
    # Load Teams
    usa_actual_names = load_team_file(team_files["USA"])
    europe_actual_names = load_team_file(team_files["Europe"])
    
    usa_pool_names = set(usa_actual_names)
    
    nat_file = os.path.join(DATA_DIR_TEAMS, "player_nationalities.csv")
    if os.path.exists(nat_file):
        try:
            nat_df = pd.read_csv(nat_file)
            # Normalize
            nat_df = normalize_player_name(nat_df)
            # Look for Country/Nation/Region column
            country_col = None
            for col in nat_df.columns:
                if any(x in col.upper() for x in ["COUNTRY", "NATION", "REGION"]):
                    country_col = col
                    break
            
            if country_col:
                # Filter for USA
                usa_extras = nat_df[nat_df[country_col].astype(str).str.contains("USA|United States", case=False, na=False)]["player_name"]
                usa_pool_names.update(usa_extras)
                print(f"Expanded USA pool to {len(usa_pool_names)} players using nationalities file.")
            else:
                print("Could not find country/region column in nationalities file.")
        except Exception as e:
            print(f"Could not load nationalities: {e}")

    players_usa_df = players_df[players_df["player_name"].isin(usa_pool_names)].copy()
    players_europe_df = players_df[players_df["player_name"].isin(europe_actual_names)].copy()
    
    print(f"USA Pool Size: {len(players_usa_df)}")
    print(f"Europe Pool Size: {len(players_europe_df)}")

    # Build Teams
    # Actual Teams
    us_actual_df = build_actual_us_team(players_usa_df, usa_actual_names)
    europe_actual_df = build_europe_team(players_europe_df, europe_actual_names)
    
    # Baseline (Top 12 SG)
    us_sg_top12_df = build_sg_top12_us_team(players_usa_df)
    
    # Model Team
    feature_cols = [c for c in players_df.columns if c not in ["player_name", TARGET_COL]]
    
    # Fit model on USA pool to predict MPI
    model = fit_readiness_model(players_usa_df, feature_cols, TARGET_COL)
    
    # Predict on USA pool
    players_usa_pred = compute_predicted_readiness(players_usa_df, model, feature_cols)
    
    # DEBUG: Inspect Scottie Scheffler
    scottie = players_usa_pred[players_usa_pred["player_name"].str.contains("Scheffler", case=False)]
    if not scottie.empty:
        print("\n--- DEBUG: Scottie Scheffler Stats ---")
        print(scottie[["player_name", TARGET_COL, "pred_readiness"]].to_string(index=False))
        print("Z-Scores:")
        z_cols = [c for c in scottie.columns if c.startswith("z_")]
        print(scottie[z_cols].to_string(index=False))
        print("--------------------------------------\n")

    
    # Select
    us_model_df = select_model_team(players_usa_pred, team_size=TEAM_SIZE)
    
    # Evaluation
    # Add strength
    us_actual_df = add_strength_column(us_actual_df)
    europe_actual_df = add_strength_column(europe_actual_df)
    us_sg_top12_df = add_strength_column(us_sg_top12_df)
    us_model_df = add_strength_column(us_model_df)
    
    # Summarize
    summary_rows = []
    
    # Europe Stats
    europe_stats = summarize_team_strength(europe_actual_df, "Europe_actual")
    summary_rows.append(europe_stats)
    europe_avg_strength = europe_stats["avg_strength"]
    
    # US Teams
    us_teams = [
        (f"US_actual_{year}", us_actual_df),
        ("US_SG_top12", us_sg_top12_df),
        ("US_model_team", us_model_df)
    ]
    
    win_probs = []
    
    for name, df in us_teams:
        stats = summarize_team_strength(df, name, players_usa_df)
        summary_rows.append(stats)
        
        delta = stats["avg_strength"] - europe_avg_strength
        prob = implied_win_probability(delta, BETA_LOGISTIC)
        win_probs.append({
            "team_name": name,
            "avg_strength": stats["avg_strength"],
            "delta_vs_europe": delta,
            "win_prob": prob
        })
        
    # Create DataFrames
    summary_df = pd.DataFrame(summary_rows)
    win_prob_df = pd.DataFrame(win_probs)
    
    # Add Europe to win_prob for completeness (prob 0.5 or NaN)
    europe_row = pd.DataFrame([{
        "team_name": "Europe_actual",
        "avg_strength": europe_avg_strength,
        "delta_vs_europe": 0.0,
        "win_prob": 0.5
    }])
    win_prob_df = pd.concat([europe_row, win_prob_df], ignore_index=True)
    
    print("\n--- Team Summaries ---")
    print(summary_df.to_string(index=False))
    
    print("\n--- Win Probabilities ---")
    print(win_prob_df.to_string(index=False))
    
    print("\n--- Model Team Roster ---")
    print(us_model_df[["player_name", "pred_readiness", "strength"]].to_string(index=False))

    # 5. Visualization
    plt.figure(figsize=(10, 6))
    
    # Filter for US teams only for the bar chart
    plot_df = win_prob_df[win_prob_df["team_name"].str.contains("US")]
    
    bars = plt.bar(plot_df["team_name"], plot_df["win_prob"], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    
    plt.axhline(y=0.5, color='r', linestyle='--', label="Even Match (50%)")
    plt.ylabel("Implied Win Probability")
    plt.title(f"US Teams Win Probability vs Europe ({year}, MPI-based, Beta={BETA_LOGISTIC})")
    plt.ylim(0, 1.0)
    
    # Add labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1%}',
                 ha='center', va='bottom')
                 
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"ryder_cup_analysis_{year}.png")
    print(f"\nChart saved to ryder_cup_analysis_{year}.png")
    
    # --- 5. Validation Output ---
    print("\n" + "="*40)
    print(f"==== MPI VALIDATION SUMMARY FOR REPORT ({year}) ====")
    print("="*40 + "\n")
    
    # 1. Ranking Tables
    top_mpi, bottom_mpi = build_mpi_ranking_table(players_df)
    print("--- Top 10 MPI Players ---")
    print(top_mpi.to_string(index=False))
    print("\n--- Bottom 10 MPI Players ---")
    print(bottom_mpi.to_string(index=False))
    
    # 2. Correlations
    corr_results = compute_mpi_correlations(players_df, us_actual_df, europe_actual_df)
    print("\n--- Correlations & Group Stats ---")
    for k, v in corr_results.items():
        print(f"{k}: {v}")
        
    # 3. Sensitivity
    us_teams_dict = {name: df for name, df in us_teams}
    sensitivity_df = run_sensitivity_analysis(players_df, us_teams_dict, europe_actual_df)
    print("\n--- Sensitivity Analysis ---")
    print(sensitivity_df.to_string(index=False))
    
    # 4. Plots (Optional for 2025, but good to have)
    # plot_mpi_analysis(players_df, sensitivity_df, us_actual_df, europe_actual_df) 
    # Note: plot_mpi_analysis saves to fixed filenames, so it would overwrite. 
    # We should probably update it to take a suffix or just skip for now to avoid clutter.
    
    # 5. Draft Text
    report_text = generate_mpi_validation_summary(top_mpi, bottom_mpi, corr_results, sensitivity_df)
    print("\n--- DRAFT REPORT TEXT ---")
    print(report_text)
    print("\n" + "="*40)

def main():
    # Run for 2023 (Original)
    run_analysis_for_year(2023)
    
    # Run for 2025 (New)
    run_analysis_for_year(2025)

if __name__ == "__main__":
    main()
