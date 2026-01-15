import os

import matplotlib.pyplot as plt
import pandas as pd
from config import RESULTS_DIR

from analysis.network_construction import get_network_data, load_data


# Analyze individual player impact
def analyze_individual_player_impact(team_name, team_nodes, df):

    # Helper function to calculate pass completion rate
    def calculate_pass_completion_rate(excluded_player=None):
        # Filter for team passes
        team_passes = df[(df["team"] == team_name) & (df["type"] == "Pass")].copy()

        # Exclude passes involving the removed player
        if excluded_player:
            team_passes = team_passes[
                (team_passes["player"] != excluded_player)
                & (team_passes["pass_recipient"] != excluded_player)
            ]

        # Calculate completion rate
        completed_passes = team_passes["pass_outcome"].isnull().sum()
        completion_rate = (completed_passes / len(team_passes)) * 100

        return completion_rate

    # Helper function to get pass height distribution for a player
    def get_player_pass_height_distribution(player):
        # Get all passes by this player
        player_passes = df[
            (df["team"] == team_name)
            & (df["type"] == "Pass")
            & (df["player"] == player)
        ]

        # Count pass heights
        height_counts = player_passes["pass_height"].value_counts()
        total_passes = len(player_passes)

        # Calculate percentages
        distribution = {
            "Ground Pass": (height_counts.get("Ground Pass", 0) / total_passes) * 100,
            "Low Pass": (height_counts.get("Low Pass", 0) / total_passes) * 100,
            "High Pass": (height_counts.get("High Pass", 0) / total_passes) * 100,
        }

        return distribution

    # Calculate baseline
    baseline_rate = calculate_pass_completion_rate(excluded_player=None)

    # Store results
    results = []

    # Calculate impact for each player
    for player in team_nodes.index:
        # Calculate rate without this player
        rate_without = calculate_pass_completion_rate(excluded_player=player)

        # Calculate percentage change
        if baseline_rate > 0:
            pct_change = ((rate_without - baseline_rate) / baseline_rate) * 100
        else:
            pct_change = 0.0

        # Get pass height distribution
        pass_dist = get_player_pass_height_distribution(player)

        # Save results
        results.append(
            {
                "team": team_name,
                "player": player,
                "baseline_completion": baseline_rate,
                "completion_without_player": rate_without,
                "percentage_change": pct_change,
                "absolute_change": rate_without - baseline_rate,
                "ground_pass_pct": pass_dist["Ground Pass"],
                "low_pass_pct": pass_dist["Low Pass"],
                "high_pass_pct": pass_dist["High Pass"],
            }
        )

    return pd.DataFrame(results)


# Plot player impact comparison
def plot_player_impact(barca_results, utd_results):
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))
    fig.set_facecolor("white")

    # Overall title
    # fig.suptitle(
    #     "Individual Player Impact on Pass Completion Rate",
    #     fontsize=24,
    #     weight="bold",
    #     y=0.98,
    # )

    # Pass height colors
    pass_colors = {
        "Ground Pass": "#2ecc71",
        "Low Pass": "#f39c12",
        "High Pass": "#3498db",
    }

    # Sort by impact
    barca_sorted = barca_results.sort_values("percentage_change")
    utd_sorted = utd_results.sort_values("percentage_change")

    # FC Barcelona plot
    y_pos = range(len(barca_sorted))

    # Create stacked bars for each pass height
    ground_vals = []
    low_vals = []
    high_vals = []

    # Calculate values for each pass height
    for _, row in barca_sorted.iterrows():
        # Total percentage change
        total = row["percentage_change"]
        ground_pct = row["ground_pass_pct"] / 100
        low_pct = row["low_pass_pct"] / 100
        high_pct = row["high_pass_pct"] / 100

        # Calculate contributions
        ground_vals.append(total * ground_pct)
        low_vals.append(total * low_pct)
        high_vals.append(total * high_pct)

    # Plot stacked bars
    ax1.barh(
        y_pos,
        ground_vals,
        color=pass_colors["Ground Pass"],
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
        label="Ground Pass",
    )
    ax1.barh(
        y_pos,
        low_vals,
        left=ground_vals,
        color=pass_colors["Low Pass"],
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
        label="Low Pass",
    )

    # Calculate cumulative for high passes
    low_cumsum = [g + l for g, l in zip(ground_vals, low_vals)]
    ax1.barh(
        y_pos,
        high_vals,
        left=low_cumsum,
        color=pass_colors["High Pass"],
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
        label="High Pass",
    )

    # Customize axes
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(barca_sorted["player"], fontsize=10, weight="bold")
    ax1.set_xlabel(
        "Percentage Change in Pass Completion (%)",
        fontsize=12,
        weight="bold",
    )
    ax1.set_title(
        f"FC Barcelona, \nBaseline: {barca_sorted['baseline_completion'].iloc[0]:.1f}%",
        fontsize=16,
        weight="bold",
        pad=15,
    )
    ax1.axvline(x=0, color="black", linestyle="--", linewidth=1.5, alpha=0.5)
    ax1.grid(True, alpha=0.3, axis="x")
    ax1.legend(loc="best", fontsize=9, frameon=True, fancybox=True)
    x_min = barca_sorted["percentage_change"].min()
    x_max = barca_sorted["percentage_change"].max()
    x_range = x_max - x_min
    ax1.set_xlim(x_min - 0.2 * x_range, x_max + 0.2 * x_range)

    # Add value labels on bars
    for i, (idx, row) in enumerate(barca_sorted.iterrows()):
        # Determine label position and text
        value = row["percentage_change"]
        label_x = value - 0.1 if value < 0 else value + 0.1
        label_ha = "right" if value < 0 else "left"

        # Add plus sign for positive values
        if value > 0:
            label_text = f"+{value:.2f}%"
        else:
            label_text = f"{value:.2f}%"

        # Add text label
        ax1.text(
            label_x,
            i,
            label_text,
            va="center",
            ha=label_ha,
            fontsize=9,
            weight="bold",
        )

    # Manchester United plot
    y_pos = range(len(utd_sorted))

    # Create stacked bars for each pass height
    ground_vals = []
    low_vals = []
    high_vals = []

    # Calculate values for each pass height
    for _, row in utd_sorted.iterrows():
        # Total percentage change
        total = row["percentage_change"]
        ground_pct = row["ground_pass_pct"] / 100
        low_pct = row["low_pass_pct"] / 100
        high_pct = row["high_pass_pct"] / 100

        # Calculate contributions
        ground_vals.append(total * ground_pct)
        low_vals.append(total * low_pct)
        high_vals.append(total * high_pct)

    # Plot stacked bars
    ax2.barh(
        y_pos,
        ground_vals,
        color=pass_colors["Ground Pass"],
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
        label="Ground Pass",
    )
    ax2.barh(
        y_pos,
        low_vals,
        left=ground_vals,
        color=pass_colors["Low Pass"],
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
        label="Low Pass",
    )

    # Calculate cumulative for high passes
    low_cumsum = [g + l for g, l in zip(ground_vals, low_vals)]
    ax2.barh(
        y_pos,
        high_vals,
        left=low_cumsum,
        color=pass_colors["High Pass"],
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
        label="High Pass",
    )

    # Customize axes
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(utd_sorted["player"], fontsize=10, weight="bold")
    ax2.set_xlabel(
        "Percentage Change in Pass Completion (%)",
        fontsize=12,
        weight="bold",
    )
    ax2.set_title(
        f"Manchester United, \nBaseline: {utd_sorted['baseline_completion'].iloc[0]:.1f}%",
        fontsize=16,
        weight="bold",
        pad=15,
    )
    ax2.axvline(x=0, color="black", linestyle="--", linewidth=1.5, alpha=0.5)
    ax2.grid(True, alpha=0.3, axis="x")
    ax2.legend(loc="best", fontsize=9, frameon=True, fancybox=True)
    x_min = utd_sorted["percentage_change"].min()
    x_max = utd_sorted["percentage_change"].max()
    x_range = x_max - x_min
    ax2.set_xlim(x_min - 0.15 * x_range, x_max + 0.15 * x_range)

    # Add value labels on bars
    for i, (idx, row) in enumerate(utd_sorted.iterrows()):
        # Determine label position and text
        value = row["percentage_change"]
        label_x = value - 0.1 if value < 0 else value + 0.1
        label_ha = "right" if value < 0 else "left"

        # Add plus sign for positive values
        if value > 0:
            label_text = f"+{value:.2f}%"
        else:
            label_text = f"{value:.2f}%"

        # Add text label
        ax2.text(
            label_x,
            i,
            label_text,
            va="center",
            ha=label_ha,
            fontsize=9,
            weight="bold",
        )

    # Adjust layout
    plt.tight_layout()

    # Save figure
    os.makedirs(os.path.join(RESULTS_DIR, "figures"), exist_ok=True)
    plt.savefig(
        os.path.join(RESULTS_DIR, "figures", "individual_player_impact.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


# Main execution
if __name__ == "__main__":
    # Load data
    df = load_data()

    # Get network data
    barca_nodes, barca_edges = get_network_data(df, "Barcelona")
    utd_nodes, utd_edges = get_network_data(df, "Manchester United")

    # Analyze individual player impact
    barca_results = analyze_individual_player_impact("Barcelona", barca_nodes, df)
    utd_results = analyze_individual_player_impact("Manchester United", utd_nodes, df)
    plot_player_impact(barca_results, utd_results)

    # Save results
    combined_results = pd.concat([barca_results, utd_results], ignore_index=True)
    os.makedirs(os.path.join(RESULTS_DIR, "tables"), exist_ok=True)
    combined_results.to_csv(
        os.path.join(RESULTS_DIR, "tables", "individual_player_impact_results.csv"),
        index=False,
    )
