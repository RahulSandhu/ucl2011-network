import os
import random

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from config import PLAYER_IMAGE_PATHS, RESULTS_DIR
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from utils.node_icons import create_circular_image

from network_construction import (
    build_directed_weighted_graph,
    get_network_data,
    load_data,
)


# Calculate macro-level metrics
def calculate_macro_metrics(G, team_name):
    # Weighted degree metrics (Total In + Out volume)
    weighted_degrees = dict(G.degree(weight="weight"))
    max_player = max(weighted_degrees, key=weighted_degrees.get)
    min_player = min(weighted_degrees, key=weighted_degrees.get)

    # Compile metrics
    team_metrics = {
        "team": team_name,
        "avg_degree": sum(weighted_degrees.values()) / len(weighted_degrees),
        "max_degree": weighted_degrees[max_player],
        "max_player": max_player,
        "min_degree": weighted_degrees[min_player],
        "min_player": min_player,
        "avg_clustering_coefficient": nx.average_clustering(G, weight="weight"),
        "assortativity": nx.degree_assortativity_coefficient(G, weight="weight"),
        "avg_shortest_path_length": nx.average_shortest_path_length(
            G,
            weight="distance",
        ),
    }

    return team_metrics


# Calculate micro-level metrics
def calculate_micro_metrics(G, team_name):
    # Initialize list to store results
    micro_results = []

    # Degree Centrality
    degree_cent = nx.degree_centrality(G)

    # Betweenness
    betweenness_cent = nx.betweenness_centrality(G, weight="distance")

    # Eigenvector
    eigenvector_cent = nx.eigenvector_centrality(G, weight="weight", max_iter=1000)

    # Store results for each player
    for player in G.nodes():
        micro_results.append(
            {
                "team": team_name,
                "player": player,
                "degree_centrality": degree_cent[player],
                "betweenness_centrality": betweenness_cent[player],
                "eigenvector_centrality": eigenvector_cent[player],
            }
        )

    return micro_results


# Create scatter plots for micro-level analysis
def create_micro_plot(micro_df):
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 8))

    # Set overall title
    fig.suptitle(
        "Micro Analysis: FC Barcelona vs Manchester United",
        fontsize=18,
        fontweight="bold",
    )

    # Define team colors
    team_colors = {"Barcelona": "#004D98", "Manchester United": "#DA291C"}

    # Axis limits
    x_min = micro_df["degree_centrality"].min()
    x_max = micro_df["degree_centrality"].max()
    x_padding = (x_max - x_min) * 0.15

    y1_min = micro_df["eigenvector_centrality"].min()
    y1_max = micro_df["eigenvector_centrality"].max()
    y1_padding = (y1_max - y1_min) * 0.15

    y2_min = micro_df["betweenness_centrality"].min()
    y2_max = micro_df["betweenness_centrality"].max()
    y2_padding = (y2_max - y2_min) * 0.15

    # Set jitter amount
    jitter = 0.005

    # Process each player only once and add to both plots
    for _, player_row in micro_df.iterrows():
        # Extract player info
        player_name = player_row["player"]
        team = player_row["team"]

        # Get image path
        if player_name in PLAYER_IMAGE_PATHS:
            img_path = PLAYER_IMAGE_PATHS[player_name]
            if os.path.exists(img_path):
                # Create circular image
                circular_img = create_circular_image(img_path, size=80)

                # Create imagebox once and reuse it
                imagebox = OffsetImage(circular_img, zoom=0.35)

                # Add jitter to positions
                x_jitter1 = random.uniform(-jitter, jitter)
                y_jitter1 = random.uniform(-jitter, jitter)
                x_jitter2 = random.uniform(-jitter, jitter)
                y_jitter2 = random.uniform(-jitter, jitter)

                # Position for plot 1
                x_pos1 = player_row["degree_centrality"] + x_jitter1
                y_pos1 = player_row["eigenvector_centrality"] + y_jitter1

                # Position for plot 2
                x_pos2 = player_row["degree_centrality"] + x_jitter2
                y_pos2 = player_row["betweenness_centrality"] + y_jitter2

                # Add image to plot 1
                ab1 = AnnotationBbox(
                    imagebox,
                    (x_pos1, y_pos1),
                    frameon=True,
                    pad=0.1,
                    bboxprops=dict(
                        edgecolor=team_colors[team],
                        facecolor="white",
                        linewidth=2.5,
                        boxstyle="circle,pad=0.05",
                    ),
                )
                axes[0].add_artist(ab1)

                # Create a new imagebox for plot 2
                imagebox2 = OffsetImage(circular_img, zoom=0.35)

                # Add image to plot 2
                ab2 = AnnotationBbox(
                    imagebox2,
                    (x_pos2, y_pos2),
                    frameon=True,
                    pad=0.1,
                    bboxprops=dict(
                        edgecolor=team_colors[team],
                        facecolor="white",
                        linewidth=2.5,
                        boxstyle="circle,pad=0.05",
                    ),
                )
                axes[1].add_artist(ab2)

    # Configure plot 1
    axes[0].set_xlabel("Degree Centrality", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Eigenvector Centrality", fontsize=13, fontweight="bold")
    # axes[0].set_title("Degree vs Eigenvector Centrality", fontsize=14, fontweight="bold")
    axes[0].grid(alpha=0.3)
    axes[0].set_xlim(x_min - x_padding, x_max + x_padding)
    axes[0].set_ylim(y1_min - y1_padding, y1_max + y1_padding)

    # Configure plot 2
    axes[1].set_xlabel("Degree Centrality", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Betweenness Centrality", fontsize=13, fontweight="bold")
    # axes[1].set_title("Degree vs Betweenness Centrality", fontsize=14, fontweight="bold")
    axes[1].grid(alpha=0.3)
    axes[1].set_xlim(x_min - x_padding, x_max + x_padding)
    axes[1].set_ylim(y2_min - y2_padding, y2_max + y2_padding)

    # Save and show figure
    plt.tight_layout()
    os.makedirs(os.path.join(RESULTS_DIR, "figures"), exist_ok=True)
    plt.savefig(
        os.path.join(RESULTS_DIR, "figures", "scatter_plots_centrality.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


# Main execution
if __name__ == "__main__":
    # Load data
    df = load_data()

    # Get network data for both teams
    barca_nodes, barca_edges = get_network_data(df, "Barcelona")
    utd_nodes, utd_edges = get_network_data(df, "Manchester United")

    # Build graphs
    barca_graph = build_directed_weighted_graph(barca_edges)
    utd_graph = build_directed_weighted_graph(utd_edges)

    # Store networks
    networks = {
        "Barcelona": {"graph": barca_graph, "nodes": barca_nodes},
        "Manchester United": {"graph": utd_graph, "nodes": utd_nodes},
    }

    # Print network info
    for team in ["Barcelona", "Manchester United"]:
        # Get graph
        G = networks[team]["graph"]

        print(f"\n{team}:")
        print(f"Number of players (nodes): {G.number_of_nodes()}")
        print(f"Number of unique connections (edges): {G.number_of_edges()}")
        print(f"Total passes: {sum(d['weight'] for u, v, d in G.edges(data=True))}")

    # Macro-level analysis
    macro_results = []
    for team in ["Barcelona", "Manchester United"]:
        metrics = calculate_macro_metrics(networks[team]["graph"], team)
        macro_results.append(metrics)

    # Create DataFrame for macro-level results
    macro_df = pd.DataFrame(macro_results)
    os.makedirs(os.path.join(RESULTS_DIR, "tables"), exist_ok=True)
    macro_df.to_csv(
        os.path.join(RESULTS_DIR, "tables", "macro_level_metrics.csv"),
        index=False,
    )

    print(macro_df.transpose())

    # Micro-level analysis
    all_micro_results = []
    for team in ["Barcelona", "Manchester United"]:
        micro_metrics = calculate_micro_metrics(networks[team]["graph"], team)
        all_micro_results.extend(micro_metrics)

    # Convert to DataFrame
    micro_df = pd.DataFrame(all_micro_results)
    micro_df.to_csv(
        os.path.join(RESULTS_DIR, "tables", "micro_level_metrics.csv"),
        index=False,
    )

    print(micro_df)

    # Create figure
    create_micro_plot(micro_df)
