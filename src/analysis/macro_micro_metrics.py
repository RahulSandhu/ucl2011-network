import random

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image, ImageDraw

from network_construction import df, get_network_data

# Get network data
barca_nodes, barca_edges, barca_passes = get_network_data(df, "Barcelona")
utd_nodes, utd_edges, utd_passes = get_network_data(df, "Manchester United")

# Combine all individual passes from both teams
combined_df = pd.concat(
    [
        barca_passes.assign(team="Barcelona"),
        utd_passes.assign(team="Manchester United"),
    ],
    ignore_index=True,
)

# Create networks for both teams
networks = {}
for team in ["Barcelona", "Manchester United"]:
    # Filter data for the team
    team_df = combined_df[combined_df["team"] == team].copy()

    # Create directed graph
    G = nx.DiGraph()

    # Add edges with weights
    for _, row in team_df.iterrows():
        sender, recipient = row["player"], row["pass_recipient"]
        if G.has_edge(sender, recipient):
            G[sender][recipient]["weight"] += 1
        else:
            G.add_edge(sender, recipient, weight=1)

    # Add distance for shortest path (inverse of weight)
    for u, v, d in G.edges(data=True):
        d["distance"] = 1 / d["weight"]

    # Store network
    networks[team] = {"graph": G, "df": team_df}

    print(f"\n{team}:")
    print(f"Number of players (nodes): {G.number_of_nodes()}")
    print(f"Number of unique connections (edges): {G.number_of_edges()}")
    print(f"Total passes: {len(team_df)}")

# Macro-level analysis
macro_results = []

# Compute macro-level metrics for both teams
for team in ["Barcelona", "Manchester United"]:
    # Get graph
    G = networks[team]["graph"]

    # Weighted degree metrics (Total In + Out volume)
    weighted_degrees = dict(G.degree(weight="weight"))
    max_player = max(weighted_degrees, key=weighted_degrees.get)
    min_player = min(weighted_degrees, key=weighted_degrees.get)

    # Compile metrics
    team_metrics = {
        "team": team,
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

    # Append results
    macro_results.append(team_metrics)

# Create DataFrame for macro-level results
macro_df = pd.DataFrame(macro_results)

print(macro_df.transpose())

# Micro-level analysis
micro_results = []

# Compute micro-level metrics for all players
for team in ["Barcelona", "Manchester United"]:
    # Get graph
    G = networks[team]["graph"]

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
                "team": team,
                "player": player,
                "degree_centrality": degree_cent[player],
                "betweenness_centrality": betweenness_cent[player],
                "eigenvector_centrality": eigenvector_cent[player],
            }
        )

# Convert to DataFrame
micro_df = pd.DataFrame(micro_results)

print(micro_df)

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(12, 8))

# Set overall title
fig.suptitle(
    "Player Centrality Analysis: Barcelona vs Manchester United",
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

    # Load and process player image
    img_path = player_images[player_name]

    # Open image
    img = Image.open(img_path)
    img = img.convert("RGBA")

    # Resize
    size = 80
    img = img.resize((size, size), Image.Resampling.LANCZOS)

    # Create circular mask
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, size, size), fill=255)

    # Apply mask
    output = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    output.paste(img, (0, 0))
    output.putalpha(mask)

    # Create imagebox once and reuse it
    imagebox = OffsetImage(output, zoom=0.35)

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
    imagebox2 = OffsetImage(output, zoom=0.35)

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
axes[0].set_title("Degree vs Eigenvector Centrality", fontsize=14, fontweight="bold")
axes[0].grid(alpha=0.3)
axes[0].set_xlim(x_min - x_padding, x_max + x_padding)
axes[0].set_ylim(y1_min - y1_padding, y1_max + y1_padding)

# Configure plot 2
axes[1].set_xlabel("Degree Centrality", fontsize=13, fontweight="bold")
axes[1].set_ylabel("Betweenness Centrality", fontsize=13, fontweight="bold")
axes[1].set_title("Degree vs Betweenness Centrality", fontsize=14, fontweight="bold")
axes[1].grid(alpha=0.3)
axes[1].set_xlim(x_min - x_padding, x_max + x_padding)
axes[1].set_ylim(y2_min - y2_padding, y2_max + y2_padding)

# Save and show figure
plt.tight_layout()
plt.savefig(
    "../../results/figures/scatter_plots_centrality.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()
