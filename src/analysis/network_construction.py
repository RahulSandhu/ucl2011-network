# https://mplsoccer.readthedocs.io/en/latest/gallery/pitch_plots/plot_pass_network.html
# https://soccermatics.readthedocs.io/en/latest/gallery/lesson1/plot_PassNetworks.html
# https://www.youtube.com/watch?v=pW7rltisoqo
# https://www.youtube.com/watch?v=ZOC4DSHiKVU

import os
from ast import literal_eval

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from config import DATA_DIR, PLAYER_IMAGE_PATHS, PLAYER_NAME_MAP, RESULTS_DIR
from utils.node_icons import add_image_node
from utils.pitch import draw_pitch


# Load and prepare data
def load_data():
    # Load event data
    df = pd.read_csv(os.path.join(DATA_DIR, "barca_manutd_2011_events.csv"))

    # Parse location column
    df["location"] = df["location"].apply(
        lambda x: literal_eval(str(x)) if pd.notnull(x) else None
    )

    # Parse pass end location column
    df["pass_end_location"] = df["pass_end_location"].apply(
        lambda x: literal_eval(str(x)) if pd.notnull(x) else None
    )

    # Map player names
    df["player"] = df["player"].map(PLAYER_NAME_MAP).fillna(df["player"])

    # Map pass recipient names
    df["pass_recipient"] = (
        df["pass_recipient"].map(PLAYER_NAME_MAP).fillna(df["pass_recipient"])
    )

    return df


# Get nodes and edges for a team
def get_network_data(df, team):
    # Get min and max minutes from the data
    min_minute = df["minute"].min()
    max_minute = df["minute"].max()

    # Filter data for team for full match
    mask = (
        (df["team"] == team)
        & (df["minute"] >= min_minute)
        & (df["minute"] <= max_minute)
    )
    df_slice = df[mask].copy()

    # Filter for completed passes only
    df_passes = df_slice[
        (df_slice["type"] == "Pass") & (df_slice["pass_outcome"].isnull())
    ].copy()

    # Check if there are any passes
    if df_passes.empty:
        return None, None

    # Extract coordinates
    df_passes["x_plot"] = df_passes["location"].apply(lambda x: x[1])
    df_passes["y_plot"] = df_passes["location"].apply(lambda x: x[0])

    # Nodes
    avg_loc = df_passes.groupby("player").agg(
        {
            "x_plot": ["mean"],
            "y_plot": ["mean", "count"],
        }
    )
    avg_loc.columns = ["x", "y", "count"]

    # Select top 11 players by pass count
    avg_loc = avg_loc.nlargest(11, "count")

    # Get list of top 11 players
    valid_players = avg_loc.index.tolist()

    # Filter passes to only include valid players
    df_passes_filtered = df_passes[
        (df_passes["player"].isin(valid_players))
        & (df_passes["pass_recipient"].isin(valid_players))
    ].copy()

    # Edges
    pass_between = (
        df_passes_filtered.groupby(["player", "pass_recipient"])
        .agg({"id": "count"})
        .reset_index()
    )
    pass_between.rename(columns={"id": "pass_count"}, inplace=True)

    # Merge locations for drawing arrows
    pass_between = pass_between.merge(avg_loc, left_on="player", right_index=True)
    pass_between = pass_between.rename(columns={"x": "x_start", "y": "y_start"})
    pass_between = pass_between.merge(
        avg_loc,
        left_on="pass_recipient",
        right_index=True,
    )
    pass_between = pass_between.rename(columns={"x": "x_end", "y": "y_end"})

    return avg_loc, pass_between


# Build directed weighted graph from edges
def build_directed_weighted_graph(edges):
    # Check if edges is valid
    if edges is None or edges.empty:
        return None

    # Build directed graph
    G = nx.DiGraph()

    # Add edges with weights
    for _, row in edges.iterrows():
        # Extract sender, recipient, and weight
        sender = row["player"]
        recipient = row["pass_recipient"]
        weight = row["pass_count"]

        # Add edge with weight
        if G.has_edge(sender, recipient):
            G[sender][recipient]["weight"] += weight
        else:
            G.add_edge(sender, recipient, weight=weight)

    # Add distance for shortest path
    for u, v, d in G.edges(data=True):
        d["distance"] = 1 / d["weight"]

    return G


# Create figure for a team
def create_team_figure(nodes, edges, edge_color, title_main, fig_name):
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 14))
    fig.set_facecolor("white")
    fig.suptitle(
        title_main,
        fontsize=22,
        color="black",
        weight="bold",
        y=0.98,
    )

    # Subtitle
    fig.text(
        0.5,
        0.94,
        "Data provided by StatsBomb (https://statsbomb.com)",
        ha="center",
        fontsize=11,
        color="black",
        style="italic",
    )

    # Draw pitch
    draw_pitch(ax)

    # Draw network
    if edges is not None and not edges.empty and nodes is not None and not nodes.empty:
        # Draw edges
        for _, row in edges.iterrows():
            # Determine width and alpha based on pass count
            pass_cnt = row["pass_count"]
            width = 0.5 + (pass_cnt / 6)
            alpha = 0.4 if pass_cnt < 2 else 0.8

            # Draw arrow
            ax.annotate(
                "",
                xy=(row["x_end"], row["y_end"]),
                xytext=(row["x_start"], row["y_start"]),
                arrowprops=dict(
                    arrowstyle="-",
                    color=edge_color,
                    connectionstyle="arc3,rad=0.1",
                    linewidth=width,
                    alpha=alpha,
                ),
            )

        # Draw player images as nodes
        for player_name, row in nodes.iterrows():
            # Get image path
            img_path = PLAYER_IMAGE_PATHS[player_name]

            # Calculate zoom based on degree
            base_zoom = 0.3
            zoom = base_zoom + (row["count"] / 200)

            # Add player image node
            add_image_node(ax, img_path, row["x"], row["y"], zoom, edge_color)

            # Draw player name below image
            ax.text(
                row["x"],
                row["y"] - 4,
                player_name,
                fontsize=9,
                ha="center",
                va="top",
                color="black",
                weight="bold",
                zorder=11,
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.8,
                ),
            )

    # Adjust layout and save figure
    plt.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.02)
    os.makedirs(os.path.join(RESULTS_DIR, "figures"), exist_ok=True)
    plt.savefig(
        os.path.join(RESULTS_DIR, "figures", fig_name),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


# Main execution
if __name__ == "__main__":
    # Load data
    df = load_data()

    # Get Barcelona network data
    barca_nodes, barca_edges = get_network_data(df, "Barcelona")

    # Build Barcelona graph
    barca_graph = build_directed_weighted_graph(barca_edges)

    # Create Barcelona figure
    create_team_figure(
        barca_nodes,
        barca_edges,
        edge_color="#f2e691",
        title_main="FC Barcelona Passing Network",
        fig_name="barcelona_passing_network.png",
    )

    # Get Manchester United network data
    utd_nodes, utd_edges = get_network_data(df, "Manchester United")

    # Build Manchester United graph
    utd_graph = build_directed_weighted_graph(utd_edges)

    # Create Manchester United figure
    create_team_figure(
        utd_nodes,
        utd_edges,
        edge_color="black",
        title_main="Manchester United Passing Network",
        fig_name="manchester_united_passing_network.png",
    )
