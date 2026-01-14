import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from config import PLAYER_IMAGE_PATHS, RESULTS_DIR
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from utils.node_icons import create_circular_image
from utils.pitch import draw_pitch

from network_construction import load_data


# Build historical passing network with failure rates
def historical_network(df, team):
    # Get top 11 players by completed passes
    mask = (df["team"] == team) & (df["type"] == "Pass") & (df["pass_outcome"].isnull())
    pass_counts = df[mask]["player"].value_counts()
    top_11 = pass_counts.nlargest(11).index.tolist()

    # Filter passes between top 11 players only
    mask = (
        (df["team"] == team)
        & (df["type"] == "Pass")
        & (df["player"].isin(top_11))
        & (df["pass_recipient"].isin(top_11))
    )
    df_passes = df[mask].copy()

    # Build directed graph with edge attributes
    G = nx.DiGraph()
    edge_data = {}

    # Calculate edge attributes for each passing connection
    for (sender, recipient), group in df_passes.groupby(["player", "pass_recipient"]):
        # Calculate weight and failure rate
        total_passes = len(group)
        completed_passes = group["pass_outcome"].isnull().sum()
        failure_rate = 1 - (completed_passes / total_passes)

        # Store edge data
        edge_data[(sender, recipient)] = {
            "weight": completed_passes,
            "failure_rate": failure_rate,
        }

    # Add edges to graph
    for (sender, recipient), data in edge_data.items():
        G.add_edge(sender, recipient, **data)

    # Calculate player positions based on average pass locations
    df_with_loc = df_passes[df_passes["location"].notna()].copy()

    # Extract x and y coordinates from location tuples
    df_with_loc["x"] = df_with_loc["location"].apply(lambda loc: loc[1])
    df_with_loc["y"] = df_with_loc["location"].apply(lambda loc: loc[0])

    # Get mean positions for each player
    player_positions = (
        df_with_loc.groupby("player")
        .agg(
            {
                "x": "mean",
                "y": "mean",
            }
        )
        .to_dict("index")
    )

    return G, player_positions


# Simulate passing sequences
def simulate_passes(G, source, target, max_passes=25):
    # Initialize simulation state
    current = source
    path = [current]
    pass_outcomes = []
    visit_count = {current: 1}

    # Main simulation loop
    for _ in range(max_passes):
        # Check if target has been reached
        if current == target:
            return path, len(path) - 1, "Success", pass_outcomes

        # Get available passing options
        neighbors = list(G.successors(current))

        # Calculate adjusted weights for each neighbor
        weights = []
        valid_neighbors = []

        # Adjust weights based on visit count to avoid loops
        for neighbor in neighbors:
            # Get base weight from graph
            base_weight = G[current][neighbor]["weight"]

            # Apply penalty for revisiting players to avoid loops
            visit_penalty = 0.3 ** visit_count.get(neighbor, 0)

            # Remove penalty if neighbor is the target
            if neighbor == target:
                visit_penalty = 1.0

            # Calculate adjusted weight
            adjusted_weight = base_weight * visit_penalty

            # Only consider neighbors with significant weight
            if adjusted_weight > 0.01:
                weights.append(adjusted_weight)
                valid_neighbors.append(neighbor)

        # Convert weights to probabilities
        total_weight = sum(weights)
        probabilities = [w / total_weight for w in weights]
        next_player = np.random.choice(valid_neighbors, p=probabilities)

        # Determine if pass succeeds based on failure rate
        failure_rate = G[current][next_player]["failure_rate"]
        pass_succeeded = failure_rate < 0.5

        # Record pass outcome
        pass_outcomes.append(
            {
                "from": current,
                "to": next_player,
                "success": pass_succeeded,
            }
        )

        # If pass failed, end simulation
        if not pass_succeeded:
            return path, len(path), "Pass failed", pass_outcomes

        # Update path and visit count
        path.append(next_player)
        visit_count[next_player] = visit_count.get(next_player, 0) + 1
        current = next_player

    return path, len(path) - 1, "Max passes reached", pass_outcomes


# Build network from simulation results
def simulation_network(df):
    # Track pass success and failure for each edge
    edge_stats = {}

    # Process each simulation result
    for idx, row in df.iterrows():
        # Get pass outcomes from simulation
        pass_outcomes = row["Pass_Outcomes"]

        # Process each pass in the sequence
        for outcome in pass_outcomes:
            # Extract sender and receiver
            sender = outcome["from"]
            receiver = outcome["to"]
            edge = (sender, receiver)

            # Initialize edge stats if not present
            if edge not in edge_stats:
                edge_stats[edge] = {"success": 0, "fail": 0}

            # Update success/failure counts
            if outcome["success"]:
                edge_stats[edge]["success"] += 1
            else:
                edge_stats[edge]["fail"] += 1

    # Build graph from edge statistics
    G = nx.DiGraph()
    for (sender, receiver), stats in edge_stats.items():
        total = stats["success"] + stats["fail"]
        success_rate = stats["success"] / total
        G.add_edge(sender, receiver, weight=total, success_rate=success_rate)

    return G


# Analyze passing sequence patterns and player involvement
def analyze_sequences(df, team_name):
    # Get successful and failed sequences
    successful = df[df["Outcome"] == "Success"]["Sequence"].tolist()
    failed = df[df["Outcome"] == "Pass failed"]["Sequence"].tolist()

    # Calculate average sequence length
    avg_length_success = df[df["Outcome"] == "Success"]["Num_Passes"].mean()
    avg_length_failed = df[df["Outcome"] == "Pass failed"]["Num_Passes"].mean()

    # Count sequences with direct passes (length <= 3)
    direct_success = len(df[(df["Outcome"] == "Success") & (df["Num_Passes"] <= 3)])
    direct_failed = len(df[(df["Outcome"] == "Pass failed") & (df["Num_Passes"] <= 3)])

    # Find most common sequence patterns (first 3 players)
    sequence_patterns = {}
    for seq in successful + failed:
        players = seq.split(" → ")
        if len(players) >= 3:
            pattern = " → ".join(players[:3])
            sequence_patterns[pattern] = sequence_patterns.get(pattern, 0) + 1

    # Sort by frequency
    common_patterns = sorted(
        sequence_patterns.items(), key=lambda x: x[1], reverse=True
    )[:10]

    # Identify "long pass" sequences (goalkeeper directly to forwards/midfielders)
    long_pass_count = 0
    defensive_players = [
        "Van der Sar",
        "Valdés",
        "Ferdinand",
        "Vidić",
        "Piqué",
        "Abidal",
        "Alves",
    ]
    attacking_players = [
        "Rooney",
        "Giggs",
        "Park",
        "Chicharito",
        "Messi",
        "Xavi",
        "Iniesta",
        "Villa",
    ]

    # Count long pass sequences
    for seq in successful + failed:
        players = seq.split(" → ")
        if len(players) >= 2:
            if players[0] in defensive_players and players[1] in attacking_players:
                long_pass_count += 1

    # Calculate player involvement
    player_stats = {}
    for _, row in df.iterrows():
        # Extract sequence and outcome
        outcome = row["Outcome"]
        sequence = row["Sequence"]
        players = sequence.split(" → ")

        # Update stats for each player in the sequence
        for player in players:
            if player not in player_stats:
                player_stats[player] = {"total": 0, "success": 0, "fail": 0}
            player_stats[player]["total"] += 1

        # Update success/failure counts
        if outcome == "Success":
            for player in players:
                player_stats[player]["success"] += 1
        elif outcome == "Pass failed":
            last_player = players[-1]
            player_stats[last_player]["fail"] += 1

    # Compile results
    results = {
        "team": team_name,
        "avg_length_success": avg_length_success,
        "avg_length_failed": avg_length_failed,
        "direct_sequences_success": direct_success,
        "direct_sequences_failed": direct_failed,
        "long_pass_sequences": long_pass_count,
        "common_patterns": common_patterns,
        "player_stats": player_stats,
    }

    return results


# Plot player involvement comparison
def plot_involvement_comparison(barca_stats, united_stats):
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6))
    fig.patch.set_facecolor("white")

    # Plot FC Barcelona
    players_barca = list(barca_stats.keys())
    if "Villa" in players_barca:
        players_barca.remove("Villa")
    success_barca = [barca_stats[p]["success"] for p in players_barca]
    fail_barca = [barca_stats[p]["fail"] for p in players_barca]
    x_pos = np.arange(len(players_barca))

    # Customize bar plot
    ax1.bar(
        x_pos,
        success_barca,
        label="Successful sequences",
        color="#28a745",
        alpha=0.8,
    )
    ax1.bar(
        x_pos,
        fail_barca,
        bottom=success_barca,
        label="Failed sequences",
        color="#dc3545",
        alpha=0.8,
    )
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(players_barca, rotation=45, ha="right")
    ax1.set_ylabel("Number of appearances", fontsize=12, fontweight="bold")
    ax1.set_title("FC Barcelona \n(Valdés → Villa)", fontsize=14, fontweight="bold")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Plot Manchester United
    players_united = list(united_stats.keys())
    if "Chicharito" in players_united:
        players_united.remove("Chicharito")
    success_united = [united_stats[p]["success"] for p in players_united]
    fail_united = [united_stats[p]["fail"] for p in players_united]
    x_pos = np.arange(len(players_united))

    # Customize bar plot
    ax2.bar(
        x_pos,
        success_united,
        label="Successful sequences",
        color="#28a745",
        alpha=0.8,
    )
    ax2.bar(
        x_pos,
        fail_united,
        bottom=success_united,
        label="Failed sequences",
        color="#dc3545",
        alpha=0.8,
    )
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(players_united, rotation=45, ha="right")
    ax2.set_ylabel("Number of appearances", fontsize=12, fontweight="bold")
    ax2.set_title(
        "Manchester United \n(Van der Sar → Chicharito)",
        fontsize=14,
        fontweight="bold",
    )
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    # Save and display figure
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    os.makedirs(os.path.join(RESULTS_DIR, "figures"), exist_ok=True)
    plt.savefig(
        os.path.join(RESULTS_DIR, "figures", "player_involvement.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


# Plot simulation networks on pitch
def plot_simulation_networks(teams_data):
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    fig.patch.set_facecolor("white")

    # Map axes to teams data
    # teams_data structure: (name, df, source, target, success_count, total_count, positions)
    plot_data = [
        (teams_data[0] + (ax1,)),
        (teams_data[1] + (ax2,)),
    ]

    # Plot each team's simulation network
    for (
        team_name,
        sim_df,
        source,
        target,
        success,
        total,
        true_positions,
        ax,
    ) in plot_data:
        # Build simulation network
        G_sim = simulation_network(sim_df)

        # Draw the pitch
        draw_pitch(ax)

        # Add subplot title
        ax.set_title(
            f"{source} → {target} \n({success}/{total} successful simulations)",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )

        # Draw edges between players
        for u, v, data in G_sim.edges(data=True):
            # Get player positions
            pos_u = true_positions[u]
            pos_v = true_positions[v]

            # Color based on success rate
            color = "#00ff00" if data["success_rate"] > 0.5 else "#ff0000"

            # Width based on number of passes
            width = 2 + (data["weight"] / 5)
            alpha = 0.6

            # Draw arrow
            ax.annotate(
                "",
                xy=(pos_v["x"], pos_v["y"]),
                xytext=(pos_u["x"], pos_u["y"]),
                arrowprops=dict(
                    arrowstyle="-|>",
                    mutation_scale=25,
                    color=color,
                    lw=width,
                    alpha=alpha,
                    connectionstyle="arc3,rad=0.1",
                ),
            )

        # Draw player nodes with images
        for node in G_sim.nodes():
            # Get player position
            pos = true_positions[node]
            x, y = pos["x"], pos["y"]

            # Calculate node importance
            in_deg = G_sim.in_degree(node)
            out_deg = G_sim.out_degree(node)
            total_deg = in_deg + out_deg

            # Set border color for source and target
            if node == source:
                border_color = "#FFD700"
            elif node == target:
                border_color = "#FF6347"
            else:
                border_color = "white"

            # Load and process player image
            if node in PLAYER_IMAGE_PATHS:
                img_path = PLAYER_IMAGE_PATHS[node]
                if os.path.exists(img_path):
                    # Use utility function for circular image
                    circular_img = create_circular_image(img_path, size=120)

                    # Scale image based on node importance
                    base_zoom = 0.25
                    zoom = base_zoom + (total_deg * 0.005)

                    # Add image to plot
                    imagebox = OffsetImage(circular_img, zoom=zoom)
                    ab = AnnotationBbox(
                        imagebox,
                        (x, y),
                        frameon=True,
                        pad=0.05,
                        bboxprops=dict(
                            edgecolor=border_color,
                            facecolor="white",
                            linewidth=3 if node in [source, target] else 1.5,
                            boxstyle="circle,pad=0.05",
                        ),
                    )
                    ax.add_artist(ab)

            # Add player name label
            ax.text(
                x,
                y - 6,
                node,
                fontsize=12,
                ha="center",
                va="top",
                fontweight="bold",
                color="black",
                zorder=10,
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.7,
                ),
            )

    # Save and display figure
    plt.tight_layout(rect=[0, 0.02, 1, 0.92])
    os.makedirs(os.path.join(RESULTS_DIR, "figures"), exist_ok=True)
    plt.savefig(
        os.path.join(RESULTS_DIR, "figures", "passing_simulation.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


# Main execution
if __name__ == "__main__":
    # Load match data
    df = load_data()

    # Set random seed for reproducibility
    np.random.seed(42)

    # Build networks for both teams
    networks = {}
    for team in ["Barcelona", "Manchester United"]:
        G, positions = historical_network(df, team)
        networks[team] = {"graph": G, "positions": positions}

    # Run FC Barcelona simulations
    barca_results = []
    G_barca = networks["Barcelona"]["graph"]
    for i in range(1, 101):
        path, num_passes, outcome, pass_outcomes = simulate_passes(
            G_barca,
            "Valdés",
            "Villa",
            max_passes=25,
        )
        sequence = " → ".join(path)
        barca_results.append(
            {
                "Outcome": outcome,
                "Sequence": sequence,
                "Num_Passes": num_passes,
                "Pass_Outcomes": pass_outcomes,
            }
        )
    barca_df = pd.DataFrame(barca_results)

    # Run Manchester United simulations
    united_results = []
    G_united = networks["Manchester United"]["graph"]
    for i in range(1, 101):
        path, num_passes, outcome, pass_outcomes = simulate_passes(
            G_united,
            "Van der Sar",
            "Chicharito",
            max_passes=25,
        )
        sequence = " → ".join(path)
        united_results.append(
            {
                "Outcome": outcome,
                "Sequence": sequence,
                "Num_Passes": num_passes,
                "Pass_Outcomes": pass_outcomes,
            }
        )
    united_df = pd.DataFrame(united_results)

    # Save simulation results
    os.makedirs(os.path.join(RESULTS_DIR, "tables"), exist_ok=True)
    barca_df.to_csv(
        os.path.join(RESULTS_DIR, "tables", "barca_simulations.csv"),
        index=False,
    )
    united_df.to_csv(
        os.path.join(RESULTS_DIR, "tables", "united_simulations.csv"),
        index=False,
    )

    # Analyze sequences and player involvement for both teams
    barca_analysis = analyze_sequences(barca_df, "Barcelona")
    united_analysis = analyze_sequences(united_df, "Manchester United")

    # Extract player stats
    barca_stats = barca_analysis["player_stats"]
    united_stats = united_analysis["player_stats"]

    # Save involvement stats
    pd.DataFrame(barca_stats).transpose().to_csv(
        os.path.join(RESULTS_DIR, "tables", "barca_player_involvement.csv"),
        index=True,
    )
    pd.DataFrame(united_stats).transpose().to_csv(
        os.path.join(RESULTS_DIR, "tables", "united_player_involvement.csv"),
        index=True,
    )

    # Create figure: Player Involvement
    plot_involvement_comparison(barca_stats, united_stats)

    # Calculate success counts
    barca_success = (barca_df["Outcome"] == "Success").sum()
    united_success = (united_df["Outcome"] == "Success").sum()

    # Prepare data for plotting networks
    teams_data = [
        (
            "Barcelona",
            barca_df,
            "Valdés",
            "Villa",
            barca_success,
            len(barca_df),
            networks["Barcelona"]["positions"],
        ),
        (
            "Manchester United",
            united_df,
            "Van der Sar",
            "Chicharito",
            united_success,
            len(united_df),
            networks["Manchester United"]["positions"],
        ),
    ]

    # Create figure: Simulation Network
    plot_simulation_networks(teams_data)

    # Save sequence analysis
    analysis_df = pd.DataFrame(
        [
            {
                "Team": "Barcelona",
                "Avg_Passes_Success": barca_analysis["avg_length_success"],
                "Avg_Passes_Failed": barca_analysis["avg_length_failed"],
                "Direct_Success": barca_analysis["direct_sequences_success"],
                "Direct_Failed": barca_analysis["direct_sequences_failed"],
                "Long_Pass_Count": barca_analysis["long_pass_sequences"],
            },
            {
                "Team": "Manchester United",
                "Avg_Passes_Success": united_analysis["avg_length_success"],
                "Avg_Passes_Failed": united_analysis["avg_length_failed"],
                "Direct_Success": united_analysis["direct_sequences_success"],
                "Direct_Failed": united_analysis["direct_sequences_failed"],
                "Long_Pass_Count": united_analysis["long_pass_sequences"],
            },
        ]
    )
    analysis_df.to_csv(
        os.path.join(RESULTS_DIR, "tables", "sequence_analysis.csv"),
        index=False,
    )
