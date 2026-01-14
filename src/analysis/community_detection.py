import os

import graph_tool.all as gt
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from config import DATA_DIR, PLAYER_IMAGE_PATHS, PLAYER_NAME_MAP, RESULTS_DIR
from infomap import Infomap
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from network_construction import (
    build_directed_weighted_graph,
    get_network_data,
    load_data,
)

from utils.node_icons import create_circular_image
from utils.pitch import draw_pitch


# Convert directed graph to undirected with weight summing
def to_undirected_weighted(G):
    # Create undirected graph
    Gu = nx.Graph()

    # Add nodes with attributes
    for n, attrs in G.nodes(data=True):
        Gu.add_node(n, **attrs)

    # Add edges with summed weights
    for u, v, attrs in G.edges(data=True):
        w = float(attrs.get("weight", 1.0))
        if Gu.has_edge(u, v):
            Gu[u][v]["weight"] += w
        else:
            Gu.add_edge(u, v, weight=w)

    return Gu


# Convert NetworkX graph to graph-tool for SBM
def nx_to_graphtool(Gu):
    # Generate graph-tool
    g = gt.Graph(directed=False)

    # Create vertex property for labels
    v_label = g.new_vertex_property("string")
    g.vertex_properties["label"] = v_label

    # Map node labels to vertex indices
    node_index = {}

    # Add vertices
    for n in Gu.nodes():
        v = g.add_vertex()
        v_label[v] = str(n)
        node_index[str(n)] = int(v)

    # Create edge property for weights
    e_weight = g.new_edge_property("double")
    g.edge_properties["weight"] = e_weight

    # Add edges with weights
    for u, v, attrs in Gu.edges(data=True):
        e = g.add_edge(g.vertex(node_index[str(u)]), g.vertex(node_index[str(v)]))
        e_weight[e] = float(attrs.get("weight", 1.0))

    return g, node_index


# Community detection functions
def detect_communities(G, method="louvain", seed=42):
    # Convert to undirected for community detection
    Gu = to_undirected_weighted(G)
    communities = []

    # Detect communities based on method
    if method == "louvain":
        communities = nx.community.louvain_communities(
            Gu,
            weight="weight",
            seed=seed,
        )
    elif method == "greedy":
        communities = nx.community.greedy_modularity_communities(
            Gu,
            weight="weight",
        )
    elif method == "infomap":
        # Use actual Infomap library
        im = Infomap(silent=True, two_level=True, seed=seed)

        # Map node labels to integers
        node_to_id = {n: i for i, n in enumerate(Gu.nodes())}
        id_to_node = {i: n for n, i in node_to_id.items()}

        # Add weighted edges
        for u, v, attrs in Gu.edges(data=True):
            w = float(attrs.get("weight", 1.0))
            im.add_link(node_to_id[u], node_to_id[v], w)

        # Run Infomap
        im.run()

        # Extract communities
        module_to_nodes = {}
        for node in im.nodes:
            mod = int(node.module_id)
            module_to_nodes.setdefault(mod, set()).add(id_to_node[int(node.node_id)])
        communities = list(module_to_nodes.values())
    elif method == "sbm":
        # Use graph-tool SBM
        g, _ = nx_to_graphtool(Gu)

        # Set seed
        gt.seed_rng(seed)

        # Run SBM with MDL
        state = gt.minimize_blockmodel_dl(
            g,
            state_args={
                "recs": [g.ep.weight],
                "rec_types": ["real-exponential"],
            },
        )

        # Extract community assignments
        blocks = state.get_blocks()

        # Convert to communities
        module_to_nodes = {}
        for v in g.vertices():
            mod = int(blocks[v])
            label = g.vp.label[v]
            module_to_nodes.setdefault(mod, set()).add(label)
        communities = list(module_to_nodes.values())

    # Create node to community mapping
    node_community = {}
    for idx, comm in enumerate(communities):
        for node in comm:
            node_community[node] = idx

    return node_community, len(communities)


# Function to create community detection figure
def create_community_figure(team_name, method, edge_color, title_main, fig_name):
    # Get network and nodes
    G = networks[team_name]["graph"]
    nodes = barca_nodes if team_name == "Barcelona" else utd_nodes
    edges = barca_edges if team_name == "Barcelona" else utd_edges

    # Detect communities
    node_community, n_communities = detect_communities(G, method)
    community_colors = [cm.Set1((i * 3) % 9 / 9) for i in range(n_communities)]

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
        0.93,
        f"Community Detection: {method.upper()}",
        ha="center",
        fontsize=11,
        color="black",
        style="italic",
    )

    # Draw pitch
    draw_pitch(ax)

    # Add legend for communities
    from matplotlib.patches import Patch

    # Create legend elements
    legend_elements = [
        Patch(
            facecolor=community_colors[i], edgecolor="black", label=f"Community {i+1}"
        )
        for i in range(n_communities)
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper right",
        fontsize=9,
        title="Communities",
        title_fontsize=10,
        frameon=True,
        fancybox=True,
        shadow=True,
        framealpha=0.9,
    )

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

        # Draw player images as nodes with community colors
        for player_name, row in nodes.iterrows():
            # Get image path
            img_path = PLAYER_IMAGE_PATHS[player_name]

            # Create circular image
            circular_img = create_circular_image(img_path, size=80)

            # Calculate zoom based on degree
            base_zoom = 0.3
            zoom = base_zoom + (row["count"] / 200)

            # Get community color
            comm_id = node_community.get(player_name, 0)
            border_color = community_colors[comm_id]

            # Create image box
            imagebox = OffsetImage(circular_img, zoom=zoom)

            # Add image to plot
            ab = AnnotationBbox(
                imagebox,
                (row["x"], row["y"]),
                frameon=True,
                pad=0.1,
                boxcoords="data",
                box_alignment=(0.5, 0.5),
                bboxprops=dict(
                    edgecolor=border_color,
                    facecolor="white",
                    linewidth=5,
                    boxstyle="circle,pad=0.05",
                ),
            )
            ax.add_artist(ab)

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


# Community efficiency analysis
def analyze_community_efficiency(team_name, node_community):
    # Load original full dataframe with all columns
    df_full = pd.read_csv(os.path.join(DATA_DIR, "barca_manutd_2011_events.csv"))
    df_full["player"] = df_full["player"].map(PLAYER_NAME_MAP).fillna(df_full["player"])

    # Filter events for this team
    team_df = df_full[df_full["team"] == team_name].copy()

    # Get unique communities
    communities = sorted(set(node_community.values()))

    # Initialize results list
    results = []

    # Analyze each community
    for comm_id in communities:
        # Get players in this community
        comm_players = [
            player for player, cid in node_community.items() if cid == comm_id
        ]

        # Filter events for players in this community
        comm_events = team_df[team_df["player"].isin(comm_players)].copy()

        # Passing efficiency
        passes = comm_events[comm_events["type"] == "Pass"].copy()
        total_passes = len(passes)
        completed_passes = passes["pass_outcome"].isnull().sum()
        pass_success_rate = (
            (completed_passes / total_passes * 100) if total_passes > 0 else 0
        )
        avg_pass_length = (
            passes["pass_length"].mean() if "pass_length" in passes.columns else 0
        )

        # Offensive contributions
        shots = comm_events[comm_events["type"] == "Shot"].copy()
        total_shots = len(shots)
        total_xg = (
            shots["shot_statsbomb_xg"].sum()
            if "shot_statsbomb_xg" in shots.columns
            else 0
        )

        # Defensive contributions
        duels = comm_events[comm_events["type"] == "Duel"].copy()
        total_duels = len(duels)
        won_duels = (
            duels[duels["duel_outcome"] == "Won"].shape[0]
            if "duel_outcome" in duels.columns
            else 0
        )
        duel_success_rate = (won_duels / total_duels * 100) if total_duels > 0 else 0

        # Interceptions
        interceptions = comm_events[comm_events["type"] == "Interception"].copy()
        total_interceptions = len(interceptions)
        successful_interceptions = (
            interceptions[interceptions["interception_outcome"] == "Success"].shape[0]
            if "interception_outcome" in interceptions.columns
            else 0
        )

        # Clearances
        clearances = comm_events[comm_events["type"] == "Clearance"].copy()
        total_clearances = len(clearances)
        aerial_clearances = (
            clearances["clearance_aerial_won"].sum()
            if "clearance_aerial_won" in clearances.columns
            else 0
        )

        # Playing style
        dribbles = comm_events[comm_events["type"] == "Dribble"].copy()
        total_dribbles = len(dribbles)
        completed_dribbles = (
            dribbles[dribbles["dribble_outcome"] == "Complete"].shape[0]
            if "dribble_outcome" in dribbles.columns
            else 0
        )
        dribble_success_rate = (
            (completed_dribbles / total_dribbles * 100) if total_dribbles > 0 else 0
        )

        # Store results organized by category
        results.append(
            {
                "team": team_name,
                "community": comm_id,
                "n_players": len(comm_players),
                # Passing metrics
                "passing_total": total_passes,
                "passing_pct": pass_success_rate,
                "passing_length": avg_pass_length,
                # Attack metrics
                "attack_shots": total_shots,
                "attack_xg": total_xg,
                "attack_dribbles": total_dribbles,
                "attack_dribble_pct": dribble_success_rate,
                # Defense metrics
                "defense_duels": total_duels,
                "defense_duel_pct": duel_success_rate,
                "defense_interceptions": total_interceptions,
                "defense_clearances": total_clearances,
                "defense_aerial": aerial_clearances,
            }
        )

    return pd.DataFrame(results)


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
    # Community detection methods
    methods = ["louvain", "greedy", "sbm", "infomap"]

    # Store Louvain results for efficiency analysis
    louvain_communities = {}

    # Create figures for FC Barcelona
    for method in methods:
        node_community, _ = detect_communities(
            networks["Barcelona"]["graph"],
            method,
        )
        if method == "louvain":
            louvain_communities["Barcelona"] = node_community
        create_community_figure(
            team_name="Barcelona",
            method=method,
            edge_color="#f2e691",
            title_main=f"FC Barcelona Passing Network",
            fig_name=f"{method}_fcb.png",
        )

    # Create figures for Manchester United
    for method in methods:
        node_community, _ = detect_communities(
            networks["Manchester United"]["graph"],
            method,
        )
        if method == "louvain":
            louvain_communities["Manchester United"] = node_community
        create_community_figure(
            team_name="Manchester United",
            method=method,
            edge_color="black",
            title_main=f"Manchester United Passing Network",
            fig_name=f"{method}_manutd.png",
        )

    # Analyze community efficiency for Louvain method
    all_efficiency = []
    for team in ["Barcelona", "Manchester United"]:
        efficiency_df = analyze_community_efficiency(team, louvain_communities[team])
        all_efficiency.append(efficiency_df)

    # Convert to single DataFrame
    combined_efficiency = pd.concat(all_efficiency, ignore_index=True)

    # Save efficiency results
    os.makedirs(os.path.join(RESULTS_DIR, "tables"), exist_ok=True)
    combined_efficiency.to_csv(
        os.path.join(RESULTS_DIR, "tables", "community_efficiency_louvain.csv"),
        index=False,
    )
