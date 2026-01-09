"""
Community detection analysis for each team with:
- Louvain
- Greedy modularity
- Infomap
- Stochastic Block Model

# NOTE: First install graph-tool (SBM) with: conda install -c conda-forge graph-tool
"""

from __future__ import annotations
import os

from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import networkx as nx

# Imports from this project
import community_utilities as comut
import network_construction as nc
from network_construction import df

# Handle libraries that might be problematic 
try:
    from infomap import Infomap
except ImportError:
    Infomap = None
    print(
        "Infomap library not detected. It's included in requirements.txt.\n"
        "If problems persist, try:\n"
        "  pip install infomap     or     conda install -c conda-forge infomap"
    )

try:
    import graph_tool.all as gt
except ImportError:
    gt = None
    print(
        "GraphTool library not detected. It can't be included in requirements.txt.\n"
        "Install it with: conda install -c conda-forge graph-tool"
    )
    

# To fix relative paths problem
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]


# Network construction

def build_team_graph_from_events(
    events_df: pd.DataFrame,
    team: str,
    weight_col: str = "pass_count",
) -> Tuple[nx.DiGraph, pd.DataFrame, pd.DataFrame]:
    """
    Build a directed, weighted NetworkX graph for a given team from events df.
    Returns:
    G : nx.DiGraph
        Directed weighted passing network.
    nodes : pd.DataFrame
        Node table.
    edges : pd.DataFrame
        Edge table.
    """    
    
    # Get nodes and edges
    nodes, edges, passes = nc.get_network_data(events_df, team=team)

    # Create directed graph
    G = nx.DiGraph()

    # Add nodes
    for player, r in nodes.iterrows():
        G.add_node(
            player,
            x=float(r.get("x_plot", np.nan)),
            y=float(r.get("y_plot", np.nan)),
            pass_count=float(r.get("pass_count", 0)),
        )

    # Add edges
    for _, r in edges.iterrows():
        u = r["player"]
        v = r["pass_recipient"]
        w = float(r.get(weight_col, 1.0))
        if G.has_edge(u, v):
            G[u][v]["weight"] += w
        else:
            G.add_edge(u, v, weight=w)

    return G, nodes, edges


def to_undirected_weighted(G: nx.DiGraph) -> nx.Graph:
    """
    Convert directed graph to undirected weighted graph by summing weights across directions.
    Returns:
    Gu : Undirected graph
    """
    # Create directed graph
    Gu = nx.Graph()

    # Add nodes
    for n, attrs in G.nodes(data=True):
        Gu.add_node(n, **attrs)

    # Add edges
    for u, v, attrs in G.edges(data=True):
        w = float(attrs.get("weight", 1.0))
        if Gu.has_edge(u, v):
            Gu[u][v]["weight"] += w
        else:
            Gu.add_edge(u, v, weight=w)
            
    return Gu


def nx_to_graphtool(Gu: nx.Graph) -> Tuple["gt.Graph", Dict[str, int]]:
    """
    Convert an undirected NetworkX graph to a graph-tool Graph (for SBM).
    Returns the graph-tool graph and a node->vertex_index mapping.
    """

    # Check graph-tool installation
    if gt is None:
        raise ImportError(
                "Graph-tool is not installed. \n"
                "Install with: conda install -c conda-forge graph-tool "
            )

    g = gt.Graph(directed=False)

    # Define vertex property: player names
    v_label = g.new_vertex_property("string")
    g.vertex_properties["label"] = v_label

    node_index: Dict[str, int] = {}

    # Create vertices
    for n in Gu.nodes():
        v = g.add_vertex()
        v_label[v] = str(n)
        node_index[str(n)] = int(v)

    # Define edge property: weights
    e_weight = g.new_edge_property("double")
    g.edge_properties["weight"] = e_weight

    # Create edges
    for u, v, attrs in Gu.edges(data=True):
        e = g.add_edge(g.vertex(node_index[str(u)]), g.vertex(node_index[str(v)]))
        e_weight[e] = float(attrs.get("weight", 1.0))

    return g, node_index



# Community detection methods

def communities_louvain(Gu: nx.Graph, seed: int = 42) -> List[set]:
    """Detect Louvain communities."""
    return nx.community.louvain_communities(Gu, weight="weight", seed=seed)


def communities_greedy(Gu: nx.Graph) -> List[set]:
    """Detect Greedy modularity maximization communities."""
    return list(nx.community.greedy_modularity_communities(Gu, weight="weight"))


def communities_infomap(Gu: nx.Graph, two_level: bool = True, seed: int = 42) -> List[set]:
    """Detect Infomap communities."""

    # Check infomap installation
    if Infomap is None:
        raise ImportError(
            "Infomap is not installed. Install it from requirements.txt, \n"
            "using pip install infomap  or  conda install -c conda-forge infomap"
        )

    # Instantiate Infomap
    im = Infomap(silent=True, two_level=two_level, seed=seed)

    # Map node labels to integers (what infomap expects)
    node_to_id = {n: i for i, n in enumerate(Gu.nodes())}
    id_to_node = {i: n for n, i in node_to_id.items()}

    # Add weighted edges
    for u, v, attrs in Gu.edges(data=True):
        w = float(attrs.get("weight", 1.0))
        im.add_link(node_to_id[u], node_to_id[v], w)

    # Run Infomap
    im.run()

    # Convert output (node_id - module_id) to communities (module_id - nodes)
    module_to_nodes: Dict[int, set] = {}
    for node in im.nodes:
        mod = int(node.module_id)
        module_to_nodes.setdefault(mod, set()).add(id_to_node[int(node.node_id)])

    return list(module_to_nodes.values())


def communities_sbm(Gu: nx.Graph, seed: int = 42) -> List[set]:
    """ Detect SBM communities with MDL (minimum description length) objective."""
    
    # Check graph-tool installation
    if gt is None:
        raise ImportError(
                "Graph-tool is not installed. \n"
                "Install with: conda install -c conda-forge graph-tool "
            )

    # Convert NetworkX graph to graph-tool Graph (includes labels and weights)
    g, _ = nx_to_graphtool(Gu)

    # Set seed for reproducibility
    gt.seed_rng(seed)

    # Get SBM partition
    # (Use edge weights as real-exponential "edge covariates")
    state = gt.minimize_blockmodel_dl(
        g,
        state_args={"recs": [g.ep.weight], "rec_types": ["real-exponential"]},
    )

    # Extract community assignments
    blocks = state.get_blocks()

    # Convert output (node_id - module_id) to communities (module_id - nodes)
    module_to_nodes: Dict[int, set] = {}
    for v in g.vertices():
        mod = int(blocks[v])
        label = g.vp.label[v]
        module_to_nodes.setdefault(mod, set()).add(label)

    return list(module_to_nodes.values())




# Execute community detection

def main_whole_match() -> None:
    """Main community detection pipeline, separated by team, considering the whole match."""

    # Make sure output directory exists
    os.makedirs(ROOT / "results/communities/", exist_ok=True)
    out_dir = ROOT / "results/communities/"
    
    # Set team names and seed
    teams = ["Barcelona", "Manchester United"]
    seed = 42

    # Define result collector
    results_rows: List[Dict] = []

    
    # Itereate each team to detect communities
    for team in teams:
        
        # Build directed network
        Gd, _, _ = build_team_graph_from_events(df, team=team)

        # Build undirected network
        Gu = to_undirected_weighted(Gd)

        # LOUVAIN
        # Run Louvain, get modularity, print summary and save communities as a csv and image
        comm_louv = communities_louvain(Gu, seed=seed)
        mod_louv = comut.modularity_safe(Gu, comm_louv)
        memb_louv = comut.communities_to_membership(comm_louv)
        comut.print_summary(team, "louvain", comm_louv, mod_louv)
        comut.save_membership_csv(
            out_dir / f"{team.replace(' ', '_')}_louvain.csv",
            team,
            "louvain",
            memb_louv,
        )
        results_rows.append(comut.summarize_partition(team, "louvain", Gu, comm_louv))
        comut.plot_team_communities(
            events_df=df,
            team=team,
            comm_membership=memb_louv,
            period=None,
            method_name="Louvain",
            out_path=out_dir / f"Image_{team.replace(' ', '_')}_louvain.png",
        )

        # GREEDY
        # Run Greedy, get modularity, print summary and save communities in csv
        comm_greedy = communities_greedy(Gu)
        mod_greedy = comut.modularity_safe(Gu, comm_greedy)
        memb_greedy = comut.communities_to_membership(comm_greedy)
        comut.print_summary(team, "greedy", comm_greedy, mod_greedy)
        comut.save_membership_csv(
            out_dir / f"{team.replace(' ', '_')}_greedy.csv",
            team,
            "greedy",
            memb_greedy,
        )
        results_rows.append(comut.summarize_partition(team, "greedy", Gu, comm_greedy))
        comut.plot_team_communities(
            events_df=df,
            team=team,
            comm_membership=memb_greedy,
            period=None,
            method_name="Greedy",
            out_path=out_dir / f"Image_{team.replace(' ', '_')}_greedy.png",
        )

        # INFOMAP
        # Run Infomap, get modularity (just for comparison), print summary and save communities in csv
        comm_info = communities_infomap(Gu, two_level=True, seed=seed)
        mod_info = comut.modularity_safe(Gu, comm_info)
        memb_infomap = comut.communities_to_membership(comm_info)
        comut.print_summary(team, "infomap", comm_info, mod_info)
        comut.save_membership_csv(
            out_dir / f"{team.replace(' ', '_')}_infomap.csv",
            team,
            "infomap",
            memb_infomap,
        )
        results_rows.append(comut.summarize_partition(team, "infomap", Gu, comm_info))
        comut.plot_team_communities(
            events_df=df,
            team=team,
            comm_membership=memb_infomap,
            period=None,
            method_name="Infomap",
            out_path=out_dir / f"Image_{team.replace(' ', '_')}_infomap.png",
        )

        # SBM
        # Run SBM, get modularity (just for comparison), print summary and save communities in csv
        comm_sbm = communities_sbm(Gu, seed=seed)
        mod_sbm = comut.modularity_safe(Gu, comm_sbm)
        memb_sbm = comut.communities_to_membership(comm_sbm)
        comut.print_summary(team, "sbm_graphtool_mdl", comm_sbm, mod_sbm)
        comut.save_membership_csv(
            out_dir / f"{team.replace(' ', '_')}_sbm_graphtool_mdl.csv",
            team,
            "sbm_graphtool_mdl",
            memb_sbm,
        )
        results_rows.append(comut.summarize_partition(team, "sbm_graphtool_mdl", Gu, comm_sbm))
        comut.plot_team_communities(
            events_df=df,
            team=team,
            comm_membership=memb_sbm,
            period=None,
            method_name="SBM",
            out_path=out_dir / f"Image_{team.replace(' ', '_')}_SBM.png",
        )



    # Build final summary table
    results_df = pd.DataFrame(results_rows)

    # Order methods in summary table
    method_order = ["louvain", "greedy", "infomap", "sbm_graphtool_mdl"]
    results_df["method"] = pd.Categorical(results_df["method"], categories=method_order, ordered=True)
    results_df = results_df.sort_values(["team", "method"]).reset_index(drop=True)

    # Handle number notation
    display_df = results_df.copy()
    display_df["modularity"] = display_df["modularity"].map(
        lambda x: f"{x:.6f}" if pd.notna(x) else ""
    )

    # Print summary table
    print("\n==================== FINAL COMMUNITY DETECTION SUMMARY ====================")
    print(display_df[["team", "method", "n_nodes", "n_edges", "n_communities", "sizes", "modularity"]].to_string(index=False))

    # Save the summary table to CSV for your report
    display_df.to_csv(out_dir / "community_detection_summary.csv", index=False)


    print("\nDone. Community CSV outputs saved to:")
    print(f"  {out_dir}")
    print("Summary table saved to:")
    print(f"  {out_dir / 'community_detection_summary.csv'}")



def main_half_match() -> None:
    """Alternative community detection pipeline, separated by team,
    considering one half/period of the match at a time."""
    
    # Make sure output directory exists
    os.makedirs(ROOT / "results/communities/half", exist_ok=True)
    out_dir = ROOT / "results/communities/half"
    
    # Set team names, seed and possible halves/periods
    teams = ["Barcelona", "Manchester United"]
    seed = 42
    periods = [(1, "H1"), (2, "H2")]

    # Define result collector
    results_rows: List[Dict] = []

    
    # Itereate each team and period to detect communities
    for team in teams:
        for period_num, period_label in periods:
    
            # Filter the correct period
            df_p = df[df["period"] == period_num].copy()
            
            # Build directed network
            Gd, _, _ = build_team_graph_from_events(df_p, team=team)
    
            # Build undirected network
            Gu = to_undirected_weighted(Gd)
    
            # LOUVAIN
            # Run Louvain, get modularity, print summary and save communities in csv
            comm_louv = communities_louvain(Gu, seed=seed)
            mod_louv = comut.modularity_safe(Gu, comm_louv)
            memb_louvain = comut.communities_to_membership(comm_louv)
            comut.print_summary_period(team, period_label, "louvain", comm_louv, mod_louv)
            comut.save_membership_csv_period(
                out_dir / f"{team.replace(' ', '_')}_{period_label}_louvain.csv",
                team,
                period_label,
                "louvain",
                memb_louvain,
            )
            results_rows.append(comut.summarize_partition_period(team, period_label, "louvain", Gu, comm_louv))
            comut.plot_team_communities(
                events_df=df,
                team=team,
                comm_membership=memb_louvain,
                period=period_label,
                method_name="Louvain",
                out_path=out_dir / f"Image_{team.replace(' ', '_')}_{period_label}_louvain.png",
            )
    
            # GREEDY
            # Run Greedy, get modularity, print summary and save communities in csv
            comm_greedy = communities_greedy(Gu)
            mod_greedy = comut.modularity_safe(Gu, comm_greedy)
            memb_greedy = comut.communities_to_membership(comm_greedy)
            comut.print_summary_period(team, period_label, "greedy", comm_greedy, mod_greedy)
            comut.save_membership_csv_period(
                out_dir / f"{team.replace(' ', '_')}_{period_label}_greedy.csv",
                team,
                period_label,
                "greedy",
                memb_greedy,
            )
            results_rows.append(comut.summarize_partition_period(team, period_label, "greedy", Gu, comm_greedy))
            comut.plot_team_communities(
                events_df=df,
                team=team,
                comm_membership=memb_greedy,
                period=period_label,
                method_name="Greedy",
                out_path=out_dir / f"Image_{team.replace(' ', '_')}_{period_label}_greedy.png",
            )
    
            # INFOMAP
            # Run Infomap, get modularity (just for comparison), print summary and save communities in csv
            comm_info = communities_infomap(Gu, two_level=True, seed=seed)
            mod_info = comut.modularity_safe(Gu, comm_info) 
            memb_infomap = comut.communities_to_membership(comm_info)
            comut.print_summary_period(team, period_label, "infomap", comm_info, mod_info)
            comut.save_membership_csv_period(
                out_dir / f"{team.replace(' ', '_')}_{period_label}_infomap.csv",
                team,
                period_label,
                "infomap",
                memb_infomap,
            )
            results_rows.append(comut.summarize_partition_period(team, period_label, "infomap", Gu, comm_info))
            comut.plot_team_communities(
                events_df=df,
                team=team,
                comm_membership=memb_infomap,
                period=period_label,
                method_name="Infomap",
                out_path=out_dir / f"Image_{team.replace(' ', '_')}_{period_label}_infomap.png",
            )

    
            # SBM
            # Run SBM, get modularity (just for comparison), print summary and save communities in csv
            comm_sbm = communities_sbm(Gu, seed=seed)
            mod_sbm = comut.modularity_safe(Gu, comm_sbm)
            memb_sbm = comut.communities_to_membership(comm_sbm)
            comut.print_summary_period(team, period_label, "sbm_graphtool_mdl", comm_sbm, mod_sbm)
            comut.save_membership_csv_period(
                out_dir / f"{team.replace(' ', '_')}_{period_label}_sbm_graphtool_mdl.csv",
                team,
                period_label,
                "sbm_graphtool_mdl",
                memb_sbm,
            )
            results_rows.append(comut.summarize_partition_period(team, period_label, "sbm_graphtool_mdl", Gu, comm_sbm))
            comut.plot_team_communities(
                events_df=df,
                team=team,
                comm_membership=memb_sbm,
                period=period_label,
                method_name="SBM",
                out_path=out_dir / f"Image_{team.replace(' ', '_')}_{period_label}_SBM.png",
            )


    # Build final summary table
    results_df = pd.DataFrame(results_rows)

    # Order methods in summary table
    method_order = ["louvain", "greedy", "infomap", "sbm_graphtool_mdl"]
    results_df["method"] = pd.Categorical(results_df["method"], categories=method_order, ordered=True)
    results_df = results_df.sort_values(["team", "period", "method"]).reset_index(drop=True)

    # Handle number notation
    display_df = results_df.copy()
    display_df["modularity"] = display_df["modularity"].map(
        lambda x: f"{x:.6f}" if pd.notna(x) else ""
    )

    # Print summary table
    print("\n==================== FINAL COMMUNITY DETECTION SUMMARY ====================")
    print(display_df[["team", "period", "method", "n_nodes", "n_edges", "n_communities", "sizes", "modularity"]].to_string(index=False))

    # Save the summary table to CSV for your report
    display_df.to_csv(out_dir / "community_detection_period_summary.csv", index=False)


    print("\nDone. Community CSV outputs saved to:")
    print(f"  {out_dir}")
    print("Summary table saved to:")
    print(f"  {out_dir / 'community_detection_summary.csv'}")



if __name__ == "__main__":
    main_whole_match()
    main_half_match()
