"""
Script containing utilities or helper functions used in the community detection script.
"""

from __future__ import annotations
import os

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

import network_construction as nc



def communities_to_membership(communities: List[set]) -> Dict[str, int]:
    """Convert list of communities into a node - community_id mapping."""
    
    membership: Dict[str, int] = {}
    for cid, comm in enumerate(communities):
        for node in comm:
            membership[str(node)] = int(cid)
    return membership


def modularity_safe(Gu: nx.Graph, communities: List[set]) -> Optional[float]:
    """Compute modularity, but only when possible."""
    
    try:
        return float(nx.community.modularity(Gu, communities, weight="weight"))
    except Exception:
        return None

    
def save_membership_csv(out_path: Path, team: str, method: str, membership: Dict[str, int]) -> None:
    """ Save node - community mapping as a CSV file."""
    df_out = (
        pd.DataFrame({"player": list(membership.keys()), "community": list(membership.values())})
        .sort_values(["community", "player"])
        .reset_index(drop=True)
    )
    df_out.insert(0, "team", team)
    df_out.insert(1, "method", method)
    df_out.to_csv(out_path, index=False)


def save_membership_csv_period(out_path: Path, team: str, period: str, method: str, membership: Dict[str, int]) -> None:
    """ Save node - community mapping as a CSV file for period analysis."""
    df_out = (
        pd.DataFrame({"player": list(membership.keys()), "community": list(membership.values())})
        .sort_values(["community", "player"])
        .reset_index(drop=True)
    )
    df_out.insert(0, "team", team)
    df_out.insert(1, "period", period)
    df_out.insert(2, "method", method)

    df_out.to_csv(out_path, index=False)


def print_summary(team: str, method: str, communities: List[set], mod: Optional[float]) -> None:
    """
    Print summary of a community detection result with the team name, community detection method, number of detected communities, size of each community and, if possible, the modularity value.
    """
    sizes = sorted([len(c) for c in communities], reverse=True)
    print("\n" + "=" * 80)
    print(f"Team: {team}")
    print(f"Method: {method}")
    print(f"#Communities: {len(communities)}")
    print(f"Sizes: {sizes}")
    if mod is not None:
        print(f"Modularity (weighted): {mod:.4f}")
    else:
        print("Modularity: (not computed)")
    print("=" * 80)


def print_summary_period(team: str, period: str, method: str, communities: List[set], mod: Optional[float]) -> None:
    """
    Print summary of a community detection result with the team name, community detection method, number of detected communities, size of each community and, if possible, the modularity value.
    """
    sizes = sorted([len(c) for c in communities], reverse=True)
    print("\n" + "=" * 80)
    print(f"Team: {team}")
    print(f"Period: {period}")
    print(f"Method: {method}")
    print(f"#Communities: {len(communities)}")
    print(f"Sizes: {sizes}")
    if mod is not None:
        print(f"Modularity (weighted): {mod:.4f}")
    else:
        print("Modularity: (not computed)")
    print("=" * 80)
    

def summarize_partition(team: str, method: str, Gu: nx.Graph, communities: List[set]) -> Dict:
    """Create a summary row for the final communities results table."""
    sizes = sorted((len(c) for c in communities), reverse=True)
    return {
        "team": team,
        "method": method,
        "n_nodes": Gu.number_of_nodes(),
        "n_edges": Gu.number_of_edges(),
        "n_communities": len(communities),
        "sizes": sizes,
        "modularity": modularity_safe(Gu, communities),
    }


def summarize_partition_period(team: str, period_label: str, method: str, Gu: nx.Graph, communities: List[set]) -> Dict:
    """Create a summary row for the final communities results table."""
    sizes = sorted((len(c) for c in communities), reverse=True)
    return {
        "team": team,
        "period": period_label,
        "method": method,
        "n_nodes": Gu.number_of_nodes(),
        "n_edges": Gu.number_of_edges(),
        "n_communities": len(communities),
        "sizes": sizes,
        "modularity": modularity_safe(Gu, communities),
    }



# Visualization

def plot_team_communities(
    events_df: pd.DataFrame,
    team: str,
    comm_membership: Dict[str, int],
    period: Optional[Union[int, str]] = None,
    method_name: str = "community_method",
    out_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot the network on a pitch, coloring nodes (players) by their detected community.
    """
    
    # Set dfp in case that no period is provided (take the whole match)
    dfp = events_df
    
    # Filter the period is provided (take half of the match)
    if period is not None:
        # Accept period as "H1"/"H2" or 1/2
        if isinstance(period, str):
            p = period.strip().upper()
            if p == "H1":
                period_num = 1
            elif p == "H2":
                period_num = 2
            else:
                raise ValueError("period must be 'H1' or 'H2' when passed as a string.")
        else:
            period_num = int(period)

        # Filter period
        dfp = events_df[events_df["period"] == period_num].copy()

    
    # Get nodes/edges
    nodes, edges, passes = nc.get_network_data(dfp, team=team)

    # Handle player labels as index
    nodes_plot = nodes.reset_index().copy()
    
    # Add community id to nodes
    nodes_plot["community"] = nodes_plot["player"].map(comm_membership)

    # Flag players not present in the membership mapping
    nodes_plot["community"] = nodes_plot["community"].fillna(-1).astype(int)


    # Build figure 
    fig, ax = plt.subplots(figsize=(10, 7))

    # Draw pitch
    nc.draw_pitch(ax)

    # Get the max pass count to set transparency
    wmax = float(edges["pass_count"].max()) if (len(edges) > 0) else 1.0
    wmax = max(wmax, 1.0)


    # Iterate over each edge
    for _, r in edges.iterrows():
        # Define pass count, line width and transparency
        pc = float(r.get("pass_count", 1.0))
        lw = 0.6 + pc / 6.0 
        alpha = min(0.85, max(0.15, 0.15 + pc / (wmax * 1.2)))

        # Define start and end coordinates of the pass
        xs, ys = float(r["x_start"]), float(r["y_start"])
        xe, ye = float(r["x_end"]), float(r["y_end"])

        # Draw edge
        ax.plot([xs, xe], [ys, ye], color="0.9", alpha=alpha, linewidth=lw, zorder=1)

    # Set node sizes (scale by pass_count)
    if "pass_count" in nodes_plot.columns:
        pc = nodes_plot["pass_count"].astype(float)
        s_min, s_max = 260, 520
        pc_min, pc_max = float(pc.min()), float(pc.max())
        if pc_max > pc_min:
            nodes_plot["_size"] = s_min + (pc - pc_min) * (s_max - s_min) / (pc_max - pc_min)
        else:
            nodes_plot["_size"] = (s_min + s_max) / 2.0
    else:
        nodes_plot["_size"] = 380.0

    # Set label offset (so it doesn't overlap with the nodes)
    nodes_plot["_label_dy"] = 3 + 0.0025 * nodes_plot["_size"]

   
    # Draw nodes colored by community
    cmap = plt.get_cmap("tab20")
    for cid in sorted(nodes_plot["community"].unique()):
        dfc = nodes_plot[nodes_plot["community"] == cid]

        # Node color
        if cid == -1:
            color = "lightgray"
            label = "No passes"
        else:
            color = cmap(cid % cmap.N)
            label = f"C{cid}"
    
        ax.scatter(
            dfc["x"],
            dfc["y"],
            s=900,
            c=[color],
            edgecolors="black",
            linewidths=1.5,
            zorder=3,
            label=label,
            )

        # Player labels
        for _, rr in dfc.iterrows():
            ax.text(
                float(rr["x"]),
                float(rr["y"]) + float(rr["_label_dy"]),
                str(rr["player"]),
                ha="center",
                va="bottom",
                fontsize=10,
                zorder=4,
                bbox=dict(
                    boxstyle="round,pad=0.18",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.65,
                ),
            )
    
    # Title    
    if period is not None:
        title = f"{team} | {period} | {method_name} communities"
    else:
        title = f"{team} | {method_name} communities"

    ax.set_title(title, fontsize=14)

    # Legend
    if nodes_plot["community"].nunique() <= 10:
        ax.legend(loc="upper right", frameon=True, fontsize=9)

    fig.tight_layout()

    # Save 
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
        
    return fig

