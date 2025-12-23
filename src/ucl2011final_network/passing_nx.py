# https://mplsoccer.readthedocs.io/en/latest/gallery/pitch_plots/plot_pass_network.html
# https://soccermatics.readthedocs.io/en/latest/gallery/lesson1/plot_PassNetworks.html
# https://www.youtube.com/watch?v=pW7rltisoqo
# https://www.youtube.com/watch?v=ZOC4DSHiKVU

import os
from ast import literal_eval

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle

# Path variables
RAW_DATA_PATH = "../../data/raw/barca_manutd_2011_events.csv"
PROCESSED_DATA_DIR = "../../data/processed/"
FIGURES_DIR = "../../results/figures/"

# Setup data
df = pd.read_csv(RAW_DATA_PATH)
df["location"] = df["location"].apply(
    lambda x: literal_eval(str(x)) if pd.notnull(x) else None
)

# Name mapping
player_name_map = {
    # FC Barcelona Players
    "Víctor Valdés Arribas": "Valdés",
    "Daniel Alves da Silva": "Dani Alves",
    "Gerard Piqué Bernabéu": "Piqué",
    "Javier Alejandro Mascherano": "Mascherano",
    "Eric-Sylvain Bilal Abidal": "Abidal",
    "Sergio Busquets i Burgos": "Busquets",
    "Xavier Hernández Creus": "Xavi",
    "Andrés Iniesta Luján": "Iniesta",
    "Pedro Eliezer Rodríguez Ledesma": "Pedro",
    "Lionel Andrés Messi Cuccittini": "Messi",
    "David Villa Sánchez": "Villa",
    "Carles Puyol i Saforcada": "Puyol",
    "Seydou Kéita": "Keita",
    "Ibrahim Afellay": "Afellay",
    # Manchester United Players
    "Edwin van der Sar": "Van der Sar",
    "Fábio Pereira da Silva": "Fábio",
    "Rio Ferdinand": "Ferdinand",
    "Nemanja Vidić": "Vidić",
    "Patrice Evra": "Evra",
    "Antonio Valencia": "Valencia",
    "Michael Carrick": "Carrick",
    "Ryan Giggs": "Giggs",
    "Ji-Sung Park": "Park",
    "Wayne Mark Rooney": "Rooney",
    "Javier Hernández Balcázar": "Chicharito",
    "Paul Scholes": "Scholes",
    "Luis Antonio Valencia Mosquera": "Valencia",
    "Luís Carlos Almeida da Cunha": "Nani",
}

# Apply name mapping
df["player"] = df["player"].map(player_name_map).fillna(df["player"])
df["pass_recipient"] = (
    df["pass_recipient"].map(player_name_map).fillna(df["pass_recipient"])
)


# Pass type color mapping
def get_pass_color(pass_type, is_cross, is_switch):
    # Red for crosses
    if pd.notnull(is_cross) and is_cross:
        return "#FF6B6B"
    # Cyan for switches
    elif pd.notnull(is_switch) and is_switch:
        return "#4ECDC4"
    # Light green for through balls
    elif pd.notnull(pass_type):
        if "Through Ball" in str(pass_type):
            return "#95E1D3"
        # Light salmon for chips
        elif "Chipped" in str(pass_type):
            return "#FFA07A"
    # Default yellow
    return "#f2e691"


# Function to draw vertical green pitch
def draw_pitch(ax, title):
    # Colors
    pitch_color = "#4B8B3B"
    line_color = "white"

    # Pitch background
    pitch_rect = Rectangle(
        (-5, -5),
        90,
        130,
        facecolor=pitch_color,
        edgecolor="none",
        zorder=0,
    )
    ax.add_patch(pitch_rect)

    # Outline
    ax.plot([0, 80], [0, 0], color=line_color, linewidth=2)
    ax.plot([80, 80], [0, 120], color=line_color, linewidth=2)
    ax.plot([80, 0], [120, 120], color=line_color, linewidth=2)
    ax.plot([0, 0], [120, 0], color=line_color, linewidth=2)

    # Halfway
    ax.plot([0, 80], [60, 60], color=line_color, linewidth=2)

    # Boxes
    ax.plot([18, 62], [18, 18], color=line_color, linewidth=2)
    ax.plot([18, 18], [0, 18], color=line_color, linewidth=2)
    ax.plot([62, 62], [0, 18], color=line_color, linewidth=2)
    ax.plot([18, 62], [102, 102], color=line_color, linewidth=2)
    ax.plot([18, 18], [120, 102], color=line_color, linewidth=2)
    ax.plot([62, 62], [120, 102], color=line_color, linewidth=2)

    # Circle
    circle = Circle((40, 60), 9.15, color=line_color, fill=False, linewidth=2)
    ax.add_patch(circle)

    # Penalty spots
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-5, 85)
    ax.set_ylim(-5, 125)
    ax.set_title(title, color="black", fontsize=12, weight="bold", pad=0)


# Function to get nodes and edges for a team in a time window
def get_network_data(df, team, start_min, end_min):
    # Filter team & time
    mask = (df["team"] == team) & (df["minute"] >= start_min) & (df["minute"] < end_min)
    df_slice = df[mask].copy()

    # Successful passes only
    df_passes = df_slice[
        (df_slice["type"] == "Pass") & (df_slice["pass_outcome"].isnull())
    ].copy()

    # Check for empty data
    if df_passes.empty:
        return None, None

    # Coordinates (Vertical: x=Width, y=Length)
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

    # Edges with pass type information
    pass_between = (
        df_passes.groupby(["player", "pass_recipient"])
        .agg(
            {
                "id": "count",
                "pass_type": lambda x: x.mode()[0] if not x.mode().empty else None,
                "pass_cross": lambda x: x.sum() > 0,
                "pass_switch": lambda x: x.sum() > 0,
            }
        )
        .reset_index()
    )

    # Rename count column
    pass_between.rename(columns={"id": "pass_count"}, inplace=True)

    # Add color based on pass type
    pass_between["edge_color"] = pass_between.apply(
        lambda row: get_pass_color(
            row["pass_type"], row["pass_cross"], row["pass_switch"]
        ),
        axis=1,
    )

    # Merge average locations for plotting
    pass_between = pass_between.merge(avg_loc, left_on="player", right_index=True)
    pass_between = pass_between.rename(columns={"x": "x_start", "y": "y_start"})
    pass_between = pass_between.merge(
        avg_loc,
        left_on="pass_recipient",
        right_index=True,
    )
    pass_between = pass_between.rename(columns={"x": "x_end", "y": "y_end"})

    return avg_loc, pass_between


# Function to create figure for a team
def create_team_figure(
    team_name,
    time_windows,
    node_color,
    default_edge_color,
    title_main,
    fig_name,
):
    # Determine subplot layout based on number of windows
    if len(time_windows) == 1:
        fig, axes = plt.subplots(1, 1, figsize=(10, 10))
        axes = [axes]
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes = axes.flatten()

    # Overall figure settings
    fig.set_facecolor("white")
    fig.suptitle(
        title_main,
        fontsize=24 if len(time_windows) > 1 else 22,
        color="black",
        weight="bold",
        y=0.97 if len(time_windows) > 1 else 0.98,
    )

    # Add StatsBomb credit with more spacing
    fig.text(
        0.5,
        0.92,
        "Data provided by StatsBomb (https://statsbomb.com)",
        ha="center",
        fontsize=11,
        color="black",
        style="italic",
    )

    # Initialize variable to store total passes for full match
    total_passes = 0

    # Plot each time window
    for i, (ax, (start, end, label)) in enumerate(zip(axes, time_windows)):
        # Draw pitch
        if len(time_windows) > 1:
            draw_pitch(ax, f"{label}\n({start}'-{end}')")
        else:
            draw_pitch(ax, "")

        # Get network data using data_team_name
        nodes, edges = get_network_data(df, team_name, start, end)

        # Draw edges with color coding
        if (
            edges is not None
            and not edges.empty
            and nodes is not None
            and not nodes.empty
        ):
            # Calculate total passes for this time window
            window_passes = edges["pass_count"].sum()

            # For full match (single window), store the total
            if len(time_windows) == 1:
                total_passes = window_passes

            for _, row in edges.iterrows():
                # Edge width and transparency based on pass count
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
                        color=row["edge_color"],
                        connectionstyle="arc3,rad=0.1",
                        linewidth=width,
                        alpha=alpha,
                    ),
                )

            # Draw nodes
            ax.scatter(
                nodes["x"],
                nodes["y"],
                s=nodes["count"] * 15,
                c=node_color,
                edgecolors=default_edge_color,
                zorder=10,
            )

            # Draw names
            for idx, row in nodes.iterrows():
                ax.text(
                    row["x"],
                    row["y"] + 4,
                    idx,
                    fontsize=9,
                    ha="center",
                    va="center",
                    color="black",
                    weight="bold",
                    zorder=11,
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor="white",
                        edgecolor="none",
                        alpha=0.7,
                    ),
                )

    # Add legend for single panel (full match)
    if len(time_windows) == 1:
        # Add legend for single panel
        legend_elements = []

        # Match events based on team
        if "Barcelona" in team_name:
            # Total passes entry
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker="s",
                    color="w",
                    markerfacecolor="white",
                    markersize=10,
                    markeredgecolor="black",
                    markeredgewidth=1.5,
                    label=f"Total Passes: {total_passes}",
                    linestyle="None",
                )
            )
            # Goals
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="white",
                    markersize=10,
                    markeredgecolor="black",
                    markeredgewidth=1.5,
                    label="Goals: Pedro 27', Messi 54', Villa 69'",
                    linestyle="None",
                )
            )
            # Yellow cards
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker="s",
                    color="w",
                    markerfacecolor="yellow",
                    markersize=10,
                    markeredgecolor="black",
                    markeredgewidth=1.5,
                    label="Yellow Cards: Dani Alves 60', Villa 86'",
                    linestyle="None",
                )
            )
            # Substitutions IN
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker="^",
                    color="w",
                    markerfacecolor="green",
                    markersize=10,
                    markeredgecolor="black",
                    markeredgewidth=1.5,
                    label="Subs IN: Keita 86', Puyol 88', Afellay 90+2'",
                    linestyle="None",
                )
            )
            # Substitutions OUT
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker="v",
                    color="w",
                    markerfacecolor="red",
                    markersize=10,
                    markeredgecolor="black",
                    markeredgewidth=1.5,
                    label="Subs OUT: Villa 86', Dani Alves 88', Pedro 90+2'",
                    linestyle="None",
                )
            )
        elif "Manchester United" in team_name:
            # Total passes entry
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker="s",
                    color="w",
                    markerfacecolor="white",
                    markersize=10,
                    markeredgecolor="black",
                    markeredgewidth=1.5,
                    label=f"Total Passes: {total_passes}",
                    linestyle="None",
                )
            )
            # Goals
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="white",
                    markersize=10,
                    markeredgecolor="black",
                    markeredgewidth=1.5,
                    label="Goal: Rooney 34'",
                    linestyle="None",
                )
            )
            # Yellow cards
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker="s",
                    color="w",
                    markerfacecolor="yellow",
                    markersize=10,
                    markeredgecolor="black",
                    markeredgewidth=1.5,
                    label="Yellow Cards: Carrick 61', Valencia 79'",
                    linestyle="None",
                )
            )
            # Substitutions IN
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker="^",
                    color="w",
                    markerfacecolor="green",
                    markersize=10,
                    markeredgecolor="black",
                    markeredgewidth=1.5,
                    label="Subs IN: Nani 69', Scholes 77'",
                    linestyle="None",
                )
            )
            # Substitutions OUT
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker="v",
                    color="w",
                    markerfacecolor="red",
                    markersize=10,
                    markeredgecolor="black",
                    markeredgewidth=1.5,
                    label="Subs OUT: Fábio 69', Carrick 77'",
                    linestyle="None",
                )
            )

        # Add legend to the axes
        if legend_elements:
            axes[0].legend(
                handles=legend_elements,
                loc="upper left",
                bbox_to_anchor=(0.01, 0.99),
                fontsize=11,
                framealpha=0.95,
                edgecolor="black",
                fancybox=False,
            )

            # Adjust layout
            plt.subplots_adjust(
                left=0.02,
                right=0.98,
                top=0.9,
                bottom=0.02,
                wspace=0.08,
                hspace=0.30,
            )
    else:
        # Adjust layout
        plt.subplots_adjust(
            left=0.02,
            right=0.98,
            top=0.85,
            bottom=0.02,
            wspace=0.08,
            hspace=0.30,
        )

    # Save figure
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.savefig(
        f"{FIGURES_DIR}{fig_name}",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


# FC Barcelona windows
barca_windows = [
    (0, 45, "First Half"),
    (45, 86, "Second Half"),
    (86, 92, "Subs 1&2 (Villa→Keita, Alves→Puyol)"),
    (92, 94, "Sub 3 (Pedro→Afellay)"),
]

# Man Utd windows
utd_windows = [
    (0, 45, "First Half"),
    (45, 69, "Second Half"),
    (69, 77, "Sub 1 (Fábio→Nani)"),
    (77, 94, "Sub 2 (Carrick→Scholes)"),
]

# Create segmented figures for both teams
create_team_figure(
    "Barcelona",
    barca_windows,
    node_color="#a50044",
    default_edge_color="#f2e691",
    title_main="FC Barcelona Passing Networks",
    fig_name="temporal_barcelona_passing_networks.png",
)
create_team_figure(
    "Manchester United",
    utd_windows,
    node_color="white",
    default_edge_color="black",
    title_main="Manchester United Passing Networks",
    fig_name="temporal_manchester_united_passing_networks.png",
)

# Barcelona temporal segments
for i, (start, end, label) in enumerate(barca_windows, 1):
    barca_nodes_temp, barca_edges_temp = get_network_data(
        df,
        "Barcelona",
        start,
        end,
    )
    if barca_edges_temp is not None:
        barca_export_temp = barca_edges_temp[
            [
                "player",
                "pass_recipient",
                "pass_count",
                "pass_type",
                "pass_cross",
                "pass_switch",
                "edge_color",
            ]
        ].copy()
        barca_export_temp["team"] = "Barcelona"
        period_name = (
            label.lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("&", "and")
            .replace("→", "_")
            .replace(",", "")
        )
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        output_file = f"{PROCESSED_DATA_DIR}barcelona_{period_name}.csv"
        barca_export_temp.to_csv(output_file, index=False)

# Manchester United temporal segments
for i, (start, end, label) in enumerate(utd_windows, 1):
    utd_nodes_temp, utd_edges_temp = get_network_data(
        df,
        "Manchester United",
        start,
        end,
    )
    if utd_edges_temp is not None:
        utd_export_temp = utd_edges_temp[
            [
                "player",
                "pass_recipient",
                "pass_count",
                "pass_type",
                "pass_cross",
                "pass_switch",
                "edge_color",
            ]
        ].copy()
        utd_export_temp["team"] = "Manchester United"
        period_name = (
            label.lower()
            .replace("á", "a")
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("→", "_")
            .replace(",", "")
        )
        output_file = f"{PROCESSED_DATA_DIR}manchester_united_{period_name}.csv"
        utd_export_temp.to_csv(output_file, index=False)


# Create full match network for Barcelona
barca_full_window = [(0, 94, "Full Match")]
create_team_figure(
    team_name="Barcelona",
    time_windows=barca_full_window,
    node_color="#a50044",
    default_edge_color="#f2e691",
    title_main="FC Barcelona Passing Network",
    fig_name="barcelona_full_match_passing_network.png",
)

# Create full match network for Manchester United
utd_full_window = [(0, 94, "Full Match")]
create_team_figure(
    team_name="Manchester United",
    time_windows=utd_full_window,
    node_color="white",
    default_edge_color="black",
    title_main="Manchester United Passing Network",
    fig_name="manchester_united_full_match_passing_network.png",
)

# Barcelona full match network data
barca_nodes_combined, barca_edges_combined = get_network_data(
    df,
    "Barcelona",
    0,
    94,
)
if barca_edges_combined is not None:
    barca_combined_export = barca_edges_combined[
        [
            "player",
            "pass_recipient",
            "pass_count",
            "pass_type",
            "pass_cross",
            "pass_switch",
            "edge_color",
        ]
    ].copy()
    barca_combined_export["team"] = "Barcelona"
    output_file_combined = f"{PROCESSED_DATA_DIR}barcelona_full_match_all_players.csv"
    barca_combined_export.to_csv(output_file_combined, index=False)

# Manchester United full match network data
utd_nodes_combined, utd_edges_combined = get_network_data(
    df,
    "Manchester United",
    0,
    94,
)
if utd_edges_combined is not None:
    utd_combined_export = utd_edges_combined[
        [
            "player",
            "pass_recipient",
            "pass_count",
            "pass_type",
            "pass_cross",
            "pass_switch",
            "edge_color",
        ]
    ].copy()
    utd_combined_export["team"] = "Manchester United"
    output_file_combined = (
        f"{PROCESSED_DATA_DIR}manchester_united_full_match_all_players.csv"
    )
    utd_combined_export.to_csv(output_file_combined, index=False)
