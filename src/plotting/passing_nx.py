# https://mplsoccer.readthedocs.io/en/latest/gallery/pitch_plots/plot_pass_network.html
# https://soccermatics.readthedocs.io/en/latest/gallery/lesson1/plot_PassNetworks.html
# https://www.youtube.com/watch?v=pW7rltisoqo
# https://www.youtube.com/watch?v=ZOC4DSHiKVU

import os
from ast import literal_eval

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Circle

# Setup data
df = pd.read_csv("../../data/barca_manutd_2011_events.csv")
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


# Function to draw vertical green pitch
def draw_pitch(ax, title):
    # Colors
    pitch_color = "#4B8B3B"
    line_color = "white"

    # Draw green rectangle for pitch background
    from matplotlib.patches import Rectangle

    pitch_rect = Rectangle(
        (-5, -5), 90, 130, facecolor=pitch_color, edgecolor="none", zorder=0
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

    # Edges
    pass_between = (
        df_passes.groupby(
            [
                "player",
                "pass_recipient",
            ]
        )
        .id.count()
        .reset_index()
    )
    pass_between.rename(columns={"id": "pass_count"}, inplace=True)
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
def create_team_figure(team_name, time_windows, node_color, edge_color, title_main):
    # Create subplot
    fig, axes = plt.subplots(
        2, 2, figsize=(14, 14)
    )  # Increased from (10, 10) for more space
    axes = axes.flatten()

    # Overall figure settings - WHITE BACKGROUND
    fig.set_facecolor("white")
    fig.suptitle(title_main, fontsize=24, color="black", weight="bold", y=0.97)

    # Add StatsBomb credit - BLACK TEXT
    fig.text(
        0.5,
        0.92,
        "Data provided by StatsBomb (https://statsbomb.com)",
        ha="center",
        fontsize=11,
        color="black",
        style="italic",
    )

    # Plot each time window
    for i, (ax, (start, end, label)) in enumerate(zip(axes, time_windows)):
        # Draw Pitch
        draw_pitch(ax, f"{label}\n({start}'-{end}')")

        # Get network data
        nodes, edges = get_network_data(df, team_name, start, end)

        # Plot network if data exists
        if nodes is not None and not nodes.empty:
            # Draw Edges
            for _, row in edges.iterrows():
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

            # Draw nodes
            ax.scatter(
                nodes["x"],
                nodes["y"],
                s=nodes["count"] * 15,
                c=node_color,
                edgecolors=edge_color,
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
        else:
            ax.text(
                40,
                60,
                "Low/No Data",
                color="black",
                ha="center",
                fontsize=14,
                bbox=dict(
                    boxstyle="round,pad=0.5",
                    facecolor="white",
                    edgecolor="black",
                    alpha=0.8,
                ),
            )

    # Show plot and save
    plt.subplots_adjust(
        left=0.02,
        right=0.98,
        top=0.85,
        bottom=0.02,
        wspace=0.08,  # Increased horizontal spacing to prevent label overlap
        hspace=0.30,  # Increased vertical spacing for more room
    )
    os.makedirs("../../images/network_visualization", exist_ok=True)
    plt.savefig(
        f"../../images/network_visualization/{team_name.lower().replace(' ', '_')}_passing_networks.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


# FC Barcelona windows
barca_windows = [
    (0, 45, "First Half"),
    (45, 85, "Second Half (Start)"),
    (85, 87, "After Sub 1 (Villa->Keita)"),
    (87, 94, "After Sub 2 (Puyol->Abidal)"),
]

# Man Utd windows
utd_windows = [
    (0, 45, "First Half"),
    (45, 68, "Second Half (Start)"),
    (68, 76, "After Sub 1 (Fábio->Nani)"),
    (76, 94, "After Sub 2 (Carrick->Scholes)"),
]

# Create figures
create_team_figure(
    "Barcelona",
    barca_windows,
    node_color="#a50044",
    edge_color="#f2e691",
    title_main="FC Barcelona Passing Networks",
)
create_team_figure(
    "Manchester United",
    utd_windows,
    node_color="white",
    edge_color="black",
    title_main="Manchester United Passing Networks",
)
