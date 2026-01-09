# https://mplsoccer.readthedocs.io/en/latest/gallery/pitch_plots/plot_pass_network.html
# https://soccermatics.readthedocs.io/en/latest/gallery/lesson1/plot_PassNetworks.html
# https://www.youtube.com/watch?v=pW7rltisoqo
# https://www.youtube.com/watch?v=ZOC4DSHiKVU

import os
from ast import literal_eval

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Circle, Rectangle
from PIL import Image, ImageDraw

# To fix GUI backend problem
import matplotlib
matplotlib.use("Agg")

# To fix relative paths problem
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
IMAGES_DIR = ROOT / "images"

# Setup data
df = pd.read_csv(ROOT / "data/barca_manutd_2011_events.csv")
df["location"] = df["location"].apply(
    lambda x: literal_eval(str(x)) if pd.notnull(x) else None
)

# Name mapping with image paths
player_name_map = {
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

# Player image paths mapping
player_images = {
    "Valdés": IMAGES_DIR / "fcb/victor_valdes.jpg",
    "Dani Alves": IMAGES_DIR / "fcb/dani_alves.jpg",
    "Piqué": IMAGES_DIR / "fcb/gerard_pique.jpg",
    "Mascherano": IMAGES_DIR / "fcb/javier_mascherano.jpg",
    "Abidal": IMAGES_DIR / "fcb/eric_abidal.jpg",
    "Busquets": IMAGES_DIR / "fcb/sergio_busquets.jpg",
    "Xavi": IMAGES_DIR / "fcb/xavi.jpg",
    "Iniesta": IMAGES_DIR / "fcb/andres_iniesta.jpg",
    "Pedro": IMAGES_DIR / "fcb/pedro.jpg",
    "Messi": IMAGES_DIR / "fcb/lionel_messi.jpg",
    "Villa": IMAGES_DIR / "fcb/david_villa.jpg",
    "Puyol": IMAGES_DIR / "fcb/carles_puyol.jpg",
    "Keita": IMAGES_DIR / "fcb/seydou_keita.jpg",
    "Afellay": IMAGES_DIR / "fcb/ibrahim_afellay.jpg",
    "Van der Sar": IMAGES_DIR / "man_utd/edwin_van_der_sar.jpg",
    "Fábio": IMAGES_DIR / "man_utd/fabio.jpg",
    "Ferdinand": IMAGES_DIR / "man_utd/rio_ferdinand.jpg",
    "Vidić": IMAGES_DIR / "man_utd/nemanja_vidic.jpg",
    "Evra": IMAGES_DIR / "man_utd/patrice_evra.jpg",
    "Valencia": IMAGES_DIR / "man_utd/antonio_valencia.jpg",
    "Carrick": IMAGES_DIR / "man_utd/michael_carrick.jpg",
    "Giggs": IMAGES_DIR / "man_utd/ryan_giggs.jpg",
    "Park": IMAGES_DIR / "man_utd/jisung_park.jpg",
    "Rooney": IMAGES_DIR / "man_utd/wayne_rooney.jpg",
    "Chicharito": IMAGES_DIR / "man_utd/chicharito.jpg",
    "Scholes": IMAGES_DIR / "man_utd/paul_scholes.jpg",
    "Nani": IMAGES_DIR / "man_utd/nani.jpg",
}

# Apply name mapping
df["player"] = df["player"].map(player_name_map).fillna(df["player"])
df["pass_recipient"] = (
    df["pass_recipient"].map(player_name_map).fillna(df["pass_recipient"])
)


# Function to draw vertical green pitch
def draw_pitch(ax):
    # Set colors
    pitch_color = "#4B8B3B"
    line_color = "white"

    # Draw pitch
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


# Function to get nodes and edges for a team
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
        return None, None, None

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

    return avg_loc, pass_between, df_passes_filtered[["player", "pass_recipient"]]


# Function to create figure for a team
def create_team_figure(team_name, edge_color, title_main, fig_name):
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

    # Get network data
    nodes, edges, _ = get_network_data(df, team_name)

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
            img_path = player_images[player_name]

            # Load and process image
            img = Image.open(img_path)
            img = img.convert("RGBA")

            # Resize to base size
            size = 80
            img = img.resize((size, size), Image.Resampling.LANCZOS)

            # Create circular mask
            mask = Image.new("L", (size, size), 0)
            draw = ImageDraw.Draw(mask)
            draw.ellipse((0, 0, size, size), fill=255)

            # Apply circular mask
            output = Image.new("RGBA", (size, size), (0, 0, 0, 0))
            output.paste(img, (0, 0))
            output.putalpha(mask)

            # Calculate zoom based on degree
            base_zoom = 0.18
            zoom = base_zoom + (row["count"] / 200)

            # Create image box
            imagebox = OffsetImage(output, zoom=zoom)

            # Add image to plot
            ab = AnnotationBbox(
                imagebox,
                (row["x"], row["y"]),
                frameon=True,
                pad=0.1,
                boxcoords="data",
                box_alignment=(0.5, 0.5),
                bboxprops=dict(
                    edgecolor=edge_color,
                    facecolor="white",
                    linewidth=2,
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
    os.makedirs("../../results/figures/", exist_ok=True)
    plt.savefig(f"../../results/figures/{fig_name}", dpi=300, bbox_inches="tight")
    #plt.show()


if __name__ == "__main__":
    # Create full match networks for FC Barcelona
    create_team_figure(
        team_name="Barcelona",
        edge_color="#f2e691",
        title_main="FC Barcelona Passing Network",
        fig_name="barcelona_passing_network.png",
    )

    # Create full match networks for Manchester United
    create_team_figure(
        team_name="Manchester United",
        edge_color="black",
        title_main="Manchester United Passing Network",
        fig_name="manchester_united_passing_network.png",
    )
