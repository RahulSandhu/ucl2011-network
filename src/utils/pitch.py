from matplotlib.patches import Circle, Rectangle


# Draw vertical green pitch
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

    # Penalty area
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-5, 85)
    ax.set_ylim(-5, 125)
