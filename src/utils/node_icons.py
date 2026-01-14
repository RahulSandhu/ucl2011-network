from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image, ImageDraw


# Create circular masked image
def create_circular_image(img_path, size=80):
    # Load and process image
    img = Image.open(img_path)
    img = img.convert("RGBA")

    # Resize to base size
    img = img.resize((size, size), Image.Resampling.LANCZOS)

    # Create circular mask
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, size, size), fill=255)

    # Apply circular mask
    output = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    output.paste(img, (0, 0))
    output.putalpha(mask)

    return output


# Add image node to plot
def add_image_node(ax, img_path, x, y, zoom, edge_color):
    # Create circular image
    circular_img = create_circular_image(img_path)

    # Create image box
    imagebox = OffsetImage(circular_img, zoom=zoom)

    # Add image to plot
    ab = AnnotationBbox(
        imagebox,
        (x, y),
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
