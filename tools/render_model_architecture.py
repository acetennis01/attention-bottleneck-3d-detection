from pathlib import Path
import math
from PIL import Image, ImageDraw, ImageFont, ImageOps


SCALE = 2
WIDTH, HEIGHT = 1100, 1600
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "docs"
OUT = DOCS_DIR / "modelarchitecture.png"
LIDAR_INPUT = DOCS_DIR / "lidar_point_cloud_input.png"
CAMERA_INPUT = DOCS_DIR / "camera_image_input.png"

REGULAR = "/System/Library/Fonts/Supplemental/Arial.ttf"
BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
ITALIC = "/System/Library/Fonts/Supplemental/Arial Italic.ttf"


def p(value):
    if isinstance(value, list):
        return [p(item) for item in value]
    if isinstance(value, tuple):
        return tuple(int(v * SCALE) for v in value)
    return int(value * SCALE)


def font(size, bold=False):
    return ImageFont.truetype(BOLD if bold else REGULAR, p(size))


image = Image.new("RGB", p((WIDTH, HEIGHT)), "white")
draw = ImageDraw.Draw(image)

COLORS = {
    "text": "#172027",
    "line": "#334047",
    "camera_fill": "#fff3df",
    "camera_stroke": "#d77900",
    "lidar_fill": "#eaf4ff",
    "lidar_stroke": "#2878b5",
    "fusion_fill": "#f3ebff",
    "fusion_stroke": "#6a3dad",
    "neutral_fill": "#f5f7f8",
    "neutral_stroke": "#59636b",
    "output_fill": "#eaf8ee",
    "output_stroke": "#2f7d44",
}


def centered_text(cx, cy, lines, sizes, bolds=None, gap=4, color=None):
    if isinstance(lines, str):
        lines = [lines]
    if isinstance(sizes, int):
        sizes = [sizes] * len(lines)
    if bolds is None:
        bolds = [False] * len(lines)
    fonts = [font(size, weight) for size, weight in zip(sizes, bolds)]
    metrics = []
    for line, used_font in zip(lines, fonts):
        bounds = draw.textbbox((0, 0), line, font=used_font)
        metrics.append((bounds, bounds[3] - bounds[1]))
    total_height = sum(height for _, height in metrics) + p(gap) * (len(lines) - 1)
    y = p(cy) - total_height // 2
    for line, used_font, (bounds, height) in zip(lines, fonts, metrics):
        width = bounds[2] - bounds[0]
        draw.text(
            (p(cx) - width // 2, y - bounds[1]), line,
            font=used_font, fill=color or COLORS["text"])
        y += height + p(gap)


def box(x, y, w, h, category, lines, sizes, bolds, gap=4, radius=10):
    draw.rounded_rectangle(
        p((x, y, x + w, y + h)), radius=p(radius),
        fill=COLORS[f"{category}_fill"],
        outline=COLORS[f"{category}_stroke"], width=p(2))
    centered_text(x + w / 2, y + h / 2, lines, sizes, bolds, gap)


def arrow(x1, y1, x2, y2, color=None, width=2.2, dashed=False):
    color = color or COLORS["line"]
    start, end = p((x1, y1)), p((x2, y2))
    if dashed:
        distance = max(abs(end[0] - start[0]), abs(end[1] - start[1]))
        pieces = max(1, distance // p(12))
        for index in range(0, pieces, 2):
            a = index / pieces
            b = min((index + 1) / pieces, 1)
            draw.line(
                (
                    int(start[0] + (end[0] - start[0]) * a),
                    int(start[1] + (end[1] - start[1]) * a),
                    int(start[0] + (end[0] - start[0]) * b),
                    int(start[1] + (end[1] - start[1]) * b),
                ),
                fill=color, width=p(width))
    else:
        draw.line((*start, *end), fill=color, width=p(width))

    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    length, half = p(10), p(5)
    base_x = end[0] - length * math.cos(angle)
    base_y = end[1] - length * math.sin(angle)
    left = (
        base_x + half * math.sin(angle),
        base_y - half * math.cos(angle),
    )
    right = (
        base_x - half * math.sin(angle),
        base_y + half * math.cos(angle),
    )
    draw.polygon([end, left, right], fill=color)


def routed_arrow(points, color=None, width=2.2):
    color = color or COLORS["line"]
    scaled = [p(point) for point in points]
    draw.line(scaled, fill=color, width=p(width), joint="curve")
    (x1, y1), (x2, y2) = points[-2], points[-1]
    arrow(x1, y1, x2, y2, color, width)


def token_row(x, y, count, color, labels, cell=36, gap=8):
    positions = []
    for index in range(count):
        left = x + index * (cell + gap)
        draw.rounded_rectangle(
            p((left, y, left + cell, y + cell)), radius=p(4),
            fill=color, outline="#48535a", width=p(1.4))
        centered_text(left + cell / 2, y + cell / 2, labels[index], 12, [True])
        positions.append((left + cell / 2, y + cell / 2))
    return positions


def arrow_label(cx, cy, text, width, size=11):
    draw.rounded_rectangle(
        p((cx - width / 2, cy - 11, cx + width / 2, cy + 11)),
        radius=p(4), fill="white")
    centered_text(cx, cy, text, size, [True])


def arrow_math_label(
        cx, cy, prefix, base, subscript, width, superscript=None, size=11):
    """Draw a diagram label with real typographic sub/superscripts."""
    half_height = max(11, size)
    draw.rounded_rectangle(
        p((cx - width / 2, cy - half_height,
           cx + width / 2, cy + half_height)),
        radius=p(4), fill="white")
    prefix_font = font(size, True)
    base_font = ImageFont.truetype(ITALIC, p(size + 1))
    script_font = font(max(8, size - 3))
    prefix_width = draw.textlength(prefix, font=prefix_font)
    base_width = draw.textlength(base, font=base_font)
    sub_width = draw.textlength(subscript, font=script_font)
    sup_width = (
        draw.textlength(superscript, font=script_font)
        if superscript else 0)
    gap = p(5)
    total_width = prefix_width + gap + base_width + max(sub_width, sup_width)
    left = p(cx) - total_width / 2
    middle = p(cy)
    draw.text((left, middle), prefix, font=prefix_font,
              fill=COLORS["text"], anchor="lm")
    left += prefix_width + gap
    draw.text((left, middle), base, font=base_font,
              fill=COLORS["text"], anchor="lm")
    script_x = left + base_width
    draw.text((script_x, middle + p(4)), subscript, font=script_font,
              fill=COLORS["text"], anchor="lm")
    if superscript:
        draw.text((script_x, middle - p(5)), superscript, font=script_font,
                  fill=COLORS["text"], anchor="lm")


def draw_sensor_image(source, x, y, w, h):
    draw.rectangle(p((x, y, x + w, y + h)), fill="white")
    inset = p(7)
    available = (p(w) - 2 * inset, p(h) - 2 * inset)
    with Image.open(source) as source_image:
        fitted = ImageOps.contain(source_image.convert("RGB"), available)
    paste_x = p(x) + (p(w) - fitted.width) // 2
    paste_y = p(y) + (p(h) - fitted.height) // 2
    image.paste(fitted, (paste_x, paste_y))


# Output stack at the top.
centered_text(550, 32, "3D object detection output", 17, [True])
box(390, 52, 320, 68, "output", ["3D boxes · classes · scores", "Car · Pedestrian · Cyclist"], [18, 14], [True, False])
arrow(550, 145, 550, 120)
box(430, 145, 240, 70, "output", ["PointPillars", "3D detection head"], [19, 17], [True, True])
arrow(550, 295, 550, 215)
arrow_math_label(550, 255, "Fused LiDAR BEV", "F", "fused", 240, size=14)
box(365, 295, 370, 72, "neutral", ["Residual fusion with original LiDAR BEV", "1 × 1 projection to 384 channels"], [17, 14], [True, False])
arrow(550, 445, 550, 367)
arrow_math_label(
    550, 406, "Updated LiDAR tokens", "x̂", "lidar", 280, size=14)
box(385, 445, 330, 72, "fusion", ["LiDAR readout cross-attention", "LiDAR queries; bottleneck keys/values"], [18, 14], [True, False])

# Symmetric MBT block.
draw.rounded_rectangle(
    p((225, 585, 875, 890)), radius=p(15),
    fill=COLORS["fusion_fill"], outline=COLORS["fusion_stroke"], width=p(2.5))
centered_text(550, 608, "Symmetric attention-bottleneck fusion", 20, [True])
centered_text(550, 637, "8 attention heads per layer", 14)

box(275, 755, 245, 70, "lidar", ["LiDAR transformer", "LiDAR tokens + shared tokens"], [17, 13], [True, False])
box(580, 755, 245, 70, "camera", ["Camera transformer", "camera tokens + shared tokens"], [17, 13], [True, False])

# Four shared bottleneck tokens are shown explicitly.
centered_text(550, 666, "shared bottleneck state", 13)
token_row(454, 682, 4, "#9d59c7", ["B1", "B2", "B3", "B4"], cell=38, gap=10)

# Route the fusion-block output directly to the LiDAR-query readout.
arrow(550, 585, 550, 517, COLORS["fusion_stroke"])
arrow_math_label(
    550, 551, "Final bottleneck state", "x", "fusion", 300,
    superscript="L", size=14)

# Training-only camera readout. Camera tokens query the same final bottleneck
# state used by the LiDAR readout; its output supervises BEV center prediction.
routed_arrow(
    [(550, 575), (850, 575), (850, 481), (875, 481)],
    COLORS["fusion_stroke"], 1.8)
box(
    875, 445, 215, 72, "camera",
    ["Camera readout cross-attention", "camera queries; bottleneck K/V"],
    [13, 10], [True, False], 3, 8)
arrow(982.5, 517, 982.5, 535, COLORS["neutral_stroke"], 1.8)
box(
    905, 535, 155, 50, "neutral",
    ["Auxiliary center loss", "training only"],
    [11, 10], [True, False], 2, 7)

# Parallel updates into the shared state.
routed_arrow([(397, 755), (397, 735), (500, 735), (500, 720)], COLORS["fusion_stroke"])
routed_arrow([(703, 755), (703, 735), (600, 735), (600, 720)], COLORS["fusion_stroke"])
# Shared state returns to both branches, forming two visible loops.
routed_arrow([(454, 701), (250, 701), (250, 790), (275, 790)], COLORS["fusion_stroke"])
routed_arrow([(646, 701), (850, 701), (850, 790), (825, 790)], COLORS["fusion_stroke"])

# Repeated layer loop.
draw.line(
    p([(350, 848), (750, 848), (750, 875), (350, 875)]),
    fill=COLORS["fusion_stroke"], width=p(2), joint="curve")
arrow(350, 875, 350, 848, COLORS["fusion_stroke"], 2)
centered_text(550, 861, "repeat bottleneck update across 4 fusion layers", 13)

# Token row beneath the MBT block, matching the reference layout.
centered_text(275, 948, "LiDAR tokens (16 × 16, 128-D)", 14, [True])
centered_text(550, 948, "4 fusion tokens", 14, [True])
centered_text(825, 948, "Camera tokens (16 × 16, 128-D)", 14, [True])
token_row(142, 968, 6, "#7fb1df", ["L1", "L2", "L3", "…", "L255", "L256"], cell=34, gap=8)
token_row(467, 968, 4, "#9d59c7", ["B1", "B2", "B3", "B4"], cell=34, gap=8)
token_row(692, 968, 6, "#efbd6d", ["C1", "C2", "C3", "…", "C255", "C256"], cell=34, gap=8)

routed_arrow([(275, 930), (275, 910), (397, 910), (397, 825)], COLORS["lidar_stroke"])
arrow(550, 930, 550, 720, COLORS["fusion_stroke"])
routed_arrow([(825, 930), (825, 910), (703, 910), (703, 825)], COLORS["camera_stroke"])

# Encoders and aligned representations.
arrow(275, 1070, 275, 1010, COLORS["lidar_stroke"])
box(145, 1070, 260, 70, "lidar", ["Projection + BEV pooling", "16 × 16 tokens"], [17, 14], [True, False])
arrow(275, 1170, 275, 1140, COLORS["lidar_stroke"])
box(145, 1170, 260, 72, "lidar", ["SECOND + SECONDFPN", "384-channel LiDAR BEV"], [17, 14], [True, False])
arrow(275, 1275, 275, 1242, COLORS["lidar_stroke"])
box(145, 1275, 260, 70, "lidar", ["PillarFeatureNet", "pillar encoding + BEV scatter"], [17, 14], [True, False])

arrow(825, 1070, 825, 1010, COLORS["camera_stroke"])
box(695, 1070, 260, 78, "camera", ["Camera-to-BEV alignment", "KITTI calibration · 4 height queries"], [17, 13], [True, False])
arrow(825, 1200, 825, 1148, COLORS["camera_stroke"])
box(695, 1200, 260, 72, "camera", ["ResNet-18", "ImageNet pretrained"], [18, 14], [True, False])

# Raw sensor inputs at the bottom.
arrow(275, 1375, 275, 1345, COLORS["lidar_stroke"])
draw_sensor_image(LIDAR_INPUT, 95, 1375, 360, 155)
centered_text(275, 1553, "LiDAR point cloud", 17, [True])
centered_text(275, 1577, "x, y, z, intensity", 13)

arrow(825, 1375, 825, 1272, COLORS["camera_stroke"])
draw_sensor_image(CAMERA_INPUT, 645, 1375, 360, 155)
centered_text(825, 1553, "RGB camera image", 17, [True])
centered_text(825, 1577, "maximum 1280 × 384", 13)

# Original LiDAR BEV skip connection to residual fusion.
routed_arrow(
    [(145, 1206), (55, 1206), (55, 331), (365, 331)],
    COLORS["lidar_stroke"], 2)
draw.rectangle(p((63, 343, 286, 369)), fill="white")
centered_text(174, 356, "original LiDAR BEV residual", 12)

image.save(OUT, dpi=(300, 300))
print(OUT)
