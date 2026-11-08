"""Read one initial image, ego trajectory, and all bounding-box conditions."""

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


# RGB class colors used by the checkpoint's bounding-box renderer.
CLASS_COLORS = (
    (165, 42, 42),
    (0, 0, 255),
    (200, 44, 247),
    (255, 0, 0),
    (247, 162, 44),
    (250, 255, 2),
    (255, 209, 222),
    (204, 153, 255),
    (128, 128, 128),
    (255, 204, 204),
    (0, 255, 0),
)


def render_boxes(frames, track_colors):
    """Rasterize boxes in their supplied order, from farthest to nearest."""
    images = []
    for boxes in frames:
        canvas = np.zeros((320, 512, 3), dtype=np.uint8)
        for box in boxes:
            coords = np.asarray(box["box"], dtype=np.float64)
            if (
                coords.shape != (4,)
                or not np.isfinite(coords).all()
                or np.any(coords < 0)
                or np.any(coords > 1)
                or coords[2] <= coords[0]
                or coords[3] <= coords[1]
            ):
                raise ValueError("Each box must contain normalized [x1, y1, x2, y2] coordinates.")
            x1, y1, x2, y2 = (coords * [512, 320, 512, 320]).astype(int)
            border = min(max(min(x2 - x1, y2 - y1) * 0.1, 2), 8)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), CLASS_COLORS[box["class_id"]], cv2.FILLED)
            color = tuple(track_colors[str(box["track_id"])])
            cv2.rectangle(
                canvas,
                (int(x1 + border), int(y1 + border)),
                (int(x2 - border), int(y2 - border)),
                color,
                cv2.FILLED,
            )
        images.append(canvas)
    pixels = np.stack(images)
    return torch.from_numpy(pixels).permute(0, 3, 1, 2).float() / 127.5 - 1.0


def load_sample(sample_path):
    sample_path = Path(sample_path).resolve()
    sample = json.loads(sample_path.read_text())
    trajectory = np.asarray(sample["ego_trajectory"], dtype=np.float32)
    if trajectory.shape != (25, 2) or not np.isfinite(trajectory).all():
        raise ValueError(
            "ego_trajectory must contain 25 finite [right, forward] positions in meters."
        )
    if len(sample["bounding_boxes"]) != 25 or sample["fps"] != 6:
        raise ValueError("The sample must contain 25 bounding-box frames at 6 FPS.")
    with Image.open(sample_path.parent / sample["image"]) as source:
        image = source.convert("RGB").resize((512, 320), Image.Resampling.BILINEAR)
    boxes = render_boxes(sample["bounding_boxes"], sample["track_colors"])
    ego = torch.from_numpy(trajectory[[5, 10, 15, 20]].reshape(8))
    return image, boxes, ego
