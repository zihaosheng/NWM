"""Generate a video from the bundled nuScenes example."""

import argparse
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import torch

from nwm.conditioning import load_sample
from nwm.loading import load_pipeline


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, default=ROOT / "examples/nuscenes/sample.json")
    parser.add_argument("--weights", type=Path, default=ROOT / "configs/weights.json")
    parser.add_argument("--output", type=Path, default=ROOT / "output.mp4")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--decode-chunk-size", type=int, default=8)
    args = parser.parse_args()
    if args.steps < 1 or args.decode_chunk_size < 1:
        parser.error("--steps and --decode-chunk-size must be positive.")

    image, boxes, trajectory = load_sample(args.sample)
    pipeline = load_pipeline(args.weights, args.device)
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    frames = pipeline(
        image,
        boxes,
        trajectory,
        generator=generator,
        num_inference_steps=args.steps,
        decode_chunk_size=args.decode_chunk_size,
    )
    if frames.shape != (25, 320, 512, 3) or not np.isfinite(frames).all():
        raise RuntimeError("The model returned invalid video frames.")
    pixels = (np.clip(frames, 0, 1) * 255).round().astype(np.uint8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(
        str(args.output),
        (512, 320),
        fps=6,
        codec="libx264",
        pix_fmt_out="yuv420p",
        output_params=["-crf", "18", "-movflags", "+faststart"],
    )
    writer.send(None)
    try:
        for frame in pixels:
            writer.send(np.ascontiguousarray(frame))
    finally:
        writer.close()
    print(f"Saved {args.output.resolve()}")


if __name__ == "__main__":
    main()
