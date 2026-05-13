#!/usr/bin/env python3

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert stage-2 HumanLift multi-view frames into 3-gs_recon input format."
    )
    parser.add_argument("--frames-dir", required=True, help="Directory containing frame_XX.(png|jpg) files.")
    parser.add_argument("--output-dir", required=True, help="Target case directory under 3-gs_recon/data.")
    parser.add_argument("--num-views", type=int, default=81, help="Expected number of input views.")
    parser.add_argument("--canvas-size", type=int, default=832, help="Square RGBA canvas size.")
    parser.add_argument(
        "--alpha-mode",
        choices=("rembg", "keep"),
        default="rembg",
        help="Use rembg to estimate alpha, or keep existing alpha from RGBA inputs.",
    )
    parser.add_argument("--rembg-model", default="birefnet-general-lite", help="rembg session model name.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output directory.")
    return parser.parse_args()


def sorted_frame_paths(frames_dir: Path):
    paths = sorted(frames_dir.glob("frame_*.png")) + sorted(frames_dir.glob("frame_*.jpg")) + sorted(frames_dir.glob("frame_*.jpeg"))
    paths = sorted(paths)
    return paths


def ensure_output_dir(output_dir: Path, overwrite: bool):
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} already exists. Use --overwrite to replace it.")
        shutil.rmtree(output_dir)
    (output_dir / "images").mkdir(parents=True, exist_ok=True)


def rgba_from_image(path: Path, alpha_mode: str, rembg_session):
    image = Image.open(path)
    if image.mode == "RGBA" and alpha_mode == "keep":
        return image

    rgb = image.convert("RGB")
    if alpha_mode == "keep":
        rgba = Image.new("RGBA", rgb.size)
        rgba.paste(rgb, (0, 0))
        return rgba

    from rembg import remove

    rgba_np = remove(np.array(rgb), post_process_mask=True, session=rembg_session)
    return Image.fromarray(rgba_np).convert("RGBA")


def paste_on_canvas(rgba: Image.Image, canvas_size: int):
    canvas = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    width, height = rgba.size
    offset = ((canvas_size - width) // 2, (canvas_size - height) // 2)
    canvas.paste(rgba, offset, rgba)
    return canvas


def main():
    args = parse_args()
    frames_dir = Path(args.frames_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    frame_paths = sorted_frame_paths(frames_dir)
    if len(frame_paths) != args.num_views:
        raise RuntimeError(f"Expected {args.num_views} frames in {frames_dir}, found {len(frame_paths)}")

    ensure_output_dir(output_dir, args.overwrite)

    rembg_session = None
    if args.alpha_mode == "rembg":
        from rembg.session_factory import new_session

        rembg_session = new_session(args.rembg_model)

    for view_idx, frame_path in enumerate(frame_paths):
        rgba = rgba_from_image(frame_path, args.alpha_mode, rembg_session)
        canvas = paste_on_canvas(rgba, args.canvas_size)
        out_path = output_dir / "images" / f"lgt0_r_{view_idx:04d}.png"
        canvas.save(out_path)

    metadata = {
        "source_frames_dir": str(frames_dir),
        "num_views": args.num_views,
        "canvas_size": args.canvas_size,
        "alpha_mode": args.alpha_mode,
        "rembg_model": args.rembg_model,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (output_dir / "transforms_train.json").write_text("{}\n")
    (output_dir / "transforms_test.json").write_text("{}\n")

    print(f"Prepared {args.num_views} views in {output_dir}")


if __name__ == "__main__":
    main()
