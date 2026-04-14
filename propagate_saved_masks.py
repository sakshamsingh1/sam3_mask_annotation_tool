import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from decord import VideoReader, cpu

SAM3_ROOT = Path("sam3")
if not SAM3_ROOT.exists():
    raise FileNotFoundError(f"SAM3 checkout not found at {SAM3_ROOT}")
if str(SAM3_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM3_ROOT))

try:
    from sam3.model_builder import build_sam3_video_model
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Failed to import SAM3. Install the SAM3 runtime dependencies "
        "(for example `iopath`) in this environment."
    ) from exc

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MASK_SAVE_ROOT = Path("masks")
INPUT_CLIPS_DIR = Path(
    "clips_ytambi_h264"
)


def get_video_predictor():
    sam3_model = build_sam3_video_model(device=DEVICE)
    predictor = sam3_model.tracker
    predictor.backbone = sam3_model.detector.backbone
    return predictor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Propagate saved first-frame SAM3 masks across videos."
    )
    parser.add_argument(
        "--video-name",
        help="Only process a single video directory under data/masks and clips_h264.",
    )
    parser.add_argument(
        "--prompt-name",
        help="Only process a single prompt directory name inside the selected video.",
    )
    return parser.parse_args()


def iter_seed_jobs(video_name=None, prompt_name=None):
    if not MASK_SAVE_ROOT.exists():
        return

    for video_dir in sorted(MASK_SAVE_ROOT.iterdir()):
        if not video_dir.is_dir():
            continue
        if video_name and video_dir.name != video_name:
            continue

        video_path = INPUT_CLIPS_DIR / video_dir.name
        if not video_path.is_file():
            print(f"Skipping {video_dir}: missing video {video_path}")
            continue

        for prompt_dir in sorted(video_dir.iterdir()):
            if not prompt_dir.is_dir():
                continue
            if prompt_name and prompt_dir.name != prompt_name:
                continue

            seed_path = prompt_dir / "mask_0000.png"
            if seed_path.is_file():
                yield video_path, prompt_dir, seed_path


def load_seed_mask(seed_path):
    mask = cv2.imread(str(seed_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Failed to read seed mask: {seed_path}")
    return torch.from_numpy((mask > 127).astype(np.float32))


def get_num_frames(video_path):
    vr = VideoReader(str(video_path), ctx=cpu(0))
    num_frames = len(vr)
    del vr
    return int(num_frames)


def has_complete_mask_sequence(prompt_dir, num_frames):
    mask_paths = sorted(prompt_dir.glob("mask_*.png"))
    if len(mask_paths) != num_frames:
        return False

    expected_names = [f"mask_{idx:04d}.png" for idx in range(num_frames)]
    actual_names = [path.name for path in mask_paths]
    return actual_names == expected_names


def remove_existing_masks(prompt_dir):
    for mask_path in sorted(prompt_dir.glob("mask_*.png")):
        mask_path.unlink()


def save_mask_sequence(prompt_dir, masks):
    remove_existing_masks(prompt_dir)
    for frame_idx, mask in enumerate(masks):
        mask_path = prompt_dir / f"mask_{frame_idx:04d}.png"
        if not cv2.imwrite(str(mask_path), mask):
            raise RuntimeError(f"Failed to save propagated mask to {mask_path}")


def propagate_prompt(video_predictor, video_path, prompt_dir, seed_path):
    inference_state = video_predictor.init_state(video_path=str(video_path))
    seed_mask = load_seed_mask(seed_path)

    video_predictor.add_new_mask(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=1,
        mask=seed_mask,
    )

    num_frames = int(inference_state["num_frames"])
    propagated_masks = [None] * num_frames

    for out_frame_idx, out_obj_ids, low_res_masks, video_res_masks, obj_scores in (
        video_predictor.propagate_in_video(
            inference_state=inference_state,
            start_frame_idx=0,
            max_frame_num_to_track=max(num_frames - 1, 0),
            reverse=False,
            propagate_preflight=True,
        )
    ):
        frame_mask = None
        for mask_logits in video_res_masks:
            mask = (mask_logits.squeeze().detach().cpu().numpy() > 0).astype(np.uint8)
            mask = mask * 255
            if frame_mask is None:
                frame_mask = mask
            else:
                frame_mask = np.maximum(frame_mask, mask)
        if frame_mask is None:
            raise RuntimeError(
                f"No propagated mask returned for frame {out_frame_idx} in {video_path}"
            )
        propagated_masks[out_frame_idx] = frame_mask

    missing_frames = [idx for idx, mask in enumerate(propagated_masks) if mask is None]
    if missing_frames:
        raise RuntimeError(
            f"Missing propagated outputs for frames {missing_frames[:10]} in {video_path}"
        )

    save_mask_sequence(prompt_dir, propagated_masks)


def main():
    args = parse_args()
    jobs = list(iter_seed_jobs(video_name=args.video_name, prompt_name=args.prompt_name))
    if not jobs:
        print("No seed masks found to propagate.")
        return
    pending_jobs = []

    for video_path, prompt_dir, seed_path in jobs:
        num_frames = get_num_frames(video_path)
        if has_complete_mask_sequence(prompt_dir, num_frames):
            print(f"Skipping {prompt_dir}: found complete mask sequence ({num_frames} frames).")
            continue
        pending_jobs.append((video_path, prompt_dir, seed_path))

    if not pending_jobs:
        print("All matching prompt directories already contain complete mask sequences.")
        return

    video_predictor = get_video_predictor()

    for video_path, prompt_dir, seed_path in pending_jobs:
        print(f"Propagating {seed_path} on {video_path.name}")
        propagate_prompt(video_predictor, video_path, prompt_dir, seed_path)
        print(f"Saved propagated masks to {prompt_dir}")


if __name__ == "__main__":
    main()
