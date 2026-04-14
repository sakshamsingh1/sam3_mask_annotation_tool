import os
import sys
import time
import random
from pathlib import Path
from contextlib import nullcontext

APP_ROOT = Path(__file__).resolve().parent
GRADIO_TEMP_DIR = APP_ROOT / "gradio_temp_dir"
GRADIO_TEMP_DIR.mkdir(parents=True, exist_ok=True)
os.environ["GRADIO_TEMP_DIR"] = str(GRADIO_TEMP_DIR)
os.environ["TMPDIR"] = str(GRADIO_TEMP_DIR)
os.environ["TEMP"] = str(GRADIO_TEMP_DIR)
os.environ["TMP"] = str(GRADIO_TEMP_DIR)

import cv2
import gradio as gr
import numpy as np
from PIL import Image
import torch
from decord import VideoReader, cpu
from moviepy.editor import ImageSequenceClip

SAM3_ROOT = APP_ROOT / "sam3"
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

COLOR_PALETTE = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 128, 0),
    (128, 0, 255),
    (0, 128, 255),
    (128, 255, 0),
]

W = 768
H = W
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE == "cuda" and torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

MASK_SAVE_ROOT = APP_ROOT / "masks"
INPUT_CLIPS_DIR = APP_ROOT / "clips_ytambi_h264"
VIDEO_PATHS = [path for path in sorted(INPUT_CLIPS_DIR.glob("*")) if path.is_file()]
if not VIDEO_PATHS:
    raise FileNotFoundError(f"No videos found in {INPUT_CLIPS_DIR}")

_CLICK_PREDICTOR = None
_TEXT_PREDICTOR = None


def autocast_context():
    if DEVICE == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def get_click_predictor():
    global _CLICK_PREDICTOR
    if _CLICK_PREDICTOR is None:
        sam3_model = build_sam3_video_model(device=DEVICE)
        predictor = sam3_model.tracker
        predictor.backbone = sam3_model.detector.backbone
        _CLICK_PREDICTOR = predictor
    return _CLICK_PREDICTOR


def get_text_predictor():
    global _TEXT_PREDICTOR
    if _TEXT_PREDICTOR is None:
        _TEXT_PREDICTOR = build_sam3_video_model(device=DEVICE)
    return _TEXT_PREDICTOR


def resize_frame(frame):
    if frame.shape[0] > frame.shape[1]:
        out_w = W
        out_h = int(out_w * frame.shape[0] / frame.shape[1])
    else:
        out_h = H
        out_w = int(out_h * frame.shape[1] / frame.shape[0])
    resized = cv2.resize(frame, (out_w, out_h))
    return resized, out_w, out_h


def decord_frame_to_numpy(frame):
    if hasattr(frame, "asnumpy"):
        return frame.asnumpy()
    if hasattr(frame, "numpy"):
        return frame.numpy()
    return np.asarray(frame)


def render_mask_overlay(frame, mask, input_points=None, input_labels=None):
    color = np.array(COLOR_PALETTE[int(time.time()) % len(COLOR_PALETTE)], dtype=np.float32) / 255.0
    color = color[None, None, :]
    org_image = frame.astype(np.float32) / 255.0
    mask = mask.astype(np.float32)
    if mask.ndim == 2:
        mask = mask[:, :, None]

    painted_image = (1 - mask * 0.5) * org_image + mask * 0.5 * color
    painted_image = np.uint8(np.clip(painted_image * 255, 0, 255))

    if input_points is not None and input_labels is not None:
        for point, label in zip(input_points, input_labels):
            color_bgr = (0, 0, 255) if label == 0 else (255, 0, 0)
            cv2.circle(painted_image, tuple(point), radius=3, color=color_bgr, thickness=-1)

    return painted_image


def get_unique_prompt_dir(video_path, prompt_dirname):
    video_dir = MASK_SAVE_ROOT / Path(video_path).name
    save_dir = video_dir / prompt_dirname
    if not save_dir.exists():
        return save_dir

    suffix = 2
    while True:
        candidate = video_dir / f"{prompt_dirname}_{suffix}"
        if not candidate.exists():
            return candidate
        suffix += 1


def get_video_title(video_idx):
    return f"{video_idx + 1}/{len(VIDEO_PATHS)}: {VIDEO_PATHS[video_idx].name}"


def current_preview_image(video_state):
    if video_state["origin_images"]:
        return Image.fromarray(video_state["origin_images"][0])
    return None


def clear_prompt_state(video_state):
    video_state["masks"] = None
    video_state["click_inference_state"] = None
    video_state["text_inference_state"] = None
    video_state["preview_video_path"] = None
    video_state["query_text"] = None
    video_state["input_points"] = []
    video_state["input_labels"] = []
    video_state["scaled_points"] = []
    video_state["frame_idx"] = 0


def set_current_video(video_idx, video_state):
    video_idx = int(video_idx) % len(VIDEO_PATHS)
    clear_prompt_state(video_state)
    video_state["video_index"] = video_idx
    video_state["video_path"] = str(VIDEO_PATHS[video_idx])
    video_state["origin_images"] = None


def read_display_frames(video_path, n_frames=None):
    vr = VideoReader(video_path, ctx=cpu(0))
    frame_count = len(vr) if n_frames is None else min(len(vr), n_frames)
    frames = [decord_frame_to_numpy(vr[i]) for i in range(frame_count)]
    del vr
    return [resize_frame(frame)[0] for frame in frames]


def combine_output_masks(outputs):
    out_binary_masks = outputs.get("out_binary_masks")
    if out_binary_masks is None:
        return None

    masks = np.asarray(out_binary_masks)
    if masks.size == 0:
        return None
    if masks.ndim == 2:
        masks = masks[None, ...]

    return np.any(masks.astype(bool), axis=0).astype(np.float32)


def write_preview_video(frames, fps=15):
    video_file = GRADIO_TEMP_DIR / f"{time.time()}-{random.random()}-dir_preview.mp4"
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(str(video_file), codec="libx264", audio=False, verbose=False, logger=None)
    return str(video_file)


def load_current_video(video_state):
    video_path = video_state.get("video_path")
    if not video_path:
        gr.Warning("No video is selected.")
        return None, None, None, ""

    first_frame = read_display_frames(video_path, n_frames=1)[0]
    video_state["origin_images"] = [first_frame]

    if video_state["prompt_mode"] == "Click":
        with torch.inference_mode(), autocast_context():
            video_state["click_inference_state"] = get_click_predictor().init_state(video_path=video_path)

    return video_path, get_video_title(video_state["video_index"]), Image.fromarray(first_frame), ""


def load_initial_video(video_state):
    set_current_video(0, video_state)
    return load_current_video(video_state)


def step_and_load_video(delta, video_state):
    current_idx = int(video_state.get("video_index", 0))
    set_current_video(current_idx + delta, video_state)
    return load_current_video(video_state)


def switch_prompt_mode(prompt_mode, video_state):
    video_state["prompt_mode"] = prompt_mode
    clear_prompt_state(video_state)
    image = None
    if video_state["video_path"]:
        first_frame = read_display_frames(video_state["video_path"], n_frames=1)[0]
        video_state["origin_images"] = [first_frame]
        image = Image.fromarray(first_frame)
        if prompt_mode == "Click":
            with torch.inference_mode(), autocast_context():
                video_state["click_inference_state"] = get_click_predictor().init_state(
                    video_path=video_state["video_path"]
                )

    return (
        gr.update(visible=prompt_mode == "Click"),
        gr.update(visible=prompt_mode == "Text"),
        image,
        None,
        "",
    )


def extract_first_frame(video_state):
    if not video_state.get("video_path"):
        gr.Warning("No video is selected.")
        return None, None, ""

    first_frame = read_display_frames(video_state["video_path"], n_frames=1)[0]
    clear_prompt_state(video_state)
    video_state["origin_images"] = [first_frame]
    with torch.inference_mode(), autocast_context():
        video_state["click_inference_state"] = get_click_predictor().init_state(
            video_path=video_state["video_path"]
        )
    return Image.fromarray(first_frame), None, ""


def segment_frame(evt: gr.SelectData, prompt_mode, click_label, video_state):
    if prompt_mode != "Click":
        gr.Warning('Switch to "Click" mode to annotate with points.')
        return None
    if video_state["origin_images"] is None or video_state["click_inference_state"] is None:
        gr.Warning("The first frame has not been loaded for click prompting.")
        return None

    x, y = evt.index
    new_point = [x, y]
    label_value = 1 if click_label == "Positive" else 0
    video_state["input_points"].append(new_point)
    video_state["input_labels"].append(label_value)

    height, width = video_state["origin_images"][0].shape[:2]
    video_state["scaled_points"] = [
        [pt[0] / width, pt[1] / height] for pt in video_state["input_points"]
    ]

    with torch.inference_mode(), autocast_context():
        _, _, _, video_res_masks = get_click_predictor().add_new_points_or_box(
            inference_state=video_state["click_inference_state"],
            frame_idx=video_state["frame_idx"],
            obj_id=video_state["obj_id"],
            points=video_state["scaled_points"],
            labels=video_state["input_labels"],
            clear_old_points=True,
            rel_coordinates=True,
        )

    mask = (video_res_masks[0].squeeze().detach().cpu().numpy() > 0).astype(np.float32)
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    painted_image = render_mask_overlay(
        video_state["origin_images"][0],
        mask,
        video_state["input_points"],
        video_state["input_labels"],
    )
    video_state["masks"] = [mask[:, :, None]]
    video_state["preview_video_path"] = None
    return Image.fromarray(painted_image)


def clear_clicks(video_state):
    video_state["input_points"] = []
    video_state["input_labels"] = []
    video_state["scaled_points"] = []
    video_state["masks"] = None
    video_state["preview_video_path"] = None
    video_state["click_inference_state"] = None

    if video_state["video_path"] is not None:
        with torch.inference_mode(), autocast_context():
            video_state["click_inference_state"] = get_click_predictor().init_state(
                video_path=video_state["video_path"]
            )

    return current_preview_image(video_state), None, ""


def preview_text_query(query_text, video_state):
    video_path = video_state.get("video_path")
    if not video_path:
        gr.Warning("No video is selected.")
        return None, None, ""

    prompt = (query_text or "").strip()
    if not prompt:
        gr.Warning("Please enter a text query first.")
        return None, None, ""

    display_images = read_display_frames(video_path, n_frames=None)
    if not display_images:
        gr.Warning("No frames were loaded from the selected video.")
        return None, None, ""

    clear_prompt_state(video_state)
    video_state["origin_images"] = display_images
    video_state["query_text"] = prompt

    output_frames = []
    mask_frames = []

    with torch.inference_mode(), autocast_context():
        inference_state = get_text_predictor().init_state(resource_path=video_path)
        _, preview_outputs = get_text_predictor().add_prompt(
            inference_state=inference_state,
            frame_idx=0,
            text_str=prompt,
        )

        first_mask = combine_output_masks(preview_outputs)
        if first_mask is None:
            video_state["text_inference_state"] = inference_state
            gr.Warning(f'No mask was found on frame 0 for query "{prompt}".')
            return Image.fromarray(display_images[0]), None, ""

        for out_frame_idx, outputs in get_text_predictor().propagate_in_video(
            inference_state=inference_state,
            start_frame_idx=0,
            max_frame_num_to_track=max(len(display_images) - 1, 0),
            reverse=False,
        ):
            if out_frame_idx >= len(display_images):
                break

            frame = display_images[out_frame_idx]
            mask = combine_output_masks(outputs)
            if mask is None:
                display_mask = np.zeros(frame.shape[:2], dtype=np.float32)
            else:
                display_mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

            mask_frames.append(display_mask[:, :, None])
            output_frames.append(render_mask_overlay(frame, display_mask))

    if not output_frames:
        gr.Warning("Text prompting did not return any preview frames.")
        return Image.fromarray(display_images[0]), None, ""

    video_state["text_inference_state"] = inference_state
    video_state["masks"] = mask_frames
    video_state["preview_video_path"] = write_preview_video(output_frames)
    return Image.fromarray(output_frames[0]), video_state["preview_video_path"], ""


def clear_text_query(video_state):
    video_state["text_inference_state"] = None
    video_state["query_text"] = None
    video_state["masks"] = None
    video_state["preview_video_path"] = None
    return current_preview_image(video_state), None, ""


def save_masks(save_text, prompt_mode, video_state):
    if video_state["masks"] is None:
        gr.Warning("Please create masks before saving.")
        return None
    if not video_state["video_path"]:
        gr.Warning("Please load a video before saving.")
        return None

    prompt = (save_text or video_state.get("query_text") or "").strip()
    if not prompt:
        gr.Warning("Please enter a save label, or use a text query first.")
        return None

    prompt_dirname = "_".join(prompt.split())
    save_dir = get_unique_prompt_dir(video_state["video_path"], prompt_dirname)
    save_dir.mkdir(parents=True, exist_ok=True)

    masks = np.asarray(video_state["masks"], dtype=np.float32)
    if prompt_mode == "Click":
        masks = masks[:1]

    for idx, mask in enumerate(masks):
        mask_image = (np.squeeze(mask) > 0.5).astype(np.uint8) * 255
        mask_path = save_dir / f"mask_{idx:04d}.png"
        if not cv2.imwrite(str(mask_path), mask_image):
            raise RuntimeError(f"Failed to save mask {idx} to {mask_path}.")

    return str(save_dir)


with gr.Blocks() as demo:
    video_state = gr.State(
        {
            "prompt_mode": "Click",
            "video_index": 0,
            "video_path": None,
            "origin_images": None,
            "masks": None,
            "preview_video_path": None,
            "click_inference_state": None,
            "text_inference_state": None,
            "input_points": [],
            "input_labels": [],
            "scaled_points": [],
            "query_text": None,
            "frame_idx": 0,
            "obj_id": 1,
        }
    )

    with gr.Column():
        current_video_title = gr.Textbox(label="Current Video", interactive=False)
        video_input = gr.Video(label="Selected Video", interactive=False, elem_id="my-video1")

        with gr.Row(elem_id="my-btn"):
            prev_video_btn = gr.Button("Previous Video")
            next_video_btn = gr.Button("Next Video")

        prompt_mode = gr.Radio(["Click", "Text"], value="Click", label="Prompt Mode")
        image_output = gr.Image(label="Preview Frame", interactive=True, elem_id="my-video")
        video_output = gr.Video(label="Preview Video", elem_id="my-video")

        demo.css = """
        #my-btn {
           width: 42% !important;
           max-width: 620px !important;
           margin: 0 auto;
        }

        #my-video1 {
           width: 42% !important;
           max-width: 620px !important;
           height: auto !important;
           margin: 0 auto;
        }

        #my-video {
           width: 42% !important;
           max-width: 620px !important;
           height: auto !important;
           margin: 0 auto;
        }
        """

        gr.Markdown("Click mode edits only the first frame. Text mode previews the whole video and saves masks for every frame.")

        with gr.Group(visible=True) as click_controls:
            with gr.Row(elem_id="my-btn"):
                extract_btn = gr.Button("Extract First Frame")
                click_label = gr.Radio(["Positive", "Negative"], label="Click Type", value="Positive")
                clear_click_btn = gr.Button("Clear Clicks")

        with gr.Group(visible=False) as text_controls:
            with gr.Row(elem_id="my-btn"):
                query_text = gr.Textbox(
                    label="Text Query",
                    placeholder='Enter a text query, for example "person" or "red car"',
                )
                preview_text_btn = gr.Button("Preview Video")
                clear_text_btn = gr.Button("Clear Query")

        with gr.Row(elem_id="my-btn"):
            save_text = gr.Textbox(
                label="Save Label",
                placeholder="Optional for text mode; defaults to the text query",
            )
            save_btn = gr.Button("Save Masks")

        save_output = gr.Textbox(label="Saved Mask Folder", interactive=False)

        demo.load(
            fn=load_initial_video,
            inputs=[video_state],
            outputs=[video_input, current_video_title, image_output, save_output],
        )
        prev_video_btn.click(
            fn=lambda video_state: step_and_load_video(-1, video_state),
            inputs=[video_state],
            outputs=[video_input, current_video_title, image_output, save_output],
        )
        next_video_btn.click(
            fn=lambda video_state: step_and_load_video(1, video_state),
            inputs=[video_state],
            outputs=[video_input, current_video_title, image_output, save_output],
        )
        prompt_mode.change(
            switch_prompt_mode,
            inputs=[prompt_mode, video_state],
            outputs=[click_controls, text_controls, image_output, video_output, save_output],
        )
        extract_btn.click(
            extract_first_frame,
            inputs=[video_state],
            outputs=[image_output, video_output, save_output],
        )
        image_output.select(
            fn=segment_frame,
            inputs=[prompt_mode, click_label, video_state],
            outputs=image_output,
        )
        clear_click_btn.click(
            clear_clicks,
            inputs=[video_state],
            outputs=[image_output, video_output, save_output],
        )
        preview_text_btn.click(
            preview_text_query,
            inputs=[query_text, video_state],
            outputs=[image_output, video_output, save_output],
        )
        clear_text_btn.click(
            clear_text_query,
            inputs=[video_state],
            outputs=[image_output, video_output, save_output],
        )
        save_btn.click(
            save_masks,
            inputs=[save_text, prompt_mode, video_state],
            outputs=save_output,
        )

demo.launch(
    server_name="0.0.0.0",
    server_port=8000,
    share=True,
    allowed_paths=[str(INPUT_CLIPS_DIR), str(GRADIO_TEMP_DIR)],
)
