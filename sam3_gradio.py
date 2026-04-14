import os
import sys
import inspect
import time
import random
from pathlib import Path
from contextlib import nullcontext

APP_ROOT = Path(__file__).resolve().parent
GRADIO_TEMP_DIR = APP_ROOT / "gradio_temp"
GRADIO_TEMP_DIR.mkdir(parents=True, exist_ok=True)
os.environ["GRADIO_TEMP_DIR"] = str(GRADIO_TEMP_DIR)
os.environ["TMPDIR"] = str(GRADIO_TEMP_DIR)
os.environ["TEMP"] = str(GRADIO_TEMP_DIR)
os.environ["TMP"] = str(GRADIO_TEMP_DIR)

import gradio as gr
import cv2
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

from sam3.model_builder import build_sam3_video_model

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

W = 512 #768
H = W
device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cuda" and torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

MASK_SAVE_ROOT = APP_ROOT / "masks"
INPUT_CLIPS_DIR = APP_ROOT / "sample_videos"
VIDEO_EXAMPLES = [[str(path)] for path in sorted(INPUT_CLIPS_DIR.glob("*")) if path.is_file()]

_CLICK_PREDICTOR = None
_TEXT_PREDICTOR = None


def autocast_context():
    if device == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def make_video_component(**kwargs):
    video_signature = inspect.signature(gr.Video)
    if "source" in video_signature.parameters:
        kwargs["source"] = "upload"
    elif "sources" in video_signature.parameters:
        kwargs["sources"] = "upload"
    return gr.Video(**kwargs)


def get_click_predictor():
    global _CLICK_PREDICTOR
    if _CLICK_PREDICTOR is None:
        sam3_model = build_sam3_video_model(device=device)
        predictor = sam3_model.tracker
        predictor.backbone = sam3_model.detector.backbone
        _CLICK_PREDICTOR = predictor
    return _CLICK_PREDICTOR


def get_text_predictor():
    global _TEXT_PREDICTOR
    if _TEXT_PREDICTOR is None:
        _TEXT_PREDICTOR = build_sam3_video_model(device=device)
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


def normalize_video_path(video_input):
    if video_input is None:
        return None

    if isinstance(video_input, dict):
        for key in ("path", "name", "video"):
            value = video_input.get(key)
            if isinstance(value, dict):
                nested = value.get("path") or value.get("name")
                if nested:
                    return str(nested)
            elif value:
                return str(value)
        return None

    if isinstance(video_input, (tuple, list)):
        return normalize_video_path(video_input[0]) if video_input else None

    if isinstance(video_input, Path):
        return str(video_input)

    return str(video_input)


def current_preview_image(video_state):
    if video_state["origin_images"]:
        return Image.fromarray(video_state["origin_images"][0])
    return None


def clear_mode_state(video_state):
    video_state["masks"] = None
    video_state["click_inference_state"] = None
    video_state["text_inference_state"] = None
    video_state["input_points"] = []
    video_state["scaled_points"] = []
    video_state["input_labels"] = []
    video_state["query_text"] = None
    video_state["frame_idx"] = 0


def read_display_frames(video_path, n_frames=1):
    vr = VideoReader(video_path, ctx=cpu(0))
    frame_count = min(len(vr), n_frames)
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


def load_video(video_input, video_state):
    video_path = normalize_video_path(video_input)
    if not video_path:
        gr.Warning("Please upload a video before annotating.")
        return None, None, ""
    if not Path(video_path).exists():
        gr.Warning(f"Uploaded video was not found on disk: {video_path}")
        return None, None, ""

    first_frame = read_display_frames(video_path, n_frames=1)[0]
    clear_mode_state(video_state)
    video_state["origin_images"] = [first_frame]
    video_state["video_path"] = video_path

    with torch.inference_mode(), autocast_context():
        video_state["click_inference_state"] = get_click_predictor().init_state(video_path=video_path)

    return Image.fromarray(first_frame), None, ""


def switch_prompt_mode(prompt_mode, video_state):
    video_state["prompt_mode"] = prompt_mode
    clear_mode_state(video_state)
    if prompt_mode == "Click" and video_state["video_path"] is not None:
        with torch.inference_mode(), autocast_context():
            video_state["click_inference_state"] = get_click_predictor().init_state(
                video_path=video_state["video_path"]
            )
    preview = current_preview_image(video_state)
    return (
        gr.update(visible=prompt_mode == "Click"),
        gr.update(visible=prompt_mode == "Text"),
        preview,
        None,
        "",
    )


def extract_first_frame(video_input, video_state):
    video_path = normalize_video_path(video_input) or video_state.get("video_path")
    if not video_path:
        gr.Warning("Please upload a video before extracting the first frame.")
        return None, None, ""
    if not Path(video_path).exists():
        gr.Warning(f"Uploaded video was not found on disk: {video_path}")
        return None, None, ""

    first_frame = read_display_frames(video_path, n_frames=1)[0]
    clear_mode_state(video_state)
    video_state["origin_images"] = [first_frame]
    video_state["video_path"] = video_path

    with torch.inference_mode(), autocast_context():
        video_state["click_inference_state"] = get_click_predictor().init_state(video_path=video_path)

    return Image.fromarray(first_frame), None, ""


def segment_frame(evt: gr.SelectData, click_label, prompt_mode, video_state):
    if prompt_mode != "Click":
        gr.Warning('Switch to "Click" mode to annotate with points.')
        return None
    if video_state["origin_images"] is None or video_state["click_inference_state"] is None:
        gr.Warning('Please click "Extract First Frame" first, then annotate the image.')
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
    return Image.fromarray(painted_image)


def clear_clicks(video_state):
    video_state["input_points"] = []
    video_state["input_labels"] = []
    video_state["scaled_points"] = []
    video_state["masks"] = None
    video_state["click_inference_state"] = None

    if video_state["video_path"] is not None:
        with torch.inference_mode(), autocast_context():
            video_state["click_inference_state"] = get_click_predictor().init_state(
                video_path=video_state["video_path"]
            )

    return current_preview_image(video_state), None, ""


def preview_text_query(video_input, query_text, video_state):
    video_path = normalize_video_path(video_input) or video_state.get("video_path")
    if not video_path:
        gr.Warning("Please upload a video before running a text query.")
        return None, None, ""
    if not Path(video_path).exists():
        gr.Warning(f"Uploaded video was not found on disk: {video_path}")
        return None, None, ""

    prompt = (query_text or "").strip()
    if not prompt:
        gr.Warning("Please enter a text query first.")
        return None, None, ""

    first_frame = read_display_frames(video_path, n_frames=1)[0]
    clear_mode_state(video_state)
    video_state["origin_images"] = [first_frame]
    video_state["video_path"] = video_path
    video_state["query_text"] = prompt

    with torch.inference_mode(), autocast_context():
        inference_state = get_text_predictor().init_state(resource_path=video_path)
        _, outputs = get_text_predictor().add_prompt(
            inference_state=inference_state,
            frame_idx=0,
            text_str=prompt,
        )

    mask = combine_output_masks(outputs)
    video_state["text_inference_state"] = inference_state
    if mask is None:
        gr.Warning(f'No mask was found on frame 0 for query "{prompt}".')
        return Image.fromarray(first_frame), None, ""

    resized_mask = cv2.resize(mask, (first_frame.shape[1], first_frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    video_state["masks"] = [resized_mask[:, :, None]]
    painted_image = render_mask_overlay(first_frame, resized_mask)
    return Image.fromarray(painted_image), None, ""


def clear_text_query(video_state):
    video_state["text_inference_state"] = None
    video_state["query_text"] = None
    video_state["masks"] = None
    return current_preview_image(video_state), None, ""


def track_click_video(n_frames, video_state):
    if video_state["click_inference_state"] is None or video_state["masks"] is None:
        gr.Warning("Please complete click-based segmentation on the first frame first.")
        return None

    vr = VideoReader(video_state["video_path"], ctx=cpu(0))
    effective_n_frames = min(len(vr), int(n_frames))
    images = [decord_frame_to_numpy(vr[i]) for i in range(effective_n_frames)]
    del vr

    display_images = [resize_frame(img)[0] for img in images]
    video_state["origin_images"] = display_images

    output_frames = []
    mask_frames = []

    with torch.inference_mode(), autocast_context():
        for out_frame_idx, _, _, video_res_masks, _ in get_click_predictor().propagate_in_video(
            inference_state=video_state["click_inference_state"],
            start_frame_idx=0,
            max_frame_num_to_track=max(effective_n_frames - 1, 0),
            reverse=False,
            propagate_preflight=True,
        ):
            if out_frame_idx >= len(display_images):
                break

            frame = display_images[out_frame_idx]
            mask = np.zeros(frame.shape[:2], dtype=np.float32)
            for mask_logits in video_res_masks:
                out_mask = (mask_logits.squeeze().detach().cpu().numpy() > 0).astype(np.float32)
                out_mask = cv2.resize(out_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                mask += out_mask

            mask = np.clip(mask, 0, 1)
            mask_frames.append(mask[:, :, None])
            output_frames.append(render_mask_overlay(frame, mask))

    if not output_frames:
        gr.Warning("Tracking did not return any frames.")
        return None

    video_state["masks"] = mask_frames
    video_file = GRADIO_TEMP_DIR / f"{time.time()}-{random.random()}-tracked_output.mp4"
    clip = ImageSequenceClip(output_frames, fps=15)
    clip.write_videofile(str(video_file), codec="libx264", audio=False, verbose=False, logger=None)
    return str(video_file)


def track_text_video(n_frames, video_state):
    if video_state["text_inference_state"] is None or not video_state["query_text"]:
        gr.Warning("Please run a text query first.")
        return None

    vr = VideoReader(video_state["video_path"], ctx=cpu(0))
    effective_n_frames = min(len(vr), int(n_frames))
    images = [decord_frame_to_numpy(vr[i]) for i in range(effective_n_frames)]
    del vr

    display_images = [resize_frame(img)[0] for img in images]
    video_state["origin_images"] = display_images

    output_frames = []
    mask_frames = []

    with torch.inference_mode(), autocast_context():
        for out_frame_idx, outputs in get_text_predictor().propagate_in_video(
            inference_state=video_state["text_inference_state"],
            start_frame_idx=0,
            max_frame_num_to_track=max(effective_n_frames - 1, 0),
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
        gr.Warning("Tracking did not return any frames.")
        return None

    video_state["masks"] = mask_frames
    video_file = GRADIO_TEMP_DIR / f"{time.time()}-{random.random()}-tracked_output.mp4"
    clip = ImageSequenceClip(output_frames, fps=15)
    clip.write_videofile(str(video_file), codec="libx264", audio=False, verbose=False, logger=None)
    return str(video_file)


def track_video(prompt_mode, n_frames, video_state):
    if prompt_mode == "Text":
        return track_text_video(n_frames, video_state)
    return track_click_video(n_frames, video_state)


def save_masks(save_text, video_state):
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

    masks = np.asarray(video_state["masks"], dtype=np.float32)
    prompt_dirname = "_".join(prompt.split())
    save_dir = get_unique_prompt_dir(video_state["video_path"], prompt_dirname)
    save_dir.mkdir(parents=True, exist_ok=True)

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
            "origin_images": None,
            "masks": None,
            "video_path": None,
            "click_inference_state": None,
            "text_inference_state": None,
            "input_points": [],
            "scaled_points": [],
            "input_labels": [],
            "query_text": None,
            "frame_idx": 0,
            "obj_id": 1,
        }
    )

    with gr.Column():
        video_input = make_video_component(
            label="Upload Video",
            format="mp4",
            interactive=True,
            elem_id="my-video1",
        )

        gr.Examples(
            examples=VIDEO_EXAMPLES,
            inputs=[video_input],
            label=f"Choose a video from {INPUT_CLIPS_DIR}",
            elem_id="my-btn2",
        )

        prompt_mode = gr.Radio(["Click", "Text"], value="Click", label="Prompt Mode")
        image_output = gr.Image(label="Segmentation Preview", interactive=True, elem_id="my-video")

        # demo.css = """
        # #my-btn {
        #    width: 48% !important;
        #    max-width: 720px !important;
        #    margin: 0 auto;
        # }

        # #my-video1 {
        #    width: 48% !important;
        #    max-width: 720px !important;
        #    height: auto !important;
        #    margin: 0 auto;
        # }

        # #my-video {
        #    width: 48% !important;
        #    max-width: 720px !important;
        #    height: auto !important;
        #    margin: 0 auto;
        # }

        # #my-btn2 {
        #     width: 48% !important;
        #     max-width: 720px !important;
        #     margin: 0 auto;
        # }

        # #my-btn2 button {
        #     width: 120px !important;
        #     max-width: 120px !important;
        #     min-width: 120px !important;
        #     height: 70px !important;
        #     max-height: 70px !important;
        #     min-height: 70px !important;
        #     margin: 8px !important;
        #     border-radius: 8px !important;
        #     overflow: hidden !important;
        #     white-space: normal !important;
        # }
        # """
        demo.css = """
        #my-btn {
            width: 36% !important;
            max-width: 520px !important;
            margin: 0 auto;
        }

        #my-video1 {
            width: 36% !important;
            max-width: 520px !important;
            height: auto !important;
            margin: 0 auto;
        }

        #my-video {
            width: 36% !important;
            max-width: 520px !important;
            height: auto !important;
            margin: 0 auto;
        }

        #my-btn2 {
            width: 36% !important;
            max-width: 520px !important;
            margin: 0 auto;
        }
        """

        with gr.Group(visible=True) as click_controls:
            with gr.Row(elem_id="my-btn"):
                extract_btn = gr.Button("Extract First Frame")
                click_label = gr.Radio(["Positive", "Negative"], label="Click Type", value="Positive")
                clear_click_btn = gr.Button("Clear All Clicks")

        with gr.Group(visible=False) as text_controls:
            with gr.Row(elem_id="my-btn"):
                query_text = gr.Textbox(
                    label="Text Query",
                    placeholder='Enter a text query, for example "person" or "red car"',
                )
                preview_text_btn = gr.Button("Preview Query")
                clear_text_btn = gr.Button("Clear Query")

        with gr.Row(elem_id="my-btn"):
            n_frames_slider = gr.Slider(minimum=1, maximum=201, value=81, step=1, label="Tracking Frames N")
            track_btn = gr.Button("Track")

        with gr.Row(elem_id="my-btn"):
            save_text = gr.Textbox(label="Save Label", placeholder="Optional for text mode; defaults to the text query")
            save_btn = gr.Button("Save Masks")

        video_output = gr.Video(label="Tracking Result", elem_id="my-video")
        save_output = gr.Textbox(label="Saved Mask Folder", interactive=False)

        video_input.change(load_video, inputs=[video_input, video_state], outputs=[image_output, video_output, save_output])
        prompt_mode.change(
            switch_prompt_mode,
            inputs=[prompt_mode, video_state],
            outputs=[click_controls, text_controls, image_output, video_output, save_output],
        )
        extract_btn.click(
            extract_first_frame,
            inputs=[video_input, video_state],
            outputs=[image_output, video_output, save_output],
        )
        image_output.select(
            fn=segment_frame,
            inputs=[click_label, prompt_mode, video_state],
            outputs=image_output,
        )
        clear_click_btn.click(
            clear_clicks,
            inputs=video_state,
            outputs=[image_output, video_output, save_output],
        )
        preview_text_btn.click(
            preview_text_query,
            inputs=[video_input, query_text, video_state],
            outputs=[image_output, video_output, save_output],
        )
        clear_text_btn.click(
            clear_text_query,
            inputs=video_state,
            outputs=[image_output, video_output, save_output],
        )
        track_btn.click(track_video, inputs=[prompt_mode, n_frames_slider, video_state], outputs=video_output)
        save_btn.click(save_masks, inputs=[save_text, video_state], outputs=save_output)

demo.launch(server_name="0.0.0.0", server_port=8000, share=True)
