## 🎯 Overview

A Python library for launching a Gradio interface to create video masks using either click-based interaction or text prompts.

## 📦 Installation

```bash
git clone https://github.com/sakshamsingh1/sam3_mask_annotation_tool.git
cd sam3_mask_annotation_tool
git clone https://github.com/facebookresearch/sam3.git
```

```bash
conda create -n sam3_gradio python=3.10 -y
conda activate sam3_gradio
pip install -r requirements.txt
```

<details style="margin-top: -5px;">
  <summary><strong>More setup details</strong></summary>

  <br>

  <p><strong>Tested environment</strong></p>

  <ul>
    <li>CUDA 12.4</li>
    <li>NVIDIA A6000 Ada GPU</li>
    <li>Python <code>3.10.20</code></li>
    <li>PyTorch <code>2.6.0+cu124</code></li>
    <li>torchvision <code>0.21.0+cu124</code></li>
    <li>Gradio <code>6.11.0</code></li>
  </ul>

  <p>Library versions may need to be adjusted depending on your environment and system setup.</p>

<p><strong>Note:</strong> This codebase requires <code>sam3</code>. Please refer to the official SAM3 <a href="https://github.com/facebookresearch/sam3?tab=readme-ov-file#getting-started">Getting Started</a> guide for additional installation and setup details.</p>

</details>


## Run the app

#### Upload video UI

<p align="center">
  <img src="./assets/gradio_demo.gif" alt="Gradio demo" width="90%" />
</p>

You can upload a video, annotate it using **text prompts** or **click-based interaction**, and save the generated masks.

```bash
python sam3_gradio.py
```

#### Annotating a directory of videos

```bash
python sam3_gradio_dir.py
```

This mode is useful for annotating and generating masks for a large collection of videos. Set `INPUT_CLIPS_DIR` to the directory containing your input videos, and `MASK_SAVE_ROOT` to the directory where you want the masks to be saved.

> **Note:** Click-based annotation currently applies only to the **first frame** of each video. After annotation, you can automatically propagate the saved masks to the remaining frames using: `python propagate_saved_masks.py`

## Acknowledgement

This codebase builds on [SAM3](https://github.com/facebookresearch/sam3) and [MiniMax-Remover](https://github.com/zibojia/MiniMax-Remover). 
We thank the original authors for open-sourcing their code.

## 📧 Contact

If you have any questions, suggestions, or run into issues, please open an issue or contact [sxk230060@utdallas.edu](mailto:sxk230060@utdallas.edu).