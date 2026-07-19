# Workpiece Data Studio

Standalone local tool for generating and reviewing realistic workpiece training
images. It does not run inside Unity and does not modify the LIVEs runtime.

## What the MVP does

- Creates an isolated local project with classes `hole`, `screw`, and `tool`.
- Imports a Roboflow YOLOv8 ZIP safely for reference and class validation.
- Accepts one or more workpiece reference photographs.
- Generates 640×640 candidates with SD 1.5 image-to-image + IP-Adapter.
- On a 20+ GB school GPU, generates 1024×1024 candidates with SDXL +
  IP-Adapter + Canny ControlNet.
- On the school server, uses FLUX.2 Klein 9B BF16 with CPU offload for larger,
  instruction-driven workpiece redesigns.
- Can run a custom Ultralytics YOLO `.pt` model to create draft boxes.
- Provides a browser canvas to add, move, resize, relabel, and delete boxes.
- Tracks `pending`, `approved`, and `rejected` review states.
- Exports approved candidates as a Roboflow-compatible YOLOv8 ZIP.

Generated images are candidates, not automatic ground truth. New mechanical
structures, small holes, and screw heads must be reviewed before training.

## Install

The delivered copy on this computer is already prepared with the CUDA
environment, SD 1.5/IP-Adapter cache, the supplied YOLOv8 ZIP, `best.pt`, and
four reference images. Start it directly with `run.ps1`.

Use the setup commands below only when recreating or repairing the environment.

Open PowerShell in this folder:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup.ps1
```

Core mode installs quickly and includes a clearly marked mock generator for
testing the complete workflow without downloading large models.

Install the real local AI stack when ready:

```powershell
.\setup.ps1 -WithML
```

The first real generation downloads the selected Hugging Face base model and
IP-Adapter into this project's ignored `workspace/.huggingface` cache. Several
GB of free disk space and internet access are required.

## Run

```powershell
.\run.ps1
```

Then open <http://127.0.0.1:7865>.

The included project is named `workpiece-generation`. You can add more
reference photographs from the first tab without changing the original files.

## RTX 4060 Laptop 8 GB defaults

- Base: `stable-diffusion-v1-5/stable-diffusion-v1-5`
- Adapter: `h94/IP-Adapter`, `models/ip-adapter_sd15.bin`
- Output: 640×640
- FP16 and model CPU offload enabled
- One image at a time

Use the default strict mechanical mode (`strength 0.34`, IP-Adapter `0.82`).
It clamps structural variation and always keeps the built-in anti-deformation
negative prompt. Higher strength gives more variety but increases impossible
geometry.

Use **mechanical shape variation** when a different silhouette is required. It
raises img2img strength, weakens and ends Canny ControlNet earlier, and asks for
new proportions, mounting ears, ribs, cooling fins, and cavity placement while
still requiring circular holes and correctly seated fasteners.

For a larger semantic redesign, select **FLUX.2 Klein 9B BF16** in the school
server UI. FLUX uses native reference-image editing instead of the strength and
IP-Adapter sliders, and its distilled pipeline runs four inference steps. The
9B model is governed by the FLUX Non-Commercial License. This application's
manual review stage is mandatory for every generated candidate.

## School server

The school server workflow is documented in
`school_server/README.md`. The application runs on the remote GPU and binds to
remote localhost only; an SSH tunnel makes the same UI available at local
`http://127.0.0.1:7866`. SSH credentials are never stored in the web app.

## Output

Each project is stored under `workspace/projects/<project-id>/`. Exports contain:

```text
train/images/
train/labels/
data.yaml
generation_manifest.json
```

This generated-only ZIP can be uploaded to Roboflow or merged with an existing
YOLOv8 training set. Keep real validation and test sets separate.
