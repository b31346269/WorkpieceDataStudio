# School server workflow

This folder keeps SSH credentials out of the web application. The FastAPI
process runs on the school GPU server and listens only on remote
`127.0.0.1:7865`. An SSH tunnel exposes it to the local browser.

## 1. Deploy to a new remote folder

Run from Windows PowerShell:

```powershell
.\school_server\deploy-school-server.ps1 -HostAlias sslab-school -Bootstrap
```

The default remote folder is `~/workpiece_data_studio`. The deployment excludes
the Windows virtual environment, local Hugging Face cache, and large dataset
ZIPs. Reference images and `best.pt` are included. Dataset ZIPs can be uploaded
through the remote UI later or reused from an existing server training folder.

## 2. Open the remote UI locally

```powershell
.\school_server\connect-school-server.ps1 -HostAlias sslab-school
```

Keep that terminal open and browse to <http://127.0.0.1:7866>. Uploads,
generation, review, and exports are then executed on the school server.
The remote UI defaults to physical GPU `2`; pass `-RemoteGpu 3` or
`-RemoteGpu 6` to use another authorized card. Other ids are rejected.

Select **School GPU: SDXL + IP-Adapter + ControlNet** in the generation page.
This uses the larger SDXL base at 1024×1024. Canny ControlNet preserves
mechanical edges while `best.pt` remains the YOLO draft-label model.

Select **School GPU: FLUX.2 Klein 9B BF16** for more substantial changes to the
whole workpiece silhouette and layout. It uses native image editing, four
distilled steps, and CPU offload so it can run on the authorized RTX A5000.
The model is restricted to non-commercial use, and every output must pass the
existing manual review before export.

### How to tell local mode from school-server mode

Running `run.ps1` starts the app on the Windows computer only. It cannot use the
models or GPUs under `/home/ping` and may report that the school service is not
connected. To run the school copy, first make sure the school VPN is connected,
then run this command from the repository root in PowerShell:

```powershell
.\school_server\connect-school-server.ps1 -HostAlias sslab-school -RemoteGpu 2
```

Keep that PowerShell window open and use only <http://127.0.0.1:7866>. The
script starts the remote service on `127.0.0.1:7865` and creates the SSH tunnel
from local port `7866`. Closing the terminal closes the tunnel, although the
background remote process can remain alive.

If it fails, check each layer separately:

```powershell
ssh sslab-school "hostname"
ssh sslab-school "cd ~/workpiece_data_studio && cat workspace/school-ui.pid && tail -n 30 workspace/school-ui.log"
Test-NetConnection 127.0.0.1 -Port 7866
```

If the first command fails, reconnect the VPN or repair the `sslab-school` SSH
host entry. If the second fails, deploy/bootstrap the school copy. If only the
third fails, restart `connect-school-server.ps1` and leave its window open.

## 3. Train a larger detector

After approving candidates and exporting a synthetic ZIP on the remote UI, run:

```bash
cd ~/workpiece_data_studio
.school-env/bin/python school_server/train_yolo.py \
  --source datasets/NEW-workpiece.yolov8.zip \
  --aux-voc datasets/Tool-2.voc.zip \
  --synthetic workspace/projects/PROJECT_ID/exports/GENERATED.yolov8.zip \
  --synthetic-max-fraction 0.25 \
  --model yolo26l.pt \
  --epochs 150 \
  --imgsz 960 \
  --export-imgsz 640 \
  --device 2
```

The first `--source` is the target-domain NEW workpiece YOLOv8 ZIP. Its real
validation/test splits stay locked. `--aux-voc` converts Tool 2 from Pascal VOC
and adds only its train split. Roboflow variants whose source id appears in the
locked holdout are excluded. Approved synthetic images are train-only and are
capped at 25% of combined real train images by default.

The script exports a fixed-size, traditional YOLO ONNX with `end2end=False` so
the existing Unity `(1, 4 + classes, predictions)` decoder can read it. Its
default is the larger `yolo26l.pt`; use `yolo26m.pt` if server GPU memory is
limited. The helper refuses GPU ids outside `6,8`; use `--device 6` or `8` for
one allocated card. Do not start a multi-GPU run unless both cards are
explicitly assigned to the same job.

For a controlled experiment, use the same model, seed, epochs and image size:

1. A: NEW workpiece only (`--source`).
2. B: A + Tool 2 (`--aux-voc`).
3. C: B + approved factory synthetic (`--synthetic --synthetic-max-fraction 0.25`).
4. D: B + more synthetic (`--synthetic-max-fraction 0.40`) as a ratio ablation.

Do not use the reducer PDF images, their JSON annotations, or any image from the
locked validation/test split as WorkpieceDataStudio references.
