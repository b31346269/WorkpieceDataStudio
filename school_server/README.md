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

## 3. Train a larger detector

After approving candidates and exporting a synthetic ZIP on the remote UI, run:

```bash
cd ~/workpiece_data_studio
.school-env/bin/python school_server/train_yolo.py \
  --source workspace/projects/PROJECT_ID/imports/source-yolov8.zip \
  --synthetic workspace/projects/PROJECT_ID/exports/GENERATED.yolov8.zip \
  --model yolo26l.pt \
  --epochs 150 \
  --imgsz 960 \
  --device 2,3,6
```

The script preserves the real validation/test splits and adds synthetic images
only to `train`. Its default is the larger `yolo26l.pt`; use `yolo26m.pt` if
server GPU memory is limited, or `yolo26x.pt` only when the server has enough
memory. The helper refuses GPU ids outside `2,3,6`.
