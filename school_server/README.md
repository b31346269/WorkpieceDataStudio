# School server workflow

Start with the [student handoff guide](../HANDOFF_START_HERE.md). It covers
the Release download, SSH configuration, server bootstrap, remote UI tunnel,
a one-epoch training check, and the historical M0/C/D/E experiment.

Use a single physical GPU assigned by the lab (6 or 8). The current training
entry point is `launch_handoff_yolo11n.sh`. Historical launchers have their
own group definitions. Follow the handoff guide for dataset and checkpoint
selection, output locations, and evaluation on the fixed NEW workpiece test set.

## Workpiece + wrench segmentation pilot

The current detection ZIPs are not segmentation datasets: their labels contain
five values (`class cx cy width height`). For the segmentation pilot, create a
separate two-class polygon dataset with `workpiece` and `wrench`. The project
has two different detectors whose class name is `tool`: the main workpiece
detector's `tool` means the workpiece, while `yolo11n_tool` is the separate
wrench detector. Do not merge those meanings.

See [`SEGMENTATION_ROI_PLAN.md`](../SEGMENTATION_ROI_PLAN.md) for the complete
specification. Use `prelabel_segmentation_candidates.py` to create candidate
boxes for manual polygon correction, then run `segmentation_dataset.py` before
training. The validator intentionally refuses rectangle-only labels.

Segmentation does not need another manually drawn fixed ROI. Run the model on a
full frame (or a wide safe fallback ROI), convert each accepted mask to its own
padded/clamped Dynamic ROI, and pass the workpiece ROI to the hole/screw model
and the wrench ROI to the wrench/pose path.
