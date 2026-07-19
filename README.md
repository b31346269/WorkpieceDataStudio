# Workpiece Data Studio

Workpiece Data Studio 是一套獨立的工件影像生成與 YOLO 資料審查工具，使用
FastAPI 與瀏覽器介面操作。

固定類別：

- `0 = hole`
- `1 = screw`
- `2 = tool`

## 主要功能

- 上傳工件參考照片。
- 使用 Stable Diffusion 1.5、SDXL 或 FLUX.2 生成不同工件影像。
- 提供「嚴格保留結構」與「機械形狀變體」等模式。
- 使用自己的 Ultralytics `best.pt` 自動建立 YOLO 草稿框。
- 在瀏覽器中新增、移動、縮放、改類別或刪除標註框。
- 人工接受或淘汰每張候選圖片。
- 匯出可上傳至 Roboflow 的 YOLOv8 ZIP。

`best.pt` 只負責預標註，不參與影像生成。所有生成圖片都必須經過人工檢查，
不能直接視為正確標註。

## 本機啟動

請在專案目錄使用專案自己的虛擬環境：

```powershell
.venv\Scripts\python.exe -m uvicorn workpiece_studio.main:app --host 127.0.0.1 --port 7865
```

開啟：

<http://127.0.0.1:7865>

若需要重新安裝環境：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup.ps1 -WithML
```

## 學校伺服器

學校伺服器可使用較大的模型：

- SDXL + IP-Adapter + Canny ControlNet
- FLUX.2 Klein 9B BF16 + CPU offload
- `best.pt` YOLO 自動預標註

伺服器只允許使用實體 GPU `2`、`3` 或 `6`。完整流程請參考
[`school_server/README.md`](school_server/README.md)。

FLUX.2 Klein 9B 使用 FLUX Non-Commercial License，只能依其授權條款進行
非商業研究與開發。

## 資料處理流程

1. 建立專案並加入參考工件照片。
2. 選擇模型與機械保真模式。
3. 生成候選圖片並使用 `best.pt` 預標。
4. 人工修正框選結果並接受或淘汰圖片。
5. 只匯出通過審查的圖片為 YOLOv8 ZIP。
6. 將 ZIP 上傳至 Roboflow，或與真實訓練資料合併。

建議保留真實的 validation/test 資料，生成圖片只加入 training split。
