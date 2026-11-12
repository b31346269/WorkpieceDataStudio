# Workpiece Data Studio

## 第一次接手

1. 到 [GitHub Releases](https://github.com/b31346269/WorkpieceDataStudio/releases) 的 **Assets** 下載 `WorkpieceDataStudio-handoff-20261112.tar.gz` 和同名的 `.sha256`。交接包含程式、歷史資料與權重。GitHub 自動產生的 **Source code** 壓縮檔只有原始碼。
2. 打開 [學弟妹交接手冊](HANDOFF_START_HERE.md)，向實驗室管理員取得個人 SSH 帳號、校網／VPN 入口及 GPU 分配。
3. 照手冊登入伺服器，先完成一輪 `SMOKE` 並匯出 ONNX，再操作網頁或重跑歷史 M0／C／D／E 實驗。

Workpiece Data Studio 是一套獨立的工件影像生成與 YOLO 資料審查工具，使用
FastAPI 與瀏覽器介面操作。

固定類別：

- `0 = hole`
- `1 = screw`
- `2 = tool`

## 主要功能

- 上傳工件參考照片。
- 使用 Stable Diffusion 1.5、SDXL 或 FLUX.2 生成不同工件影像。
- 可選混合工廠環境，依 seed 輪替產線、CNC 機台、維修桌、輸送治具與倉儲品管站。
- 提供「嚴格保留結構」與「機械形狀變體」等模式。
- 使用自己的 Ultralytics `best.pt` 自動建立 YOLO 草稿框。
- 在瀏覽器中新增、移動、縮放、改類別或刪除標註框。
- 人工接受或淘汰每張候選圖片。
- 匯出可上傳至 Roboflow 的 YOLOv8 ZIP。

`best.pt` 只負責預標註，不參與影像生成。所有生成圖片都必須經過人工檢查，
不能直接視為正確標註。

## 本機啟動

若只想在 Windows 練習介面，先從 GitHub 的 **Code → Download ZIP** 下載並解壓原始碼，或執行 `git clone https://github.com/b31346269/WorkpieceDataStudio.git`。安裝 Python 3.12 後，在專案目錄的 PowerShell 執行以下指令。

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup.ps1
.\run.ps1
```

開啟：

<http://127.0.0.1:7865>

要使用本機生成與預標註，安裝 ML 依賴後重新啟動。

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup.ps1 -WithML
```

## 學校伺服器

學校伺服器可使用較大的模型：

- SDXL + IP-Adapter + Canny ControlNet
- FLUX.2 Klein 9B BF16 + CPU offload
- `best.pt` YOLO 自動預標註

交接腳本支援實體 GPU `6` 或 `8`。請使用管理員分配給你的卡。連線、SSH tunnel、訓練與評估請依 [交接手冊](HANDOFF_START_HERE.md)。

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
