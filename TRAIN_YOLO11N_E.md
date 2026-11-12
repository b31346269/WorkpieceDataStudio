# Roboflow 資料與 YOLO11n E 訓練流程

Roboflow 提供兩份已標註的真實影像資料。Workpiece Data Studio 負責生成、預標註、人工審核及匯出合成圖。訓練在學校 Linux GPU 伺服器上使用 Ultralytics 執行。先以 Tool 2 訓練 M0，再從 M0 的 `best.pt` 出發，加入 NEW workpiece 和合成圖微調出 E。

## 1. 取得歷史資料

在 [GitHub 交接 Release](https://github.com/b31346269/WorkpieceDataStudio/releases/tag/handoff-20261112) 的 **Assets** 下載完整交接包與同名 `.sha256`。三份 ZIP 已放在交接包的 `training_inputs/`，原始 M0／C／D／E 權重放在 `YOLO11n_tool2_experiments_models.tar.gz`。

| ZIP | 原始圖片數（Train / Valid / Test） | 用途 |
|---|---:|---|
| `new-workpiece-clean-yolov11.zip` | 374 / 47 / 47 | Roboflow NEW workpiece v10。E 使用 Train，M0 和 E 共用這份 Valid 和 Test。 |
| `tool2-clean-yolov11.zip` | 6,560 / 820 / 820 | Roboflow Tool 2 v20。M0 和 E 只使用 Train。 |
| `factory-synthetic-reviewed.yolov8.zip` | 321 / 0 / 0 | Workpiece Data Studio 已人工審核的合成圖，加入 E 的 Train。 |

Roboflow 版本及授權標記來自兩份 ZIP 內的 `data.yaml`。訓練腳本會依資料來源用來源 ID 或影像雜湊過濾真實影像。實際入組數以每次訓練產生的 `dataset/preparation_report.json` 為準。

[Roboflow NEW workpiece v10](https://universe.roboflow.com/project-m3ui7/new-workpiece-for-data-augmentat/dataset/10) · [Roboflow Tool 2 v20](https://universe.roboflow.com/project-m3ui7/tool-2/dataset/20)

若要產生新合成批次，先在 Workpiece Data Studio 匯入可用的訓練來源參考圖，再生成候選圖。可用工件偵測 `.pt` 產生草稿框，逐張補框、修正類別並淘汰不合格的候選，最後以 `train` 匯出 YOLO ZIP。類別順序為 `0=hole`、`1=screw`、`2=tool`，其中 `tool` 是工件本體。參考和訓練來源都要排除 NEW 的 Valid、Test 和最終評估影片。重現歷史 E 時，使用交接包內的 321 張。

[實驗一生成設定](EXPERIMENT_1_GENERATION_SETTINGS.md) · [維修桌場景設定](EXPERIMENT_ENVIRONMENT_V2.md)

## 2. 歷史訓練設定

| 組別 | 初始化 | 訓練資料 | Optimizer 與 epochs |
|---|---|---|---|
| M0 | `yolo11n.pt` | Tool 2 | `auto`，上限 150，歷史權重的訓練紀錄到第 81 epoch |
| C | M0 `best.pt` | Tool 2 + 合成 | `AdamW`，上限 60 |
| D | M0 `best.pt` | Tool 2 + NEW | `AdamW`，上限 60 |
| E | M0 `best.pt` | Tool 2 + NEW + 合成 | `AdamW`，上限 60，歷史權重的訓練紀錄到第 60 epoch |

C、D、E 分別從同一份 M0 `best.pt` 開始，微調參數為 `lr0=0.001`、`lrf=0.01`。歷史 M0 和 E 權重保存的設定為 Ultralytics `8.4.100`、影像尺寸 640、batch 32、seed 42、patience 30，並使用實體 GPU `6,8`。線上擴增包含色彩、輕微旋轉、平移、縮放與透視，關閉 Mosaic、MixUp 和 Copy-Paste。

[歷史訓練腳本](school_server/launch_yolo11n_tool2_finetune.sh) · [資料合併、擴增與匯出程式](school_server/train_yolo.py)

## 3. 在學校伺服器重跑

先依[交接手冊](HANDOFF_START_HERE.md)完成 SSH 登入、解壓、環境安裝和 `SMOKE`。在遠端 Linux 的 `tmux` 執行下列指令。示例使用已分配的 GPU `6`，分配到 `8` 時替換 GPU 參數。若要比對歷史版本，先在已建立的環境執行 `.school-env/bin/python -m pip install ultralytics==8.4.100`。交接 launcher 每次使用單 GPU，與歷史雙 GPU 的結果可能有差異。

### 從頭訓練 M0，再微調 E

```bash
cd "$HOME/workpiece_handoff"
bash school_server/launch_handoff_yolo11n.sh M0 6 m0-e-repro
```

等 M0 跑完後，在同一目錄執行。

```bash
M0="$PWD/school_training/handoff/m0-e-repro/runs/M0/weights/best.pt"
test -f "$M0" && bash school_server/launch_handoff_yolo11n.sh E 6 e-repro "$M0"
```

### 沿用歷史 M0，只重跑 E

使用下列指令可直接重跑 E 的微調階段，成果存到另一個 run ID。

```bash
cd "$HOME/workpiece_handoff"
mkdir -p historical_models
tar -xzf YOLO11n_tool2_experiments_models.tar.gz -C historical_models
M0="$PWD/historical_models/YOLO11n_M0_Tool2_Base_best.pt"
bash school_server/launch_handoff_yolo11n.sh E 6 e-from-original-m0 "$M0"
```

若要按歷史腳本依序跑 M0、C、D、E，且管理員已同時分配 GPU `6` 與 `8`，請在乾淨的交接目錄中從 `tmux` 執行 `WORKPIECE_ROOT="$PWD" bash school_server/launch_yolo11n_tool2_finetune.sh`。成果位於 `school_training/tool2_finetune/`。

## 4. 取回權重與評估 E

交接 launcher 會在 `school_training/handoff/<run ID>/runs/E/weights/` 寫出 `best.pt` 和 `best.onnx`。前者可用於推論、預標註或後續微調，後者是 640 px 固定尺寸的部署模型。歷史權重在權重壓縮檔內，檔名為 `YOLO11n_E_Tool2_NEW_Synthetic_FT_best.pt` 和同名 `.onnx`。

重跑後用保留的 NEW Test 做最終評估。依實際使用的 run ID 和 GPU 調整以下指令。

```bash
cd "$HOME/workpiece_handoff"
E_RUN="$PWD/school_training/handoff/e-repro"
.school-env/bin/yolo detect val \
  model="$E_RUN/runs/E/weights/best.pt" \
  data="$E_RUN/dataset/data.yaml" split=test imgsz=640 device=6
```

歷史 E 的 `best.pt` 保存的 **Validation** 指標是 Precision **93.37%**、Recall **93.50%**、mAP50 **96.63%**、mAP50–95 **65.81%**。獨立 Test 分數需另跑。若要檢查加入合成圖的影響，讓 D 和 E 使用同一份 Test 和相同推論設定，再比較各類別的誤判、漏判與 mAP。
