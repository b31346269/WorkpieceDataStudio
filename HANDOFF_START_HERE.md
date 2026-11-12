# Workpiece Data Studio｜學弟妹交接手冊

第一次接手，依序完成：用自己的帳號登入學校伺服器 → 跑一輪 YOLO11n 測試訓練 → 開遠端網頁完成生成、審核、匯出 → 重跑歷史實驗。已附審核好的合成圖，所以可以先重跑模型，再學如何補資料。

## 1. 交接前確認

到 GitHub Releases，在 **Assets** 下載 `WorkpieceDataStudio-handoff-20261112.tar.gz` 和同名 `.sha256`，放在同一個資料夾。開 PowerShell 切換到下載資料夾，再執行下方指令。校網、SSH 和 GPU 資訊由實驗室管理員提供。  
[GitHub Releases](https://github.com/b31346269/WorkpieceDataStudio/releases)

| 由交接人或管理員填寫 | 資訊 |
|---|---|
| 連線前要接校內網路或 VPN | ＿＿＿＿ |
| SSH 主機、Port、主機金鑰指紋 | ＿＿＿＿ |
| 個人帳號申請窗口、GPU 預約方式 | ＿＿＿＿ |
| 允許使用的儲存路徑、容量、套件下載方式 | ＿＿＿＿ |
| 歷史訓練 log、環境版本、評估影片的共享路徑 | ＿＿＿＿ |

在 Windows PowerShell 執行 `ssh -V`，確認有 OpenSSH。用文字編輯器在 `%USERPROFILE%\.ssh\config` 設定下列內容，把尖括號欄位換成管理員提供的值。首次連線時先比對主機金鑰指紋。

```sshconfig
Host sslab-school
    HostName <學校提供的主機名或IP>
    User <自己的帳號>
    Port <學校提供的SSH埠>
```

```powershell
ssh sslab-school "hostname"
Get-FileHash .\WorkpieceDataStudio-handoff-20261112.tar.gz -Algorithm SHA256
Get-Content .\WorkpieceDataStudio-handoff-20261112.tar.gz.sha256
scp .\WorkpieceDataStudio-handoff-20261112.tar.gz sslab-school:~/
ssh sslab-school
```

兩個 SHA256 值應相同。下面改在登入後的 **Linux Bash** 執行。先用 `df -h "$HOME"` 確認空間。以下以 GPU 6 為例，已分配 GPU 8 的同學把指令中的 GPU 編號改成 8。先用 `nvidia-smi -i 6` 確認分配與占用情況。

```bash
mkdir "$HOME/workpiece_handoff" && tar -xzf "$HOME/WorkpieceDataStudio-handoff-20261112.tar.gz" -C "$HOME/workpiece_handoff"
cd "$HOME/workpiece_handoff"
sha256sum -c SHA256SUMS
command -v tmux
CUDA_VISIBLE_DEVICES=6 bash school_server/bootstrap.sh
.school-env/bin/python -m pip check
```

請逐行確認成功再往下。若目錄已存在，改用新的名稱，後續指令也要一起改。環境需要 Linux 上可用的 Conda、相容的 NVIDIA 驅動和套件下載來源。若 Conda 位於其他位置，用 `CONDA_BIN=/實際路徑/bin/conda CUDA_VISIBLE_DEVICES=6 bash school_server/bootstrap.sh`。Bootstrap 應顯示 `CUDA ready: True`。第一次使用會下載 `yolo11n.pt`，首次匯出 ONNX 可能需要下載 ONNX 相關套件。受限網路請向管理員取得快取或核准的套件來源。

## 2. 先完成一輪 GPU 訓練

在 **Linux Bash** 開 tmux。進入 tmux 後再執行訓練指令，每次使用新的 run ID。

```bash
cd "$HOME/workpiece_handoff"
tmux new -s workpiece-train
bash school_server/launch_handoff_yolo11n.sh SMOKE 6 smoke-01
```

這會用 NEW workpiece 跑 1 epoch 並匯出 ONNX。按 `Ctrl+b`、放開，再按 `d` 可離開畫面讓訓練繼續。SSH 重連後用 `tmux attach -t workpiece-train` 回去，或在另一個遠端終端看進度。

```bash
cd "$HOME/workpiece_handoff"
tail -n 40 school_training/handoff/smoke-01/train.log
ls -lh school_training/handoff/smoke-01/runs/SMOKE/weights/
```

確認 log 有完成的 epoch，且有 `best.pt`、`last.pt` 和 `best.onnx`。若已有 `best.pt` 但 ONNX 匯出失敗，可處理完依賴後，用 `.school-env/bin/yolo export model=實際權重路徑 format=onnx imgsz=640 dynamic=False simplify=True nms=False` 單獨補匯出。若出現 CUDA OOM，保留 log，確認同卡的生成服務和其他工作狀態，經分配後使用空閒的卡。若需改 batch，記錄變更並用一致的 batch 重跑比較組。

## 3. 操作遠端工具

在 **Linux Bash** 開另一個 tmux。以下選用遠端 17865；若已被占用，和後續 PowerShell 指令一起改。UI 綁定 server 的 loopback。

```bash
cd "$HOME/workpiece_handoff"
tmux new -s workpiece-ui
PHYSICAL_GPU=6 PORT=17865 bash school_server/start_ui.sh
```

在自己的 **Windows PowerShell** 開另一個視窗，執行並保持連線。用瀏覽器開 [遠端 Workpiece Data Studio](http://127.0.0.1:7866)。

```powershell
ssh -o ExitOnForwardFailure=yes -N -L 127.0.0.1:7866:127.0.0.1:17865 sslab-school
```

1. 建立新專案。在「資料與模型」加入一張確認可用的訓練來源參考圖，可從 `training_inputs/factory_references/` 開始。要看既有資料摘要，可匯入 NEW workpiece ZIP。資料匯入網頁和伺服器訓練是各自的操作。
2. 要預標註，先在本機解開歷史權重壓縮檔，選一個工件偵測 `.pt` 上傳，例如 E。要先練習介面，可選「流程測試」，這種預覽不能匯出。正式生成選學校 SDXL 或 FLUX.2 Klein，先產 1–3 張並記下模式、參考圖、seed 和 prompt。FLUX.2 Klein BF16 依其非商業授權使用，模型下載權限由實驗室提供。
3. 到「審核標註」逐張查看。工件本體為 `2=tool`，空孔為 `0=hole`，已鎖入的螺絲頭為 `1=screw`。補漏框、改錯類別、刪重複框。工件變形、遮擋嚴重、螺絲幾何錯誤或標註有疑義的候選先淘汰。完成後儲存標註並接受可用圖片。
4. 到「匯出」選 `train` 並下載 YOLOv8 ZIP。從 ZIP 確認 `train/images`、`train/labels`，並檢查匯出所附的生成紀錄。合成 ZIP 的 YAML 可能把 train、val、test 都指到同一資料夾。交給訓練器的 `--synthetic` 時，它只讀 train；評估請使用訓練器另產生的 YAML。
5. 要訓練新合成批次，先另存新檔名並用 `scp` 上傳到遠端 `training_inputs/`，再使用 `SYNTHETIC_ZIP="$PWD/training_inputs/新批次.zip" bash school_server/launch_handoff_yolo11n.sh E 6 e-new-01 "$M0"`。保留原始 ZIP 和原實驗結果，才能與舊批次比較。驗收完可回 UI 的 tmux 按 Ctrl+C 停止服務。

生成背景、角度與 seed 的舊設定見 [實驗一生成設定](EXPERIMENT_1_GENERATION_SETTINGS.md)和 [維修桌環境設定](EXPERIMENT_ENVIRONMENT_V2.md)。NEW 的 valid/test、最終評估影片及其近似影像須排除於參考、標註擴增及訓練來源。

## 4. 重跑歷史 M0 → C／D／E

[E 模型訓練流程](TRAIN_YOLO11N_E.md)收錄 Roboflow 資料版本、原始 M0 權重的使用方式、E 的評估指令與歷史驗證指標。

| 資料檔 | 原始圖片數（train / valid / test） | 用途 |
|---|---:|---|
| `training_inputs/new-workpiece-clean-yolov11.zip` | 374 / 47 / 47 | D/E 使用 train，所有組共用這份真實 valid/test |
| `training_inputs/tool2-clean-yolov11.zip` | 6,560 / 820 / 820 | 四組都只取 Tool 2 的 train |
| `training_inputs/factory-synthetic-reviewed.yolov8.zip` | 321 / 0 / 0 | C/E 使用已審核合成 train |
| `YOLO11n_tool2_experiments_models.tar.gz` | 四組各有 `.pt`、`.onnx` | 既有權重，可用歷史 M0 直接開始 fine-tune |

原始來源與授權標記隨 ZIP 保留。合成圖和參考圖請透過實驗室核准的管道交接。  
[NEW workpiece v10](https://universe.roboflow.com/project-m3ui7/new-workpiece-for-data-augmentat/dataset/10) · [Tool 2 v20](https://universe.roboflow.com/project-m3ui7/tool-2/dataset/20)

| 組別 | 初始化 | Train 來源 | 上限與 optimizer |
|---|---|---|---|
| M0 | `yolo11n.pt` | Tool 2 | 150 epochs、auto |
| C | 同一份 M0 `best.pt` | Tool 2 + 合成 | 60 epochs、AdamW |
| D | 同一份 M0 `best.pt` | Tool 2 + NEW | 60 epochs、AdamW |
| E | 同一份 M0 `best.pt` | Tool 2 + NEW + 合成 | 60 epochs、AdamW |

四組皆為 640 px、batch 32、seed 42、patience 30。C/D/E 的 `lr0=0.001`、`lrf=0.01`。合成圖上限為實際加入的真實 train 張數的 100%。訓練器會以來源 ID 和／或影像雜湊做真實資料的防洩漏過濾，依資料來源而異；實際入組張數以 `preparation_report.json` 為準。同來源拍攝序列及跨專案的近似影像仍需要人工確認。歷史腳本用兩張 GPU，新交接入口每次用一張已分配的 GPU，結果可能因硬體、套件版本與平行方式而有差異。

在訓練用的 **Linux tmux** 內執行，一組完成且確認權重後再執行下一組。

```bash
cd "$HOME/workpiece_handoff"
bash school_server/launch_handoff_yolo11n.sh M0 6 m0-01
M0="$PWD/school_training/handoff/m0-01/runs/M0/weights/best.pt"
bash school_server/launch_handoff_yolo11n.sh C 6 c-01 "$M0"
bash school_server/launch_handoff_yolo11n.sh D 6 d-01 "$M0"
bash school_server/launch_handoff_yolo11n.sh E 6 e-01 "$M0"
```

若沿用舊 M0，先在專案根目錄執行 `mkdir -p historical_models` 和 `tar -xzf YOLO11n_tool2_experiments_models.tar.gz -C historical_models`，再設定 `M0="$PWD/historical_models/YOLO11n_M0_Tool2_Base_best.pt"`，從 C 開始。記下 M0 的 SHA256，三組都用這份起始權重。比對歷史 C/D/E 時，看完整權重檔名。原專案另有 A/B/C/D 與 CONTROL/SYNTHETIC 實驗，字母組別依各自 launcher 解讀。

## 5. 評估、取回結果、完成交接

訓練中的 `results.csv` 是 validation 紀錄。完成後使用同一份 NEW test、相同推論設定分別評估各組。先做完各組與參數選擇，再用 test 結果報告。

```bash
cd "$HOME/workpiece_handoff"
for entry in m0-01:M0 c-01:C d-01:D e-01:E; do
  run="${entry%:*}"
  group="${entry#*:}"
  base="$PWD/school_training/handoff/$run"
  .school-env/bin/yolo detect val model="$base/runs/$group/weights/best.pt" \
    data="$PWD/school_training/handoff/e-01/dataset/data.yaml" split=test \
    imgsz=640 device=6 project="$PWD/school_training/test_reports" name="$run" \
    2>&1 | tee "$base/test.log"
done
```

如果沿用歷史 M0，將迴圈中的 M0 路徑換成解壓後的歷史 `.pt` 並將 log 存到已建立的資料夾。比較 precision、recall、mAP50、mAP50–95 以及各類別漏判或誤判，尤其是 C 對 M0、E 對 D。若有實驗室保存的同批測試影片，用 `school_server/evaluate_videos.py --help` 查看用法。影片工具提供視覺化和偵測統計，只有具備真值的資料才能計算準確率。

每次至少保存 `command.txt`、`environment.txt`、`train.log`、`test.log`、`dataset/preparation_report.json`、`augmentation_settings.json`、run 的 `args.yaml`、`results.csv` 及權重。另記錄日期、資料 SHA256、GPU、M0 SHA256 和觀察到的失敗案例。可在 **Windows PowerShell** 取回 E 組權重與測試報告。

```powershell
scp -r sslab-school:~/workpiece_handoff/school_training/handoff/e-01/runs/E/weights .\e-01-weights
scp sslab-school:~/workpiece_handoff/school_training/handoff/e-01/test.log .\e-01-test.log
```

建議交接驗收由學弟妹親手完成，交回四樣東西：遠端登入成功的主機名、SMOKE 的 `best.pt` 與 ONNX、至少一張人工審核後匯出的合成圖，以及正式模型的 test 報告。若 SSH 失敗，先確認校網／VPN、帳號與連線入口。若 SSH 成功而網頁連不上，先看 UI tmux 的啟動 log，再確認 PowerShell 的 tunnel 視窗仍在。若訓練中斷，用新 run ID 保留現有記錄後重跑。

## 6. 選修：工件／扳手分割

`training_runs/seg_v2_clean_yolov11.zip` 是獨立的 segmentation 資料，使用 polygon，類別為 `0=workpiece`、`1=wrench`。資料來源版本為 2。附的 `yolo11n-seg.pt` 可用於初始化。已練完偵測流程後，可在 Linux 執行下列指令啟動 segmentation 工作。

```bash
cd "$HOME/workpiece_handoff"
mkdir -p datasets/workpiece_wrench_seg
.school-env/bin/python -m zipfile -e training_runs/seg_v2_clean_yolov11.zip datasets/workpiece_wrench_seg
WORKPIECE_ROOT="$PWD" bash school_server/launch_yolo11seg.sh 6
tmux ls
```

此分支使用 120 epochs、640 px、batch 16，開訓前會驗證 polygon。日誌位於 `school_training/segmentation/logs/`。若接手者負責的是扳手小圓圈偵測，另讀 [wrench_circle 標註與訓練指南](WRENCH_CIRCLE_DATASET_PREPARATION_GUIDE.md)。
