# 實驗室電腦交接與 GPU 6/8 訓練流程

這份手冊是在實驗室電腦操作時使用。所有訓練只使用目前獲准的實體 GPU **6 或 8**；GPU 2、3 不可指定，也不要啟動多 GPU 訓練。

## 0. 先確認連線與權限

在實驗室電腦開 VS Code 的 Terminal，使用同學目前成功的 Server 連線方式。若 Server 內網位址仍相同，可先測試：

```bash
ssh ping@192.168.100.150
```

若同學使用的是其他 SSH host alias，請沿用那個 alias，不要猜測新的公開 IP 或分享器位址。

登入 Server 後先執行：

```bash
hostname
nvidia-smi -i 6
nvidia-smi -i 8
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv
```

確認主機是 `sslab-server`，並記錄 GPU 6、8 的型號與可用記憶體。訓練優先使用 GPU 6；若 GPU 6 被占用，才改用 GPU 8。不要改用 GPU 2 或 3。

## 1. 在實驗室電腦取得專案

如果 GitHub 已包含最新的 `school_server/` 腳本：

```bash
cd ~
git clone https://github.com/b31346269/WorkpieceDataStudio.git
cd WorkpieceDataStudio
```

注意：GitHub 若沒有最新未提交的本機變更，不能直接用舊 clone 訓練。至少要確認下列檔案存在：

```text
school_server/bootstrap.sh
school_server/deploy-school-server.ps1
school_server/train_yolo.py
school_server/launch_yolo11n_group.sh
```

若缺少檔案，請停止，不要執行訓練。這表示 GitHub clone 是舊版；請把本機最新專案 ZIP 傳到實驗室電腦，或先將最新版本正式推送到 GitHub，再重新 clone。不要用 Server 上現有的舊 `WorkpieceDataStudio` 目錄混合新舊檔案。

目前版本至少需要：

```text
school_server/train_yolo.py
school_server/launch_yolo11n_group.sh
school_server/connect-school-server.ps1
school_server/bootstrap.sh
```

## 2. 從實驗室電腦部署到 GPU Server

在 VS Code PowerShell Terminal 執行（HostAlias 改成同學實際可連線的 alias）：

```powershell
.\school_server\deploy-school-server.ps1 -HostAlias sslab-server -Bootstrap
```

這會建立遠端資料夾：

```text
~/workpiece_data_studio
```

部署腳本不會把大型 Roboflow ZIP 一起打包，因此接著把兩個資料集 ZIP 上傳到 Server。ZIP 檔名必須是：

```text
training_inputs/new-workpiece-clean-yolov11.zip
training_inputs/tool2-clean-yolov11.zip
```

可在實驗室電腦執行：

```bash
scp training_inputs/new-workpiece-clean-yolov11.zip ping@192.168.100.150:~/workpiece_data_studio/training_inputs/
scp training_inputs/tool2-clean-yolov11.zip ping@192.168.100.150:~/workpiece_data_studio/training_inputs/
```

不要把 Roboflow download key 寫入 Git、Markdown 或公開訊息；若曾公開過，訓練前請到 Roboflow 重新產生 key。

## 3. 啟動遠端 WorkpieceDataStudio UI（需要生成工廠合成圖時）

在實驗室電腦執行：

```powershell
.\school_server\connect-school-server.ps1 -HostAlias sslab-server -RemoteGpu 6
```

保持此 Terminal 不要關閉，瀏覽器開：

```text
http://127.0.0.1:7866
```

在 UI 中使用 `best.pt` 做預標註，選擇工廠場景，人工審查後匯出：

```text
factory-synthetic-reviewed.yolov8.zip
```

匯出的 ZIP 放到：

```text
~/workpiece_data_studio/training_inputs/factory-synthetic-reviewed.yolov8.zip
```

## 4. 建立並依序訓練四組 YOLO11n

四組使用相同 `yolo11n.pt`、150 epochs、640 image size、seed 42；只改變訓練資料：

```text
A = NEW workpiece
B = NEW workpiece + Tool 2
C = NEW workpiece + Tool 2 + 審查後工廠合成圖
D = NEW workpiece + Tool 2 + 同一批審查後工廠合成圖
```

每次只啟動一組，優先指定實體 GPU 6：

```bash
cd ~/workpiece_data_studio
bash school_server/launch_yolo11n_group.sh A 6
bash school_server/launch_yolo11n_group.sh B 6
bash school_server/launch_yolo11n_group.sh C 6
bash school_server/launch_yolo11n_group.sh D 6
```

若 GPU 6 正在使用，才將最後的 `6` 改成 `8`；不可同時指定 `6,8`。

不要同時啟動兩組。每組完成後確認 log：

```bash
tail -n 40 school_training/logs/YOLO11n_A.log
nvidia-smi -i 6
```

結果會在：

```text
~/workpiece_data_studio/school_training/YOLO11n_A/
~/workpiece_data_studio/school_training/YOLO11n_B/
~/workpiece_data_studio/school_training/YOLO11n_C/
~/workpiece_data_studio/school_training/YOLO11n_D/
```

## 5. 重要資料與評估原則

- NEW workpiece 的 valid/test split 是固定 holdout，不可拿來做合成圖或訓練。
- Tool 2 只作為輔助訓練資料；不可用它覆蓋 NEW 的 valid/test。
- 合成圖一定要人工審查後才能進 C、D。
- 四組都要在同一份固定 NEW test 上比較 mAP50、mAP50-95、precision、recall。
- 絕對不要把 `C:\Users\b3134\Desktop\LIVEs` 複製、修改或刪除。

## 6. 若 SSH 仍失敗

請不要改 `.160` 分享器設定，也不要使用同學帳號。請把以下資訊交給 Server 管理者：

```text
Server hostname: sslab-server
Server internal SSH target: 192.168.100.150:22
Required physical GPU: 6 only
```

若 Server 有 Tailscale，請管理員只分享 `sslab-server` 給你的個人帳號，再使用該私有位址 SSH。
