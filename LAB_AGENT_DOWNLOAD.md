# 實驗室 Agent：下載與建立訓練環境

請在實驗室電腦的 VS Code Terminal 執行。不要使用同學帳號；使用你自己的 Windows／Server 帳號。

## 1. 下載最新專案

```bash
cd ~
git clone https://github.com/b31346269/WorkpieceDataStudio.git
cd WorkpieceDataStudio
git pull origin main
```

若資料夾已存在，請不要刪除它，先執行：

```bash
cd ~/WorkpieceDataStudio
git fetch origin
git status
git pull --ff-only origin main
```

## 2. 確認版本完整

以下檔案都必須存在：

```bash
test -f school_server/train_yolo.py && echo OK
test -f school_server/launch_yolo11n_group.sh && echo OK
test -f school_server/bootstrap.sh && echo OK
test -f SCHOOL_SERVER_HANDOFF.md && echo OK
```

GitHub 已包含 25 張工廠參考圖，位置是：

```text
training_inputs/factory_references/
```

Roboflow 資料集 ZIP 不放在 GitHub，請另外取得並放到：

```text
training_inputs/new-workpiece-clean-yolov11.zip
training_inputs/tool2-clean-yolov11.zip
```

不要把 Roboflow download key、Server 密碼、SSH 私鑰提交到 GitHub。

## 3. 連線到 GPU Server

沿用實驗室已驗證的 SSH 入口；若入口是公開轉送位址：

```bash
ssh ping@140.123.97.160
```

登入後確認：

```bash
hostname
nvidia-smi -i 6
nvidia-smi -i 8
```

只使用實體 GPU 6 或 8。優先 GPU 6；GPU 6 被占用時才使用 GPU 8。不要使用 GPU 2、3，也不要同時啟動兩組訓練。

## 4. 建立 Server 端專案

在 Server 上建立乾淨目錄後，將此專案部署到：

```text
~/workpiece_data_studio
```

Windows 實驗室電腦可從專案根目錄執行：

```powershell
.\school_server\deploy-school-server.ps1 -HostAlias sslab-school -Bootstrap
```

部署後確認 Server 端有：

```bash
cd ~/workpiece_data_studio
ls school_server/train_yolo.py school_server/launch_yolo11n_group.sh
```

## 5. 四組 YOLO11n 訓練

資料組別固定為：

```text
A = NEW workpiece
B = NEW workpiece + Tool 2
C = NEW workpiece + Tool 2 + 審查後工廠合成圖
D = NEW workpiece + Tool 2 + 同一批審查後工廠合成圖
```

每次只執行一組，優先 GPU 6：

```bash
cd ~/workpiece_data_studio
bash school_server/launch_yolo11n_group.sh A 6
bash school_server/launch_yolo11n_group.sh B 6
bash school_server/launch_yolo11n_group.sh C 6
bash school_server/launch_yolo11n_group.sh D 6
```

若 GPU 6 被占用，才把最後的 `6` 改成 `8`。訓練 log 位於：

```text
~/workpiece_data_studio/school_training/logs/
```

完整流程、資料洩漏防護與 UI 生成步驟請閱讀 `SCHOOL_SERVER_HANDOFF.md`。
