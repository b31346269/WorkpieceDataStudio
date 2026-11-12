# 工件／板手分割與 Dynamic ROI 整合規格

## 結論

目前的 `tool2-clean-yolov11.zip`、`new-workpiece-clean-yolov11.zip`、
`factory-synthetic-reviewed.yolov8.zip` 都是偵測框資料：標註列是
`class cx cy width height`，不是分割多邊形。更重要的是，現有 E 模型的
`tool` 類別曾混入板手影像，不能再把它當成乾淨的工件類別；板手應由 Unity
另外掛載的 `yolo11n_tool`（Wrench Model Asset）負責。未來主偵測模型應只保留
`hole`、`screw`，分割模型則另用 `workpiece`、`wrench` 兩個 polygon 類別。
因此不能直接把這三個 ZIP 或兩個 detector 的框當成高品質
`workpiece + wrench` segmentation model，也不能把矩形框冒充成真實 mask。

新的分割資料集應固定成兩個類別，並在匯入時做語意重命名：

```yaml
names:
  0: workpiece
  1: wrench
```

每個影像要用 YOLO segmentation polygon 標註工件與板手的實際可見輪廓。
`hole`、`screw` 不要放入這個分割模型；它們仍由現有偵測模型處理。

| 來源 | 目前模型內名稱 | 分割資料集名稱 |
|---|---|---|
| 未來主偵測／分割資料 | `hole`、`screw` | 分割資料另標 `workpiece` |
| 額外板手 detector（目前 `wrenchModelAsset: yolo11n_tool`） | `tool` | `wrench` |

目前 E 的三類模型只作既有 baseline；不要把它直接當成未來乾淨的
`hole/screw` 模型。重新訓練時要先從 Tool2、NEW、合成資料移除 `tool` 標註，
再建立只含 `hole`、`screw` 的模型。

## ROI 是否需要重新設計？

不需要再手動畫一個「分割專用固定 ROI」。建議流程是：

```text
完整畫面（或只用很寬的安全 ROI）
    ↓
YOLO11n-seg：同時找 workpiece、wrench 的 mask
    ↓
每個 mask 的外接框 + 10～20% padding + 邊界 clamp
    ↓
兩個獨立 Dynamic ROI
    ├─ Workpiece ROI：供 hole/screw detector
    └─ Wrench ROI：供板手／target-center 或後續 pose
```

也就是說，**mask 是像素級範圍，ROI 是後續裁切用的矩形**。分割模型先
負責找物體；ROI 再由 mask 自動產生。工件和板手不應共用一個 ROI，否則
板手伸出工件外時會被裁斷。

固定 ROI 仍保留作 fallback：分割信心不足、mask 接近畫面邊界或追蹤遺失時，
暫時回到固定 ROI／完整畫面，而不是把座標跳到 `(0, 0)`。

## 目前可立即使用的工具

- `school_server/segmentation_dataset.py`：嚴格檢查 train/val 是否真的含有
  polygon；遇到目前五欄 detection label 會停止並報告，不會產生假 mask。
- `school_server/prelabel_segmentation_candidates.py`：用主工件 detector 與
  額外板手 `tool` detector 產生候選框與 overlay，供人工修成 polygon；輸出
  明確標成 candidate，不會被誤當成可訓練的 segmentation label。
- `school_server/train_yolo11_seg.py`：在學校 `ping` 環境使用 GPU 6 或 8
  訓練 `YOLO11n-seg`，並匯出固定 640 的 ONNX。
- `school_server/segment_video.py`：用訓練完成的 segmentation 模型逐幀評估
  影片，輸出每個物件的 box 與 normalized polygon JSON，方便驗證 Dynamic ROI。
- Unity `YoloDet_windows.cs` 的固定 ROI 映射已改成依 `cameraRawImage`／
  `ROIRect` 實際 local space 計算，並處理鏡像、非置中 ROI、畫面外 clamp；
  layout 尚未完成時會回退完整畫面，不會把錯誤 UV 餵給模型。
- Unity `DynamicRoiCalculator.cs` 提供 polygon → padded/clamped ROI 及平滑
  函式，待 segmentation ONNX decoder 完成後由 workpiece／wrench 各自呼叫。

## 在 ping 伺服器的執行順序

1. 先建立兩類 polygon 資料集，資料夾布局為
   `train/images`, `train/labels`, `valid/images`, `valid/labels`，並建立
   `data.yaml`。兩個測試影片要整段保留作 test，不可抽相鄰影格混入 train。
2. 在 `~/workpiece_data_studio` 啟用已存在的環境並驗證：

   ```bash
   source .school-env/bin/activate
   python school_server/segmentation_dataset.py \
     --data datasets/workpiece_wrench_seg/data.yaml
   ```

3. 驗證報告 `ready_for_yolo_seg: true` 後再訓練。只用目前核准的 6 或 8
   號卡，不要把 2、3 號混入：

   ```bash
   python school_server/train_yolo11_seg.py \
     --data datasets/workpiece_wrench_seg/data.yaml \
     --model yolo11n-seg.pt --epochs 120 --imgsz 640 \
     --batch 16 --device 6 \
     --name YOLO11n_workpiece_wrench_seg
   ```

4. 用兩段保留的測試影片評估：

   ```bash
   python school_server/segment_video.py \
     --model school_training/segmentation/runs/YOLO11n_workpiece_wrench_seg/weights/best.pt \
     --source test_videos/808041065.167705.mp4 \
     --output school_training/segmentation/video_167705.json \
     --device 6
   ```

5. 通過 mask IoU、漏分割率、Dynamic ROI 是否截斷物體的檢查後，才匯出
   ONNX 到 `LIVEs/Assets/Models`，再接 Unity 的 segmentation decoder。這
   個 ONNX 的 output layout 與目前 detection decoder 不同，不能直接替換
   現有 `YoloDet_windows.cs` 的 `TensorProccess`。

目前尚未把 segmentation ONNX decoder 硬接到 Unity，原因是尚無合法的兩類
polygon 資料與對應模型；先把 detection decoder 當 segmentation decoder 會
把 mask coefficient/prototype 解碼錯，反而讓 ROI 比目前更不可靠。

## 資料與評估原則

- 每個影像可有一個 workpiece 與一個 wrench；被遮擋仍沿可見輪廓標註，並
  在 manifest 記錄 occlusion。
- 至少按「影片／場景／距離／光線」分割 train、valid、test；同一段影片的
  相鄰影格不能跨 split。
- 工件分割與板手分割分開計算 IoU、Dice、漏檢率；不能只看總體平均。
- 兩段老師展示用影片只作最後 test，不能拿來訓練或調參。
- 外接 WebCam 不需要新的手動 ROI；模型收到完整影像後，Dynamic ROI 由
  mask 自動算出。現有固定 ROI 的座標映射仍需沿用同一份影像到 UI 的轉換，
 不能把 ROI 尺寸直接當成 UV 比例。

## 重要限制

如果目前沒有 polygon 標註，最正確的下一步是先標註少量代表性影像（例如每
個場景／距離／光線各 30～50 張），而不是立即訓練一個看似完成、實際只學
矩形框邊界的 segmentation model。可先用現有 `best.pt` 產生 workpiece
候選框，再人工修成 polygon；板手則必須新增類別與輪廓標註。
