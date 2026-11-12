# `wrench_circle` 資料集準備指南

這份文件是給資料拍攝與 Roboflow 標註使用的操作規格。目標只有一個：找出板手頭上方指定的小圓形特徵，類別名稱固定為 `wrench_circle`，任務是 Object Detection bounding box。

## 先記住四個原則

1. 只框小圓圈，不框整個板手頭，也不框整支板手。
2. 圖片必須是沒有紅圈、紫色輪廓、偵測框、類別文字或 Unity UI 的原始 RGB 畫面。
3. 完全看不到圓圈時不畫框；不要用上一幀的位置補框。
4. train、validation、test 要按「拍攝序列／實體工件／場景群組」切分，不可把同一段影片的相鄰影格隨機分散到三組。

## Roboflow project 設定

- Project type：`Object Detection`
- Project name：建議 `Wrench Circle Detection`
- Classes：只有一個 `wrench_circle`
- 不加入 `workpiece`、`wrench`、`hole`、`screw` 等其他類別
- License、visibility 依原專案政策設定；不要把含人物或工廠資訊的原始影片公開分享
- 第一版以 **full-image detector** 為主，保持 fallback 能搜尋整張相機畫面。之後若要比較 ROI，再建立獨立 ROI 版本，不要把不同座標／尺度的資料不註明混在一起

## 可以從哪裡拿圖片

可以使用目前 workpiece+wrench segmentation project 裡的原始圖片，但必須：

- 只匯入沒有疊圖的原始 image
- 舊的 workpiece/wrench polygon 不直接拿來當 `wrench_circle` 標註
- 有舊 `.txt` 或 polygon label 時，先不要一併上傳；新 project 重新畫 box
- 已經畫上紅色圈選、紫色工件輪廓、紅色板手輪廓、綠色偵測框或螢幕錄影 UI 的圖片不要放入訓練資料
- 圖片中同時有孔洞／螺絲並不需要刪除；本 project 只標可見的 `wrench_circle`，但這些圖片未來若拿去訓練 single 3-class detector，必須重新完整標註可見 `hole`／`screw`

## 什麼畫面要標 `wrench_circle`

### 正樣本：要畫一個 box

- 圓圈完整可見。
- 圓圈部分被手、工件或反光遮住，但仍能明確知道它是指定特徵。
- 圓圈大小、旋轉、亮度、金屬反光和背景不同，但仍能辨認。

box 畫在「影像中實際可見、可辨認的小圓形特徵」外接矩形上。第一版不猜被遮住的完整圓周；同一個遮擋規則要在 train、val、test 一致。

### Negative sample：不要畫任何 box

- 畫面有板手，但指定圓圈完全不可見。
- 只有板手柄、板手頭背面或其他方向，找不到指定圓圈。
- 圓圈只剩很小、無法和螺絲／孔洞區分的反光碎片。
- 畫面沒有板手，或只有工件上的孔洞／螺絲。
- 背景有外觀相似的圓形物，例如螺帽、孔、旋鈕、軸承，但不是板手指定圓圈。

Negative 圖片必須保留在 dataset；如果圖片裡沒有任何可見 `wrench_circle`，labels 應是空檔案，而不是刪除圖片。

### 不可標的情況

- 圓圈被完全遮住：不畫框。
- 只有人工紅圈看得到，原始圓圈本身看不到：不要用紅圈位置做標註。
- 不確定是指定圓圈還是工件螺絲：先放入待確認資料夾，不要硬標。
- 框住整個板手頭、套筒或整支板手：這是錯誤標註。

## 拍攝內容矩陣

每一種條件都不必拍成連續長影片；可錄短序列，再每隔 10–15 幀抽一張，避免近重複影格。下表是最低涵蓋方向，實際可依設備可行性調整。

| 變因 | 至少涵蓋 |
|---|---|
| 工件 | 目前已知工件、至少 2–3 種不同形狀／材質工件、至少 1 種未來不會放入 train 的 unseen 工件 |
| 板手方向 | 頭部朝左／右／上／下；旋轉約 0°、45°、90°、135°；斜放與水平放置 |
| 相機角度 | 近似俯拍、輕微斜拍、明顯斜拍；相機左右偏移；不同高度 |
| 距離／尺度 | 遠景（圓圈小）、中景、近景（圓圈大）；不要全部只在同一距離 |
| 光線 | 均勻光、側光、較暗、亮反光、金屬高光；避免所有正樣本都在同一背景 |
| 動態 | 靜止、慢速移動、正常移動、輕微手震與 motion blur |
| 遮擋 | 手部部分遮擋、板手部分被工件遮擋、圓圈完全不可見的 negative |
| 背景 | 白色工作台、木桌、金屬台、工廠雜物；相似圓形干擾物 |
| 場景組成 | 只有板手、板手放在工件上、板手靠近但未覆蓋圓圈、無板手的工件畫面 |

### 建議資料量

第一版可先準備：

- `wrench_circle` 正樣本：至少 300 張，建議 500–800 張
- 完全不可見／沒有板手／相似圓形干擾的 negative：至少 100 張，建議占總數 25–40%
- 至少 6 個獨立 recording sequence；不要用一支影片的 500 張相鄰影格冒充 500 個樣本
- 每個主要角度／距離／光線條件至少有 20–30 張相互間隔的圖片

數量本身不能保證泛化；條件多樣性和獨立序列比重複影格更重要。

## Train / validation / final test 怎麼分

先為每張圖片保存 metadata：`image_id`、`recording_sequence`、`workpiece_family`、`physical_wrench`、`camera_view`、`distance_group`、`lighting_group`、`has_circle_visible`。

分組規則：

- Train：已知工件與部分場景／角度／距離。
- Validation：不同拍攝序列，不要只從 train 影片抽相鄰幀。
- Final test：至少一個從未出現在 train 的工件家族，以及從未用來調閾值的獨立拍攝序列。
- 同一段影片、同一次擺放、相鄰幀、相同原圖衍生裁切不可跨 split。
- 你提供給我們做過診斷、調參或看過輸出的影片，應歸類為開發資料，不再宣稱是完全盲測。

## Full-image 與 ROI 版本

### 第一版：full-image

整張原始相機圖直接標 `wrench_circle`。這是目前推薦的起點，因為上游 wrench segmentation 失敗時仍可搜尋全圖，也最容易和部署畫面對齊。

### 第二版：ROI 比較組

等 full-image baseline 有資料後，再用實際 wrench box/mask 加 10–25% padding 製作 RGB ROI。保存每張 ROI 的來源影像、原圖框、padding、縮放比例和 split 群組。ROI 的 label 必須轉成 ROI 座標。

ROI 版本要和 full-image 分開報告，不能將兩種輸入的 mAP 直接混成一個數字。若部署 fallback 會 full-image 搜尋，模型或至少驗證資料也必須包含 full-image 路徑。

## 標註自我檢查

上傳前逐張確認：

- 圖片沒有紅／紫手動畫線、UI、偵測框或信心文字。
- 每個可見指定圓圈恰好一個 `wrench_circle` box。
- 沒有把板手頭、套筒或柄一起框進去。
- 相似螺絲／孔洞沒有誤標成 `wrench_circle`。
- 完全不可見圓圈的 labels 是空的，不是沿用前一幀框。
- 部分遮擋的 box 遵守同一種「可見範圍」規則。
- 每張圖的 metadata 與 recording sequence 正確，沒有 train/test 混組。

Roboflow 產生 dataset version 前，先看 class count、未標註圖片數、sample gallery 和 label overlay。訓練前在本機／學校伺服器跑資料 audit；遇到未知類別、缺 label、空圖片、相鄰序列跨 split 就停止。

## 與 single 3-class detector 的關係

本 project 的 `wrench_circle` 圖片不能直接加入 single 3-class training set，除非重新標出同一張圖裡所有可見 `hole` 和 `screw`。否則三類模型會把未標註的真目標當背景，造成錯誤負例。

## 你目前不需要準備的東西

- 不需要重新描 `workpiece`／`wrench` segmentation polygon。
- 不需要現在就製作 wrench-only segmentation dataset。
- 不需要標註被完全遮住的孔洞／螺絲。
- 不需要判斷板手圓圈和螺絲的機械同軸或鎖緊狀態。
- 不需要在圖片上畫紅圈提示模型。

## 第一批交付建議

先交付一個小型但多樣的開發批次：約 120–180 張 full-image（正樣本約 80–120、negative 約 40–60），涵蓋至少 4 個 recording sequence、2 種工件家族、3 種相機角度和 3 種距離。先做標註品質 audit，再擴充到正式資料量；不要一開始上傳數百張未檢查的相鄰幀。
