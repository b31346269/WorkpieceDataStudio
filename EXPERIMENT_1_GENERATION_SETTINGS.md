# 實驗 1：方形外殼＋中央法蘭生成設定

這份文件記錄第一組成功產生並人工標註 100 張圖片時所使用的實際設定，方便後續重現與比較 A／B／C／D 實驗。

## 實驗目標

- 工件主特徵：低矮方形或短矩形鑄鋁外殼＋中央法蘭／軸承座。
- 改變外輪廓、比例、法蘭尺寸、肋條、散熱鰭片及安裝耳位置。
- 保持孔洞及螺絲稀疏，降低人工修正標註的負擔。
- 使用真實流水線、金屬工作臺或加工治具背景。
- 類別固定為 `0=hole`、`1=screw`、`2=tool`。

## 正向 Prompt

```text
Photorealistic factory inspection photograph of a clearly different cast-aluminum gearbox housing on a real horizontal slotted metal conveyor or machining fixture. Near-overhead 80 to 90 degree view from 70 to 90 cm, with the workpiece occupying about 25 to 35 percent of the image. Preserve high variation in the outer silhouette, proportions, flange position, cast reinforcement ribs, cooling fins and mounting ears. The top face must remain sparse: approximately 4 to 6 well-separated functional circular holes and only 1 to 2 installed screws. Include a recognizable machined flange, bearing seat, or asymmetric rib structure. Every screw is fully seated flush inside its matching hole with a visible Phillips, Torx or hex-socket recessed drive head; use either matte black-oxide steel or natural silver steel. No loose hardware and no exposed threaded shaft. Realistic aluminum texture, machining marks, mild factory wear, natural industrial illumination and smartphone inspection photography.
```

## 負向 Prompt

```text
dense holes, many holes, nine or more holes, dense screws, four or more screws, upright screw, inverted screw, upside-down screw, exposed threaded shaft, loose screw, floating screw, stacked screw, ambiguous stud, plain featureless lid, identical housing shape, white studio background, CGI, CAD, close-up, low camera angle
```

## 生成參數

| 項目 | 設定 |
|---|---|
| 模型／Provider | `flux2_klein`（FLUX.2 Klein 9B BF16） |
| 品質模式 | `shape_variation` |
| 場景 | `conveyor_fixture` |
| 構圖 | `letterbox` |
| 輸出尺寸 | 1024 × 1024 |
| 要求 Strength | `0.32` |
| 要求 IP-Adapter scale | `0.72` |
| 要求 Guidance scale | `5.5` |
| 要求 Steps | `32` |
| FLUX 實際 Guidance | `1.0`（蒸餾模型固定） |
| FLUX 實際 Steps | `4`（蒸餾模型固定） |
| 預標註模型 | 使用者上傳的 `best.pt` |
| 預標註信心門檻 | `0.35` |
| 人工審核 | 每張修正框選後按「接受並儲存」 |

> UI/API 仍會記錄要求的 Strength、IP-Adapter、Guidance 與 Steps；FLUX.2 Klein 實際推理採模型固定的 4 steps、guidance 1.0。

## 構圖與內容規則

- 視角為接近正上方的 80–90 度，不使用低角度或側面為主的產品照。
- 相機距離約 70–90 cm。
- 工件約占畫面 25–35%，四周保留工作臺或治具背景。
- 目標約 4–6 個分散孔洞及 1–2 顆已鎖妥螺絲。
- 螺絲頭必須朝上並帶有清楚凹槽，可使用黑色氧化鋼或銀色金屬。
- 不允許倒插、直立、裸露牙桿、鬆散、重疊或懸浮螺絲。
- `best.pt` 只負責預標註，最後標註以人工確認為準。
- 不用自動框數直接淘汰圖片，因為 `best.pt` 可能將法蘭或同一孔洞重複偵測。

## 第一組完成狀態

- 已人工完成：100 張。
- 這 100 張作為「方形外殼＋中央法蘭」合成資料組。
- 後續實驗應沿用相同模型、構圖、背景、孔洞／螺絲限制與人工審核流程，只替換指定的工件幾何特徵。

## 實驗 2 僅替換的幾何特徵

第二組改為「偏心法蘭＋不對稱肋條」：大型法蘭必須明顯偏向一側或角落，幾何中心保留實心鑄件、非對稱肋條交會或淺矩形凹面；外輪廓、安裝耳、肋條及散熱鰭片不可左右鏡像。其餘生成與預標註參數全部保持不變。
