# 工件生成環境 Prompt v2

## 啟用範圍

- 前 119 張已人工審核圖片：保留，不重新生成、不更改標註。
- 從第 120 張起：使用本文件的真實維修工作桌環境描述。
- 工件幾何配方仍為實驗 2「偏心法蘭＋不對稱肋條」。
- 相機角度、距離、工件占比、孔洞與螺絲規則均維持原成功設定。

## 影片觀察

兩段最終測試影片的背景主要不是整潔的自動化產線，而是長期使用中的實驗室／維修工作桌：

- 泛黃、龜裂、殘膠、油污與刮痕的桌面。
- 有使用痕跡的深綠色防靜電工作墊。
- 周圍散放鉗子、剪線鉗、扳手、氣動工具、尺、線材、棉棒、膠帶、透明零件袋與工程圖紙。
- 手機畫面有輕微偏色、曝光不均、動態柔化與感光雜訊。
- 部分畫面有人手從側面拿持工件，但主要辨識面仍可見。

## 新增環境 Prompt

```text
Match the visual domain of a busy, long-used university factory or maintenance workbench rather than a spotless automated production cell. Randomly vary the horizontal surface between a worn yellow-beige laminated bench with cracks, peeled tape, adhesive residue, dark grease and oil stains, scratches and dust, and a scratched dark-green anti-static work mat with smudges, metal filings and small debris. Surround the workpiece irregularly with only a few cropped, visibly used maintenance items near the outer image margins, selected from colored-handle wire cutters or pliers, an open-end wrench, a metal ruler, a pneumatic hand tool, a black cable, cotton swabs, masking tape, a crumpled transparent parts bag and a partially visible engineering drawing. The items must look naturally left from ongoing work, with varied colors and orientations, not arranged symmetrically or staged. Keep a clear inspection zone around the workpiece: background items may approach it but must not cross over, overlap or hide the housing, flange, holes or installed screws. In a small minority of images, one natural bare hand may enter from one image edge and lightly hold only an outer side wall; it must not cover the top face or any annotation target. Keep surrounding bench holes, loose hardware and circular tool details sparse so they are not confused with workpiece targets.
```

## 新增負面限制

```text
spotless factory, showroom-clean fixture, pristine polished workbench, sterile empty bench, symmetric staged tool layout, background hole beside the workpiece, loose background fastener, tool crossing the workpiece, tool covering the workpiece, hand covering the top face, hand covering a hole, hand covering a screw, extra workpiece, readable brand logo
```

## 保持不變的核心設定

- 視角：要求 88–90 度，使實際輸出約落在 80–90 度。
- 距離：70–90 cm，優先約 80 cm。
- 工件占畫面：約 20–30%。
- 小孔洞：2–4 個，彼此分散。
- 已安裝螺絲：1–2 顆，黑色或銀色，必須正確鎖入孔洞且凹槽朝上。
- 模型：FLUX.2 Klein。
- 第 2 組幾何：偏心法蘭、不對稱外框與不對稱肋條。

## 實驗紀錄建議

後續評估時可將合成資料標記為：

- `env_v1_clean_fixture`：前 119 張中的合成圖。
- `env_v2_used_maintenance_bench`：第 120 張起的合成圖。

這樣可以額外比較加入真實桌面域後，對兩段最終測試影片的泛化能力是否改善。
