# IronLane

本 repo 是「1141 嵌入式智慧影像分析與實境界面」期末專題用的 JetBot 自走車導航範例：**把資料夾丟到 JetBot 上、開 notebook、跑起來就能動**。

- 主要執行入口：`final.ipynb`
- 專案目標：影像推論（ResNet18 方向點 + YOLOv8 障礙策略）→ 融合控制 → 輸出左右輪速度

> 注意：`pyproject.toml` 只用於**開發端**管理，**不代表 JetBot 真正的依賴環境**。JetBot 的 runtime 依賴（TensorRT / PyCUDA / jetbot 套件等）是車上原生環境提供的。

---

## Demo Pipeline

Camera → FrameHub → Worker(s) 推論 → ResultHub → Controller → Motors

- ResNet18：輸出轉向點 `steer_xy`
- YOLOv8：輸出 slow/stop/avoid/halt 等策略（policy）
- Controller：融合 steer / yolo，最後輸出左右輪速度 `[-1, 1]`

---

## Quick Start

### 1) 部署到 JetBot

你只需要把這兩個東西丟上去：
- `ironlane/`（整個資料夾）
- `final.ipynb`

建議放在：

`~/jetbot/notebooks/ironlane`  
`~/jetbot/notebooks/final.ipynb`

或是自己想要的路徑上


### 2) 準備模型引擎（TensorRT）

本 repo 不包含 TensorRT engine，需要在 JetBot 上把 ONNX 轉成 TRT engine。

這個專案預設用兩個檔名（可以自行改 notebook 內路徑）：

- ResNet18 engine：`final_new2.plan`
- YOLOv8 engine：`finalreport.engine`

用 JetBot 內建的 `trtexec` 轉換（在 JetBot 上跑）：

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=<onnx file path> \
  --saveEngine=<output engine path> \
  --fp16 \
  --shapes=input:1x3x224x224 \
  --workspace=2048
```

> 建議：把 engine 放在 `final.ipynb` 旁邊，或放到 `ironlane/` 內固定位置，然後在 notebook 內把路徑改成一致即可。

### 3) 開啟 `final.ipynb` 並運行

開始控制 loop 後，車會依據推論結果輸出馬達速度

## 專案結構

- ironlane/control.py：控制器融合邏輯（steer + lane + yolo → left/right speed）
- ironlane/hub.py：多執行緒共享資料（最新影像/最新推論結果/停止旗標）
- ironlane/modules.py：TensorRT engine 包裝 + worker thread（ResNet / YOLO）
- ironlane/yolo_policy.py：YOLO 解碼 + NMS + policy 生成（slow/stop/avoid/halt）
- ironlane/utils.py：數值工具（clamp / clip 等）
- final.ipynb：端到端 runtime 範例（相機→推論→控制→馬達）

## 提醒

JetBot 資源有限，非必要的視覺化可能會造成效能瓶頸，進而導致相機畫面卡住或模型推論延遲。這裡特別指在即時相機畫面上使用 OpenCV 進行繪製（例如畫 bbox、文字疊圖等）這類操作，建議非必要時關閉或降低更新頻率。


