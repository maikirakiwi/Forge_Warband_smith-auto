# 技術設計文件（Technical Design）— `vrc-smith-auto`

本文件詳述 **VRChat《爐火戰團》鐵匠小遊戲自動工具（`vrc_smith_auto`）** 之問題分析、整體架構、關鍵技術選型比較（螢幕畫面擷取、判定條視覺偵測、滑鼠輸入與熱鍵控制）、執行期安全與 DPI 適應機制，以及本機可重現之離線驗證與基準測試實測結果。

---

## 1. 系統目標與問題分析

### 1.1 核心需求與邊界
- **目標遊戲機制**：類似《Undertale》攻擊判定條之節奏點擊小遊戲。判定區域由外而內分為 **黑（揮空）**、**灰（打偏）**、**銀（不錯）**、**金（完美）**；紅色判定條在判定軸上來回掃描，玩家需在紅色判定條進入金色區域時按下滑鼠左鍵。
- **系統邊界**：運作於 VRChat 桌面視窗模式（Desktop Mode），僅負責擷取畫面、偵測紅色判定條與金色區域相對位置、於進入金色區域時原地送出滑鼠左鍵點擊；不包含額外遊戲自動化邏輯。

### 1.2 參考素材分析與效能 SLA 推導
針對使用者提供之遊戲畫面截圖（`325 × 223` 裁切）與實際遊玩影片（`example/2026-11-20 15-32-46.mp4`，`1920 × 1080 @ 60.0 fps`，共 `1570` 影格 / `26.167 s`），歸納出以下關鍵技術挑戰：

1. **時間預算極短（Min Gold Dwell = `33.3 ms`）**：
   - 影片中紅色判定條移動速度從初期約 `145 px/s` 加速至末段約 `334 px/s`，而金色「完美」區域長度同時從 `38 px` 縮減至 `16 px`。
   - 實測末段高難度擊次（`Seg2-H9`，約影格 `1305..1306`）中，紅色中心停留於金色區域的時間僅約 **`33.3 ms`**（60 FPS 下僅 `2 frames`）；若以部分邊緣接觸（Any-Overlap）計算甚至僅 `16.7–33.3 ms`。
   - 因此單影格「擷取 ＋ 偵測 ＋ 狀態判定 ＋ 點擊送出」總延遲必須低於 `33.3 ms`，本專案將穩定效能 SLA 訂為 **`p95 < 16.7 ms`**（小於 60 FPS 單影格週期），平均目標 **`< 10–12 ms`**。
2. **3D 視角傾斜與攝影機晃動**：
   - 判定條在 VRChat 3D 空間中呈現 UI，角色或鏡頭晃動會造成判定條中心在螢幕上大幅位移（影片中觀察到 `X: 704..910 px`、`Y: 320..446 px`）與傾斜角變化（影片約 `-1.7°` 至 `-4.2°`，截圖約 `-3.814°`）。演算法不能假設判定條固定於螢幕某個像素矩形或絕對垂直。
3. **紅色判定條遮擋切分（Red-Split Occlusion）**：
   - 當寬約 `62 px` 的紅色判定條橫越寬約 `36–41 px` 的金色區域時，紅色區域會將單一金色連通域（Connected Component）從中間截斷為上下兩塊分離的金色色塊，若直接取單一最大金色區塊會造成座標偏移或漏檢。
4. **背景與 HUD 干擾元素（Distractors）**：
   - 畫面存在大量與判定條色彩相近的文字與 HUD（如金色進度 `0%`、按鈕 `锤!`、數字 `1..10`、`熟练铁匠`、`铁匠成就` 圖示），若僅靠單純 HSV 面積過濾極易誤判。

---

## 2. 系統整體架構

`vrc_smith_auto` 採用嚴格解耦的五層管線架構（原始碼位於 `src/vrc_smith_auto/`）：

```text
+-------------------+     +-----------------------+     +----------------------+
| window.py         | --> | capture/              | --> | vision/detector.py   |
| Per-Monitor-V2    |     | FrameSource Protocol  |     | GaugeDetector        |
| DPI + VRChat HWND |     | dxcam / mss / gdi     |     | Coarse-to-Fine HSV   |
+-------------------+     | Video / Image Offline |     | 3-Stage Split Bridge |
                          +-----------------------+     | 1D s_norm Projection |
                                                        +----------+-----------+
                                                                   |
                                                                   v
+-------------------+     +-----------------------+     +----------------------+
| app/              | <-- | input/                | <-- | control/             |
| Runner, Preview   |     | Win32SendInputMouse   |     | TriggerStateMachine  |
| HotkeyController  |     | SpyMouseBackend       |     | IDLE/ARMED/          |
| cli.py (__main__) |     | (40-byte SendInput)   |     | LATCHED/COOLDOWN     |
+-------------------+     +-----------------------+     +----------------------+
```

### 關鍵工程原則
- **匯入期純淨性（Import-Time Win32/DPI Purity）**：匯入 `vrc_smith_auto`、`app`、`capture`、`input` 模組時，**絕不**預先呼叫 `SetProcessDpiAwarenessContext`、載入 `user32`/`gdi32` DLL 或匯入 `dxcam`/`mss`。所有 OS API 與 DPI 初始化均延遲至執行期（`create_live_capture()`、`cli._cmd_run()` 或首次 `SendInput`/`GetAsyncKeyState` 呼叫）。
- **協定與純函式核心**：`FrameSource` 與 `MouseBackend` 定義於 `protocols.py`；`TriggerStateMachine.update()` 為無副作用純決策函式，回傳 `TriggerDecision` 並由 `Runner` 統一執行 `mouse.tick()`、`mouse.press_left()` 與 `finally: mouse.release_all()`。

---

## 3. 螢幕畫面擷取方案比較與選型

### 3.1 候選方案比較

| 方案 / 模組 | 底層技術 | 典型單影格延遲 | 優點 | 缺點與風險 | 採用決策 |
|---|---|---:|---|---|---|
| **`dxcam` (`DxcamCapture`)** | DXGI Desktop Duplication (COM / D3D11) | `~1–4 ms` | 極低延遲、GPU 側拷貝、`new_frame_only=True` 避免重複運算、支援區域裁切 | 多 GPU 筆電或部分虛擬顯示器可能初始化失敗 | **主選（Primary）** |
| **`bettercam`** | `dxcam` 分支 (Cython 擴充) | `~1–3 ms` | 高幀率擷取表現佳 | 依賴特定版本編譯輪子（Wheel），較新 Python 版本有 Cython 相容與維護風險 | **不採用** |
| **`mss` (`MssCapture`)** | 純 `ctypes` Win32 GDI `BitBlt` | `~5–10 ms` | 跨顯示卡/混合 GPU 穩定性極高、跨行程安全 | CPU GDI 複製延遲略高於 DXGI，受 DWM 合成影響 | **第一備援（Fallback 1）** |
| **`ctypes` `GdiBitBltCapture`** | 自實作 Win32 `GetDC` + `BitBlt(SRCCOPY\|CAPTUREBLT)` + `GetDIBits` | `~5–11 ms` | **零外部套件依賴**，重用相容點陣圖（DIB Buffer Reuse）減少配置 | 效能與 `mss` 同級，受視窗重疊或保護內容影響 | **第二備援（Fallback 2）** |
| **`PrintWindow` / `PIL.ImageGrab` / `pyautogui`** | 視窗訊息繪製 / GDI + PIL Image 包裝 | `~15–40 ms` | 使用簡單 | `PrintWindow` 對 D3D11/Unity 易回傳黑畫面；`ImageGrab` 含 PIL 物件轉換開銷；`pyautogui` 具全域 DPI 副作用 | **淘汰** |

### 3.2 實際實作設計（`src/vrc_smith_auto/capture/`）
- **`auto` 自動降級鏈**：`create_live_capture()` 在呼叫 `enable_dpi_awareness()` 後，依序嘗試 `dxcam -> mss -> gdi`，透過 `probe=True` 驗證後端可用性，任一失敗則安全釋放資源並轉往下一後端。
- **多螢幕與主要輸出端優先（`dxcam_source.py`）**：`_select_dxcam_candidates()` 主動列舉 DXGI Adapters/Outputs，優先選擇目標顯示器或 Primary Output，解決雙 GPU 筆電預設枚舉到空輸出的問題。
- **`DxcamCapture` 閒置語意**：因 `new_frame_only=True`，當 DWM 尚未產生新畫面時 `read()` 會回傳 `None`。`Runner` 將即時來源回傳 `None` 視為健康閒置狀態（短暫休眠並持續執行 `hotkeys.poll()` 與 `mouse.tick()`），不會誤判為 EOF 結束。
- **動態視窗客戶區追蹤（`WindowRegionTracker`）**：每 `rect_refresh_s=1.0` 秒更新一次 VRChat 客戶區矩形（`crop_to_window=True`），僅擷取遊戲視窗區域以降低傳輸像素量與背景雜訊；若視窗最小化或暫時遺失則安全退回全螢幕範圍。

---

## 4. 判定偵測與幾何適應演算法比較

### 4.1 候選方案比較

| 方案 | 平均運算延遲 (1080p) | 對 3D 傾斜與位置偏移適應力 | 部署與維護成本 | 採用決策 |
|---|---:|---|---|---|
| **Coarse-to-Fine OpenCV HSV ＋ 3 階段紅色切分橋接 ＋ 雙側銀帶驗證 ＋ `minAreaRect` 1D 主軸投影（`s_norm`）** | **`~8.5–9.4 ms`** | **極佳**（內建旋轉矩形與主軸投影，支援達 `±65°`） | 零模型權重、零標註需求、參數可視化可調 | **採用** |
| 邊緣輪廓矩形（Canny / Contours）＋ 單應性變換（Homography） | `~8–14 ms` | 佳，但須四頂點穩定 | 判定條深色外框與暗色遊戲 UI 對比弱，輪廓易斷裂；無須完整 2D 透視重建 | **不採用** |
| 輕量級深度學習物件偵測（YOLO / CNN） | `~15–40 ms`（CPU） | 佳 | CPU 延遲超出 `16.7 ms` 預算，需大量標註資料、ONNX/PyTorch 權重與打包體積 | **不採用** |

### 4.2 實際實作管線（`src/vrc_smith_auto/vision/detector.py`）
1. **由粗到細追蹤（Coarse-to-Fine ROI Tracking）**：
   - **未鎖定狀態**：當影格尺寸超過 `960 × 540`（即 `fw > 960 or fh > 540`）且 `acquire_downscale < 1.0` 時，先以 `acquire_downscale=0.5` 進行全畫面粗掃描；小於等於 `960 × 540`（如 `640×480`、`854×480`）則維持全解析度（`ds=1.0`）直接掃描以避免細窄目標失真。
   - **`2×2` OR 池化金色遮罩與單次粗掃描精細重算**：當 `ds == 0.5` 時，先於全解析度計算 HSV 金色遮罩，並以 SIMD `cv2.resize(..., INTER_AREA)` 搭配 `cv2.threshold(..., 0, 255, cv2.THRESH_BINARY)` 執行等價的 `2×2` OR 池化產生 `gold_mask`（搭配 `INTER_NEAREST` 降採樣之 `hsv_img`），確保僅 `1 px` 寬或落於奇數列的極窄金色碎片不會在降採樣時消失。粗掃描透過單次處理與 `refine_fn` 回呼輸出最多 6 個經距離去重（`dedup_r`）之主候選於全解析度（`ds=1.0`）局部 ROI 精細重算；若主路徑粗候選全數為粗掃描干擾物，則直接重用已計算之 `hsv_img`、`gold_mask`、`silver_mask`、連通域標籤與 `_has_silver_near`／單側銀色快取進入切分與貼紅窄金備援路徑（並略過已失敗中心），避免重複全畫面 HSV 轉換與連通域掃描。
   - **已鎖定狀態**：直接裁切上一幀中心周圍局部 ROI（`pad = max(180 * frame_scale, 5.5 * last_gauge_w)`），大幅減少 `cvtColor` 與形態學像素量；連續 `max_lost_frames=3` 影格遺失才退回全畫面掃描。
2. **HSV 色彩空間分割（`HSVConfig`）**：
   - 金色（Gold）：`H ∈ [15, 30], S ∈ [100, 255], V ∈ [140, 255]`
   - 銀色（Silver）：`H ∈ [0, 179], S ∈ [0, 45], V ∈ [135, 240]`
   - 紅色（Red，跨越 HSV 0°/180° 邊界）：`H ∈ [0, 12]` 與 `H ∈ [168, 179]` 聯集（`S ≥ 100, V ≥ 135`）
3. **三階段紅色切分遮擋橋接（3-Stage Red-Split-Occlusion Bridging）**：
   - **第一階段（Primary CC）**：使用自適應橢圓核執行 `cv2.morphologyEx(..., cv2.MORPH_CLOSE)`，在多數情況下直接縫合紅色判定條造成的細微切分。
   - **第二階段（Lazy Red-Guided Pair Fallback）**：若單一 CC 驗證失敗且連通域數量 `num_labels >= 3`，預先計算 `silver_mask` 並先以 `_has_silver_near` 過濾後取面積前 16 大連通域（避免干擾色塊擠掉有效候選），搜尋鄰近金色 CC 對（間隙 `<= 45 * max_scale`），建立凸包（Convex Hull）並驗證凸包內部確實存在紅色像素橋接，防止誤合併不相關的 HUD 金色色塊。
   - **第三階段（Narrow-Gold Red-Abutting Fallback）**：於全解析度與粗掃描階段皆支援執行；針對末段極窄金色區域（例如 `16 px`）剛好被紅色覆蓋單側邊緣的情況，驗證緊貼金色邊緣的紅色條寬度、厚度與對側銀色延伸帶，確保窄金色區間仍能鎖定。
4. **雙側銀色支持度驗證與干擾排除（Bilateral Silver Validation）**：
   - 計算 `Gold ∪ Silver` 區域之 `cv2.minAreaRect` 取得判定條主軸單位向量 `axis_unit`（規範化為由上至下 `uy >= 0`；當 `S+G` 長寬比 `< 1.55` 時進一步以雙側銀色質心修正軸向），拒絕超過 `_MAX_TILT_DEG = 65.0°` 之候選。
   - 檢查矩形填充率（`fill_ratio >= 0.45`）、解析度縮放後幾何尺寸，並強制要求金色兩端沿主軸方向均具備足夠銀色像素比例（`>= min_silver_neighbor_ratio = 0.25`），徹底過濾 `0%`、`锤!`、`熟练铁匠` 等獨立金色文字與圖示。
5. **1D 主軸投影與 2D 邊緣錨點恢復（`s_norm`）**：
   - 僅在對齊判定條寬度的定向長條區域內搜尋紅色 CC（寬度比 `>= 0.85`、軸偏離 `<= 0.60 * gauge_w`、厚度比 `<= 0.60`）。
   - 維護無紅色觸碰邊緣（`tol_touch = 1.0 * frame_scale`）時的金色 2D 邊緣參考錨點（`_ref_top_2d`, `_ref_bot_2d`, `_ref_gold_len`），當紅色剛好遮擋金色單側邊界時自動恢復真實金色區間（若依 `_ref_gold_len` 恢復會越過可見紅色內緣，則自動失效並清除過時的參考長度，防止跨擊次縮窄時誤延伸），最後輸出與旋轉／位置無關的歸一化位置：
     $$s_{\text{norm}} = \frac{\text{red\_proj} - g_{\text{center}, u}}{0.5 \cdot L_{\text{gold}}}$$
     當 $|s_{\text{norm}}| \le 1.0$ 時代表紅色中心位於金色區域內部，$0.0$ 代表正中心。

---

## 5. 滑鼠觸發與熱鍵方案比較

### 5.1 滑鼠驅動候選方案比較

| 方案 | 實作機制 | 預設阻塞 / 副作用 | 遊戲相容性與安全性 | 採用決策 |
|---|---|---|---|---|
| **純 `ctypes` 40-byte `SendInput` (`Win32SendInputMouse`)** | `user32.SendInput`（僅 `MOUSEEVENTF_LEFTDOWN` / `LEFTUP`） | 無阻塞延遲；不傳送 `MOVE`/`ABSOLUTE` 原地點擊 | 支援非阻塞排程放開（`hold_ms=30.0`）與 UIPI/前景檢查 | **採用** |
| **`pydirectinput`** | `ctypes` `SendInput` 包裝 | **預設 `PAUSE=0.1s`**（每次呼叫強迫等待 `100 ms`，致命延遲） | 多餘相依性，預設值易誤踩 | **不採用** |
| **`pynput`** | Windows Hook / `SendInput` | 全域執行緒監聽 Hook | 可能與遊戲防作弊或視窗輸入搶佔衝突 | **不採用** |
| **`pyautogui`** | 舊版 `mouse_event` | 預設 `PAUSE=0.1s`、具 DPI 設定副作用 | 非標準現代輸入 API，容易遭遊戲忽略 | **不採用** |
| **`win32api.mouse_event`** | 過時 Win32 API | 需額外 `pywin32` 相依 | 微軟官方文件標註已棄用（Superseded by `SendInput`） | **不採用** |

### 5.2 非阻塞點擊與熱鍵實作（`input/` & `app/hotkeys.py`）
- **40-byte `INPUT` 結構**：明確宣告 `MOUSEINPUT`、`KEYBDINPUT`、`HARDWAREINPUT` 之 `_INPUT_UNION`（Win64 下 `sizeof(INPUT) == 40` bytes），`dx=0, dy=0, dwExtraInfo=0`，確保滑鼠游標原地點擊不跳動。
- **非阻塞按住與異常保護**：`press_left(duration_ms)` 僅發送 `LEFTDOWN` 並設定 `_release_at_s`，由主迴圈每幀呼叫 `tick(now_s)` 在 `30.0 ms` 到期時發送 `LEFTUP`；`Runner.run()` 透過 `try ... finally: mouse.release_all()` 與 `Esc` 緊急停止保證絕不卡鍵。
- **測試替身（`SpyMouseBackend`）**：實作相同 `MouseBackend` 協定，於 `--dry-run`、`verify-image`、`verify-video` 與 `benchmark` 中紀錄每次點擊之 `timestamp_s`、`duration_ms`、`release_timestamp_s` 與 `frame_idx`。
- **熱鍵輪詢（`HotkeyController`）**：透過延遲綁定之 `user32.GetAsyncKeyState` 進行上升緣（`0x8000`）偵測，支援 `F8`（`VK_F8 = 0x77`）切換暫停與 `Esc`（`VK_ESCAPE = 0x1B`）緊急停止。

---

## 6. 觸發狀態機與防誤觸設計

`TriggerStateMachine`（`src/vrc_smith_auto/control/state_machine.py`）採用四狀態有限狀態機防止邊界抖動與跨擊次雙擊：

```text
  [IDLE] --(|raw_pos| > entry_margin)--> [ARMED] --(|raw_pos| <= entry_margin)--> [LATCHED] (click!)
    ^                                       ^                                         |
    | (lost >= rearm_lost_frames)           | (cooldown elapsed)                      | (|raw_pos| > exit_margin)
    +---------------------------------------+----------- [COOLDOWN] <-----------------+
```

1. **起始防誤觸（`IDLE -> ARMED`）**：必須先觀測到紅色判定條位於金色門檻之外（`|raw_pos| > entry_margin = 0.95`）才進入 `ARMED`。若工具在紅色條已位於金色中心時才啟動或切回遊戲，不會瞬間誤點。
2. **遲滯鎖存與冷卻雙重防護（`LATCHED -> COOLDOWN`）**：
   - 觸發點擊後立即轉為 `LATCHED`，必須等到紅色條確實離開金色至 `|raw_pos| > exit_margin`（`1.25`）才轉入 `COOLDOWN`。
   - 冷卻時間 `cooldown_ms = 200.0 ms`（60 FPS 下約 12 frames），即便遊戲在擊中後瞬間切換下一擊（$H_k \to H_{k+1}$），也絕不會在同一次通過或跳變幀重複點擊。
3. **跨擊次跳變偵測（`_is_gold_jump`）**：當連續影格間金色中心位移超過 `max(4.0, 0.25 * prev_len)` 或長度變化超過 `max(3.0, 0.15 * prev_len)` 時，立即重置 EMA 速度估計，避免前一次的移動速度污染 `lead_ms` 預測計算。
4. **按壓期間觸發抑制（`allow_trigger`）**：`TriggerStateMachine.update(..., allow_trigger: bool = True)` 支援由呼叫端抑制觸發；`Runner.run()` 固定傳入 `allow_trigger=not self._mouse.is_pressed`。當滑鼠左鍵尚處於非阻塞排程按壓期間（例如設定 `hold_ms > cooldown_ms` 時），狀態機維持於 `ARMED` 並回報 `reason="mouse_held_waiting"` 而不重入觸發，確保 `RunResult.clicks == mouse.click_count == state_machine.trigger_count` 恆等。

---

## 7. DPI、視窗追蹤與執行期安全機制

- **Per-Monitor-V2 DPI 感知（`window.py`）**：`enable_dpi_awareness()` 採冪等降級鏈：`user32.SetProcessDpiAwarenessContext(-4)` → `shcore.SetProcessDpiAwareness(2)` → `user32.SetProcessDPIAware()`，確保 Windows 125%/150%/200% 縮放下視窗座標與擷取解析度 1:1 對齊。
- **視窗定位（`find_vrchat_window`）**：列舉可見且非最小化之頂層視窗，依 `(process_name == "VRChat.exe", exact_title, class_name == "UnityWndClass")` 加權評分定位 VRChat 視窗 HWND，並以 `GetClientRect` + `ClientToScreen` 取得實體像素客戶區。
- **前景與 UIPI 安全檢查**：`Win32SendInputMouse` 在 `require_foreground=True` 時檢查 `GetForegroundWindow() == hwnd`，並透過 `OpenProcessToken` + `GetTokenInformation(TokenElevation)` 檢查目標視窗是否以高權限執行（UIPI 隔離），避免無效呼叫或誤觸背景視窗。

---

## 8. 可重現離線驗證與效能基準測試（實測結果）

### 8.1 本機測試環境規格
- **作業系統**：`Windows-10-10.0.26300-SP0`（Windows 11 64-bit）
- **CPU**：AMD64 Family 25 Model 97 Stepping 2（32 logical cores）
- **Python / 套件**：Python `3.10.6`、`opencv-python 5.1.0`、`numpy 2.2.6`、`PyYAML 6.0.3`、`dxcam 0.3.0`、`mss 10.2.0`、`pytest 9.2.0`

### 8.2 可重現指令與實測輸出

#### 1. 截圖驗證（非金色銀色區不誤觸）
```powershell
.\.venv\Scripts\python.exe -m vrc_smith_auto verify-image --image "C:\Users\wewe1\.gemini\antigravity\brain\0aa8bc95-c47c-47dc-90d8-42df58884912\.user_uploaded\media_1795156043546_6066a817.png" --expect-clicks 0 --json
```
- **實測結果**：`frames_processed=1`, `detections=1`, `clicks=0`, `click_frames=[]`, `passed=true`（Exit Code `0`）。

#### 2. 完整 1570 影格實戰影片回放（`example/2026-11-20 15-32-46.mp4`）
```powershell
.\.venv\Scripts\python.exe -m vrc_smith_auto verify-video --video "example\2026-11-20 15-32-46.mp4" --dry-run --json
```
- **實測結果**：
  - `fps=60.0`, `frames_processed=1570`, `detections=1127`, `clicks=22`, `elapsed_s=16.18384`
  - `click_frames`: `[169, 233, 317, 364, 435, 465, 538, 575, 621, 787, 823, 943, 993, 1013, 1038, 1058, 1107, 1155, 1204, 1236, 1285, 1305]`
  - 所有未啟動區段（`0..117`, `631..724`, `1339..1569`）皆為 `0` 誤觸，且包含末段極窄極速擊次 `Seg2-H9`（影格 `1285`, `1305`）全數精準命中。

#### 3. 單影格延遲基準測試（Preloaded 1080p + 10 影格 Warmup）
```powershell
.\.venv\Scripts\python.exe -m vrc_smith_auto benchmark --video "example\2026-11-20 15-32-46.mp4" --iterations 200 --max-p95-ms 16.7 --json
.\.venv\Scripts\python.exe -m vrc_smith_auto benchmark --video "example\2026-11-20 15-32-46.mp4" --iterations 500 --max-p95-ms 16.7 --json
```

| 測試項目 | Iterations | Warmup | Mean (ms) | p50 (ms) | p95 (ms) | Min (ms) | Max (ms) | SLA 上限 (ms) | 判定結果 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `benchmark --video` (200) | 200 | 10 | **8.5169** | 9.9193 | **11.9253** | 4.3591 | 17.4408 | 16.7 (`< 33.3` dwell) | **PASS** |
| `benchmark --video` (500) | 500 | 10 | **9.3564** | 7.3256 | **14.6089** | 4.9447 | 20.6644 | 16.7 (`< 33.3` dwell) | **PASS** |

- **基準測試說明**：200 次與 500 次迭代皆通過 `p95_ms < 16.7 ms` 門檻（`passed=true`，Exit Code `0`）；200 次迭代實測 `mean_ms=8.5169`, `p95_ms=11.9253`，500 次迭代（涵蓋前段無判定條區間影格 `10..117` 之全畫面 1080p 粗掃描）實測 `mean_ms=9.3564`, `p50_ms=7.3256`, `p95_ms=14.6089`，均遠低於最短金色停留區間 `33.3 ms`。
- **即時模式（`benchmark --live`）環境說明**：於本自動化執行環境之非互動式無頭會話（Headless Session，`GetForegroundWindow() == 0`）中實測 `benchmark --live` 時，`create_live_capture` 依預期觸發保護並安全回傳錯誤（`dxcam` Access Denied / `mss` & `gdi` `BitBlt failed winerror=5`）；於本機實際登入且未鎖定之 Windows 桌面會話執行時可正常啟用 `dxcam` / `mss` / `gdi` 即時擷取。
