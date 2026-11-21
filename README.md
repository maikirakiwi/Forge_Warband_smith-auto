# VRChat《爐火戰團》鐵匠小遊戲自動工具（`vrc-smith-auto`）

> 適用於 **VRChat 桌面視窗模式（Desktop Mode）** 的低延遲視覺判定與原地滑鼠自動點擊工具：當紅色判定條進入金色（「完美」）區域時自動送出滑鼠左鍵點擊，忽略黑（揮空）、灰（打偏）、銀（不錯）區域與無判定條畫面。

詳細架構選型、演算法設計、替代方案比較與本機實測數據，請參閱技術設計文件：[docs/TECHNICAL_DESIGN.md](docs/TECHNICAL_DESIGN.md)。

---

## 1. 功能範圍與設計原則

- **單一任務聚焦**：僅負責偵測鐵匠小遊戲判定條並於金色區域自動觸發滑鼠左鍵原地點擊，不包含移動角色、接取任務、領獎等額外遊戲流程自動化。
- **不移動游標（In-Place Click）**：透過純 `ctypes` 40-byte Win32 `SendInput` 送出不帶移動旗標（無 `MOUSEEVENTF_MOVE` / `MOUSEEVENTF_ABSOLUTE`）的左鍵按下與放開事件，滑鼠只需預先停在遊戲點擊按鈕上即可。
- **視角與位置適應**：採用 Per-Monitor-V2 DPI 感知、VRChat 客戶區（Client Rect）追蹤、Coarse-to-Fine HSV 偵測、三階段紅色切分橋接與主軸歸一化座標（`s_norm`），可容忍 3D 攝影機傾斜與判定條位移。
- **安全防護機制**：內建前景視窗保護（Focus Guard）、UIPI 提權偵測、`F8` 暫停/恢復切換、`Esc` 緊急停止並強制釋放左鍵，以及完全不觸發系統滑鼠的 `--dry-run` 驗證模式。

---

## 2. 環境需求與安裝（`uv` / `.venv`）

- **作業系統**：Windows 10 / Windows 11（64-bit）
- **Python 版本**：Python `>= 3.10`（本機驗證環境為 Python `3.10.6`）
- **核心依賴**：`numpy>=1.26`、`opencv-python>=4.8`、`PyYAML>=6.0`、`dxcam>=0.0.5`、`mss>=9.0`、`pytest>=8.0`

### 使用 `uv` 建立虛擬環境與安裝（建議）

```powershell
# 1. 建立 Python 3.10+ 虛擬環境
uv venv --python 3.10 .venv

# 2. 安裝專案（可編輯模式）與所有選配依賴（dxcam, mss, pytest 等）
uv pip install --python .venv\Scripts\python.exe -e ".[all]"
```

### 或使用標準 `pip` 安裝

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

> **提示**：若不想啟動虛擬環境，所有指令皆可直接透過 `.\.venv\Scripts\python.exe -m vrc_smith_auto ...` 或 `.\.venv\Scripts\vrc-smith-auto.exe ...` 執行。

---

## 3. 快速開始與 VRChat 桌面遊玩操作

1. **啟動 VRChat 桌面模式**：建議使用「視窗化（Windowed）」或「無邊框視窗（Borderless Windowed）」模式，確保畫面顯示在主要螢幕上。
2. **先以 `--dry-run --preview` 測試偵測**：
   ```powershell
   .\.venv\Scripts\python.exe -m vrc_smith_auto run --dry-run --preview
   ```
   此時會開啟即時視覺除錯視窗（顯示金色框、紅色條、主軸向量、`norm_pos`、`tilt_deg` 與狀態機狀態），**不會實際送出任何系統左鍵點擊**。在預覽視窗按 `q` 或關閉視窗即可退出。
3. **開始實機遊玩**：
   ```powershell
   .\.venv\Scripts\vrc-smith-auto.exe run
   ```
   - 進入鐵匠小遊戲後，將滑鼠游標停留在畫面上的點擊區域（例如「锤!」按鈕上方）。
   - 工具會在 VRChat 視窗處於前景（Focus）時自動於金色區域觸發原地左鍵點擊。

### 全域熱鍵控制（Hotkeys）

| 熱鍵 | 行為說明 |
|---|---|
| **`F8`** | 切換 **暫停（PAUSED）／恢復（ARMED）**。進入暫停時會立即釋放滑鼠左鍵（`release_all()`）並重置狀態機，避免恢復時誤觸。 |
| **`Esc`** | **緊急停止（Emergency Stop）**。立即呼叫 `release_all()` 確保滑鼠左鍵不卡住，並安全結束主迴圈。 |

---

## 4. CLI 指令參考

可透過 `.\.venv\Scripts\python.exe -m vrc_smith_auto` 或已安裝之 `.\.venv\Scripts\vrc-smith-auto.exe` 執行以下 4 個子指令：

### 4.1 `run` — 即時螢幕擷取與自動點擊

```powershell
.\.venv\Scripts\python.exe -m vrc_smith_auto run [--config PATH] [--backend auto|dxcam|mss|gdi] [--dry-run] [--preview] [--no-focus-guard]
```

- `--config PATH`：載入自訂 YAML 或 JSON 設定檔。
- `--backend {auto,dxcam,mss,gdi}`：指定螢幕擷取後端（預設 `auto`，依序嘗試 `dxcam -> mss -> gdi`）。
- `--dry-run`：乾跑模式（使用 `SpyMouseBackend`，不呼叫 Win32 `SendInput`）。
- `--preview`：開啟 OpenCV 即時疊加除錯視窗（按 `q` 可關閉）。
- `--no-focus-guard`：關閉前景視窗檢查（允許在目標視窗非前景時觸發，僅建議用於離線視窗測試）。

### 4.2 `verify-image` — 單張靜態截圖離線驗證

```powershell
.\.venv\Scripts\python.exe -m vrc_smith_auto verify-image --image <PATH> [--expect-clicks 0] [--json]
```

- 對單張截圖執行偵測與狀態機判定，比對實際點擊次數是否符合 `--expect-clicks`（預設 `0`），不符合時回傳非零結束碼（Exit Code）。

### 4.3 `verify-video` — 影片完整離線回放驗證

```powershell
.\.venv\Scripts\python.exe -m vrc_smith_auto verify-video --video <PATH> [--dry-run] [--json]
```

- 以確定性（Deterministic）逐影格時間戳記回放 MP4/AVI 影片，輸出總影格數、偵測數、觸發點擊次數與 `click_frames` 影格編號清單。

### 4.4 `benchmark` — 單影格延遲基準測試與 SLA 檢查

```powershell
.\.venv\Scripts\python.exe -m vrc_smith_auto benchmark [--video <PATH> | --live] [--iterations 200] [--max-p95-ms 16.7] [--json]
```

- `--video PATH`：預先將影片影格載入記憶體（避免磁碟 H.264 解碼 I/O 干擾演算法計時），執行 10 影格暖機（Warmup）後量測 `--iterations` 次的 `GaugeDetector + TriggerStateMachine` 處理延遲。
- `--live`：對真實桌面進行即時擷取＋偵測＋狀態機延遲量測（需在已解鎖之互動式 Windows 桌面會話執行）。
- `--max-p95-ms`：預設 `16.7` ms（60 FPS 單影格時間，低於參考影片最短金色停留區間 `33.3 ms`）。若實測 `p95_ms > max_p95_ms`，程式會先輸出 JSON 報告（含 `p95_ms` 與 `passed: false`），再回傳非零結束碼。

---

## 5. 設定檔說明（`AppConfig` YAML / JSON）

可建立 `config.yaml` 並透過 `run --config config.yaml` 載入，未指定的欄位會自動採用預設值：

```yaml
capture:
  backend: auto          # auto | dxcam | mss | gdi
  target_fps: 60
  monitor_idx: 0
  crop_to_window: true   # 自動追蹤 VRChat 客戶區範圍以減少擷取像素

window:
  title: VRChat
  class_name: UnityWndClass
  require_foreground: true
  rect_refresh_s: 1.0

trigger:
  hold_ms: 30.0          # 左鍵按住時間 (ms)，非阻塞排程釋放
  cooldown_ms: 200.0     # 雙擊防抖冷卻時間 (ms)
  entry_margin: 0.95     # 進入觸發門檻 (|s_norm| <= 0.95)
  exit_margin: 1.25      # 離開遲滯門檻 (|s_norm| > 1.25 才解除鎖存)
  lead_ms: 0.0           # 可選速度超前預測補償 (ms)
  max_lead_offset: 0.5
  rearm_lost_frames: 3
  dry_run: false
```

---

## 6. 可重現驗證與本機實測結果

以下指令皆於本機 Windows 11（64-bit, Python `3.10.6`, OpenCV `5.1.0`, NumPy `2.2.6`）實測通過：

### 6.1 單元測試套件

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit -v
```

### 6.2 單張遊戲截圖驗證（非金色銀色區 → 0 誤觸）

```powershell
.\.venv\Scripts\python.exe -m vrc_smith_auto verify-image --image "C:\Users\wewe1\.gemini\antigravity\brain\0aa8bc95-c47c-47dc-90d8-42df58884912\.user_uploaded\media_1795156043546_6066a817.png" --expect-clicks 0 --json
```

實測輸出：
```json
{"image": "C:\\Users\\wewe1\\.gemini\\antigravity\\brain\\0aa8bc95-c47c-47dc-90d8-42df58884912\\.user_uploaded\\media_1795156043546_6066a817.png", "frames_processed": 1, "detections": 1, "clicks": 0, "click_frames": [], "expected_clicks": 0, "passed": true}
```

### 6.3 完整遊玩影片驗證（`1570` 影格 @ 1080p 60fps）

```powershell
.\.venv\Scripts\python.exe -m vrc_smith_auto verify-video --video "example\2026-11-20 15-32-46.mp4" --dry-run --json
```

實測輸出：
```json
{"video": "example\\2026-11-20 15-32-46.mp4", "fps": 60.0, "frames_processed": 1570, "detections": 1127, "clicks": 22, "click_frames": [169, 233, 317, 364, 435, 465, 538, 575, 621, 787, 823, 943, 993, 1013, 1038, 1058, 1107, 1155, 1204, 1236, 1285, 1305], "elapsed_s": 16.18384}
```

### 6.4 單影格延遲基準測試（SLA: `p95 < 16.7 ms` < 最短金色停留 `33.3 ms`）

```powershell
.\.venv\Scripts\python.exe -m vrc_smith_auto benchmark --video "example\2026-11-20 15-32-46.mp4" --iterations 200 --max-p95-ms 16.7 --json
```

實測輸出（200 次迭代）：
```json
{"mode": "video", "iterations": 200, "warmup": 10, "mean_ms": 8.5169, "p50_ms": 9.9193, "p95_ms": 11.9253, "min_ms": 4.3591, "max_ms": 17.4408, "max_p95_ms": 16.7, "passed": true}
```

500 次迭代指令與實測輸出：
```powershell
.\.venv\Scripts\python.exe -m vrc_smith_auto benchmark --video "example\2026-11-20 15-32-46.mp4" --iterations 500 --max-p95-ms 16.7 --json
```

```json
{"mode": "video", "iterations": 500, "warmup": 10, "mean_ms": 9.3564, "p50_ms": 7.3256, "p95_ms": 14.6089, "min_ms": 4.9447, "max_ms": 20.6644, "max_p95_ms": 16.7, "passed": true}
```

> **說明**：200 次與 500 次迭代均通過嚴格 60 FPS 門檻 `p95_ms < 16.7 ms`（`passed: true`，Exit Code `0`），平均延遲 `8.52–9.36 ms`、`p95_ms` `11.93–14.61 ms` 亦遠低於影片最短金色停留區間 `33.3 ms`。

---

## 7. 疑難排解（Troubleshooting）

1. **高 DPI 螢幕縮放導致裁切偏移**：
   - 工具在啟動即時模式時會自動呼叫 `SetProcessDpiAwarenessContext(-4)`（Per-Monitor-V2 DPI Awareness），使 `GetClientRect` / `ClientToScreen` 回傳與螢幕擷取一致的實體像素座標；如使用多螢幕且副螢幕有不同縮放比例，請確認 `capture.monitor_idx` 對應正確螢幕，或改以全視窗未跨螢幕方式執行 VRChat。
2. **VRChat 無反應或出現 UIPI 權限錯誤（`ERROR_ACCESS_DENIED (5)`）**：
   - Windows UIPI（User Interface Privilege Isolation）禁止低權限行程向系統管理員權限視窗送出 `SendInput`。若 VRChat 以系統管理員身分啟動，請改以一般使用者權限執行 VRChat，或以系統管理員身分開啟終端機執行本工具。
3. **DXGI 初始化失敗、多 GPU 筆電或 Virtual Desktop 黑畫面**：
   - 雙顯卡筆電（iGPU + dGPU）或特定虛擬顯示器可能導致 `dxcam` 初始化失敗或回傳黑畫面。雖然 `auto` 模式會自動降級，也可直接指定 GDI 後端：
     ```powershell
     .\.venv\Scripts\python.exe -m vrc_smith_auto run --backend mss
     # 或零外部依賴 Win32 GDI 後端
     .\.venv\Scripts\python.exe -m vrc_smith_auto run --backend gdi
     ```
4. **遊戲未觸發點擊（Focus Guard 阻擋）**：
   - 預設 `window.require_foreground: true`，只有在找到標題含 `VRChat`（`UnityWndClass`）且該視窗位於最前景時才會送出滑鼠左鍵，避免誤點桌面其他程式。若於測試環境對非前景視窗進行測試，可加上 `--no-focus-guard`。
5. **在無頭終端機 / 遠端鎖定桌面執行即時擷取失敗**：
   - 在無互動桌面會話（例如背景排程或鎖定螢幕，`GetForegroundWindow() == 0`）下，Windows 會拒絕存取桌面畫面（DXGI Access Denied / `BitBlt` WinError 5）。即時模式與 `benchmark --live` 必須在已登入且解鎖的本機桌面會話中執行；離線驗證請使用 `verify-image`、`verify-video` 與 `benchmark --video`。
