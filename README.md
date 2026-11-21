# VRChat《爐火戰團》鐵匠小遊戲自動工具

適用於 VRChat 桌面視窗模式。工具會偵測紅色判定條進入金色區域時，自動在原地送出滑鼠左鍵點擊。

## 環境需求

- Windows 10 / Windows 11（64 位元）
- Python 3.10 或更新版本
- VRChat 桌面模式視窗

## 安裝

建議使用 `uv` 建立虛擬環境：

```powershell
uv venv --python 3.10 .venv
uv pip install --python .venv\Scripts\python.exe -e ".[all]"
```

或使用標準 `pip`：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

## 快速開始

1. 啟動 VRChat 並進入桌面模式的鐵匠小遊戲。
2. 先執行乾跑模式與即時預覽，確認偵測正常：

```powershell
.\.venv\Scripts\python.exe -m vrc_smith_auto run --dry-run --preview
```

此模式不會送出實際滑鼠點擊。在預覽視窗按 `q` 或關閉視窗即可退出。

3. 確認偵測正常後執行：

```powershell
.\.venv\Scripts\vrc-smith-auto.exe run
```

4. 將滑鼠游標停在畫面上的點擊區域（例如「锤!」按鈕上方）。工具偵測到金色區域且 VRChat 為前景視窗時，會在原地點擊。

## 熱鍵

| 熱鍵 | 功能 |
|---|---|
| `F8` | 暫停或恢復自動點擊 |
| `Esc` | 緊急停止並釋放滑鼠左鍵 |

## 指令

以下指令皆可使用模組方式執行：

```powershell
.\.venv\Scripts\python.exe -m vrc_smith_auto <指令> [參數]
```

也可使用已安裝的執行檔：

```powershell
.\.venv\Scripts\vrc-smith-auto.exe <指令> [參數]
```

### 即時執行

```powershell
.\.venv\Scripts\python.exe -m vrc_smith_auto run [--config PATH] [--backend auto|dxcam|mss|gdi] [--dry-run] [--preview] [--no-focus-guard]
```

- `--config PATH`：載入自訂 YAML 或 JSON 設定檔。
- `--backend`：選擇螢幕擷取方式，預設為 `auto`。
- `--dry-run`：只偵測，不送出滑鼠點擊。
- `--preview`：開啟即時偵測預覽視窗。
- `--no-focus-guard`：允許 VRChat 非前景時執行，通常僅供離線測試。

## 設定檔

建立 `config.yaml`，再透過 `--config` 載入：

```yaml
capture:
  backend: auto
  target_fps: 60
  monitor_idx: 0
  crop_to_window: true

window:
  title: VRChat
  class_name: UnityWndClass
  require_foreground: true
  rect_refresh_s: 1.0

trigger:
  hold_ms: 30.0
  cooldown_ms: 200.0
  entry_margin: 0.95
  exit_margin: 1.25
  lead_ms: 0.0
  max_lead_offset: 0.5
  rearm_lost_frames: 3
  dry_run: false
```

載入方式：

```powershell
.\.venv\Scripts\python.exe -m vrc_smith_auto run --config config.yaml
```
