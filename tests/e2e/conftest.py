from __future__ import annotations

import importlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
E2E_DIR = Path(__file__).resolve().parent
if str(E2E_DIR) not in sys.path:
    sys.path.insert(0, str(E2E_DIR))
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"
REQUIREMENTS_PATH = REPO_ROOT / "requirements.txt"
README_PATH = REPO_ROOT / "README.md"
TECH_DESIGN_PATH = REPO_ROOT / "docs" / "TECHNICAL_DESIGN.md"
VENV_PYTHON_PATH = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
REFERENCE_SCREENSHOT_PATH = Path(
    r"C:/Users/wewe1/.gemini/antigravity/brain/0aa8bc95-c47c-47dc-90d8-42df58884912/.user_uploaded/media_1795156043546_6066a817.png"
)
REFERENCE_VIDEO_PATH = REPO_ROOT / "example" / "2026-11-20 15-32-46.mp4"


def ensure_src_on_path() -> Path:
    """Prepend src/ and tests/e2e/ to sys.path when present."""
    if SRC_ROOT.exists():
        src_str = str(SRC_ROOT)
        if src_str not in sys.path:
            sys.path.insert(0, src_str)
    e2e_str = str(E2E_DIR)
    if e2e_str not in sys.path:
        sys.path.insert(0, e2e_str)
    return SRC_ROOT


def import_pkg(module_name: str) -> Any:
    """Deferred package/module importer that keeps pytest collection safe."""
    ensure_src_on_path()
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        pytest.fail(f"Missing or unimportable module '{module_name}': {exc}")


def import_attr(module_name: str, attr_name: str) -> Any:
    """Deferred attribute importer with clear failure diagnostics."""
    mod = import_pkg(module_name)
    if not hasattr(mod, attr_name):
        pytest.fail(f"Module '{module_name}' does not export '{attr_name}'")
    return getattr(mod, attr_name)


def import_optional_attr(module_names: tuple[str, ...], attr_names: tuple[str, ...]) -> Any:
    """Resolve the first available public symbol from candidate modules/names."""
    ensure_src_on_path()
    last_err: Exception | None = None
    for mod_name in module_names:
        try:
            mod = importlib.import_module(mod_name)
        except ImportError as exc:
            last_err = exc
            continue
        for attr in attr_names:
            if hasattr(mod, attr):
                return getattr(mod, attr)
    pytest.fail(
        f"None of {attr_names} found in modules {module_names}"
        + (f" (last import error: {last_err})" if last_err else "")
    )


def import_cv2_np() -> tuple[Any, Any]:
    """Deferred import of OpenCV and NumPy inside test/fixture execution."""
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        pytest.fail(f"OpenCV/NumPy unavailable in interpreter '{sys.executable}': {exc}")
    return cv2, np


def import_yaml() -> Any:
    """Deferred import of PyYAML."""
    try:
        import yaml
    except ImportError as exc:
        pytest.fail(f"PyYAML unavailable in interpreter '{sys.executable}': {exc}")
    return yaml


def build_subprocess_env(extra_env: dict[str, str] | None = None) -> dict[str, str]:
    """Build UTF-8 safe environment with src/ prepended to PYTHONPATH."""
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    src_str = str(SRC_ROOT)
    if existing:
        if src_str not in existing.split(os.pathsep):
            env["PYTHONPATH"] = f"{src_str}{os.pathsep}{existing}"
    else:
        env["PYTHONPATH"] = src_str
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    if extra_env:
        env.update(extra_env)
    return env


def run_cli(
    args: list[str],
    timeout: float = 120.0,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Invoke `python -m vrc_smith_auto <args>` via current test interpreter."""
    cmd = [sys.executable, "-m", "vrc_smith_auto", *args]
    return subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=build_subprocess_env(extra_env),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def parse_cli_json(proc: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    """Parse JSON object from CLI stdout (tolerating preceding/trailing log lines)."""
    raw = (proc.stdout or "").strip()
    assert raw, (
        f"Expected JSON on stdout, got empty stdout "
        f"(returncode={proc.returncode}, stderr={proc.stderr!r})"
    )
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    candidates: list[dict[str, Any]] = []
    pos = 0
    while pos < len(raw):
        idx = raw.find("{", pos)
        if idx < 0:
            break
        try:
            obj, end = decoder.raw_decode(raw, idx)
        except json.JSONDecodeError:
            pos = idx + 1
            continue
        if isinstance(obj, dict):
            candidates.append(obj)
        pos = end

    if candidates:
        return candidates[-1]

    pytest.fail(
        f"Failed to parse JSON object from stdout: {raw!r} (stderr={proc.stderr!r})"
    )


def extract_cli_clicks(data: dict[str, Any]) -> tuple[int, list[int]]:
    """Extract total click count and click frame indices from CLI JSON dict."""
    count: int | None = None
    frames: list[int] = []

    def _is_int(v: Any) -> bool:
        return isinstance(v, int) and not isinstance(v, bool)

    for key in ("click_frames", "trigger_frames", "frames"):
        val = data.get(key)
        if isinstance(val, list):
            if all(_is_int(x) for x in val):
                frames = list(val)
                break
            if all(isinstance(x, dict) and "frame_idx" in x and _is_int(x["frame_idx"]) for x in val):
                frames = [int(x["frame_idx"]) for x in val]
                break

    if not frames and isinstance(data.get("events"), list):
        ev_frames = []
        for ev in data["events"]:
            if not isinstance(ev, dict) or "frame_idx" not in ev:
                continue
            ev_type = None
            for type_key in ("type", "event", "kind"):
                if type_key in ev and ev[type_key] is not None:
                    ev_type = ev[type_key]
                    break
            if ev_type is not None:
                lowered = str(ev_type).lower()
                if not any(tok in lowered for tok in ("click", "trigger", "press")):
                    continue
            if _is_int(ev["frame_idx"]):
                ev_frames.append(int(ev["frame_idx"]))
        if ev_frames:
            frames = ev_frames

    for key in ("clicks", "click_count", "total_clicks", "triggers"):
        val = data.get(key)
        if _is_int(val):
            count = val
            break
        if isinstance(val, list):
            count = len(val)
            if not frames and all(_is_int(x) for x in val):
                frames = list(val)
            elif not frames and all(
                isinstance(x, dict) and "frame_idx" in x and _is_int(x["frame_idx"]) for x in val
            ):
                frames = [int(x["frame_idx"]) for x in val]
            break

    if count is None:
        count = len(frames)
    return count, frames


def make_app_config(**overrides: Any) -> Any:
    """Construct AppConfig using public AppConfig().with_overrides(**overrides)."""
    AppConfig = import_attr("vrc_smith_auto.config", "AppConfig")
    cfg = AppConfig()
    if not overrides:
        return cfg
    if hasattr(cfg, "with_overrides"):
        return cfg.with_overrides(**overrides)
    if hasattr(AppConfig, "from_dict"):
        return AppConfig.from_dict(overrides)
    sig = inspect.signature(AppConfig)
    filtered = {k: v for k, v in overrides.items() if k in sig.parameters}
    return AppConfig(**filtered)


def load_app_config_from_file(path: Path) -> Any:
    """Load AppConfig from a YAML or JSON file via public contract."""
    cfg_mod = import_pkg("vrc_smith_auto.config")
    AppConfig = getattr(cfg_mod, "AppConfig")
    if hasattr(AppConfig, "from_file"):
        return AppConfig.from_file(path)
    if hasattr(cfg_mod, "load_config"):
        return cfg_mod.load_config(path)
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml") and hasattr(AppConfig, "from_yaml"):
        return AppConfig.from_yaml(path)
    if suffix == ".json" and hasattr(AppConfig, "from_json"):
        return AppConfig.from_json(path)
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw) if suffix == ".json" else import_yaml().safe_load(raw)
    return AppConfig.from_dict(data)


def make_spy_mouse(**kwargs: Any) -> Any:
    """Instantiate SpyMouseBackend cleanly via public signature inspection."""
    SpyMouseBackend = import_optional_attr(
        ("vrc_smith_auto.input.spy_mouse", "vrc_smith_auto.input"),
        ("SpyMouseBackend",),
    )
    sig = inspect.signature(SpyMouseBackend)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return SpyMouseBackend(**filtered)


def get_spy_click_events(spy: Any) -> list[Any]:
    """Return recorded click events from SpyMouseBackend public attributes."""
    for attr in ("clicks", "events", "presses", "click_events", "history", "timestamps"):
        val = getattr(spy, attr, None)
        if val is not None:
            return list(val)
    count = getattr(spy, "click_count", None)
    if isinstance(count, int):
        return [None] * count
    pytest.fail("SpyMouseBackend exposes no recognized click history or click_count attribute")


def get_spy_click_count(spy: Any) -> int:
    """Return total click count recorded by SpyMouseBackend."""
    count = getattr(spy, "click_count", None)
    if isinstance(count, int) and not isinstance(count, bool):
        return count
    return len(get_spy_click_events(spy))


def make_gauge_detector(config: Any | None = None) -> Any:
    """Instantiate GaugeDetector with optional AppConfig."""
    GaugeDetector = import_optional_attr(
        ("vrc_smith_auto.vision.detector", "vrc_smith_auto.vision"),
        ("GaugeDetector",),
    )
    if config is None:
        return GaugeDetector()
    sig = inspect.signature(GaugeDetector)
    params = sig.parameters
    if "config" in params:
        return GaugeDetector(config=config)
    if "app_config" in params:
        return GaugeDetector(app_config=config)
    kwargs: dict[str, Any] = {}
    if "hsv" in params and hasattr(config, "hsv"):
        kwargs["hsv"] = config.hsv
    if "geometry" in params and hasattr(config, "geometry"):
        kwargs["geometry"] = config.geometry
    if kwargs:
        return GaugeDetector(**kwargs)
    if len(params) >= 1:
        return GaugeDetector(config)
    return GaugeDetector()


def detect_gauge(detector: Any, frame_or_packet: Any) -> Any:
    """Invoke GaugeDetector public detection method on BGR ndarray or FramePacket."""
    FramePacket = import_attr("vrc_smith_auto.protocols", "FramePacket")
    for method_name in ("detect", "process", "analyze", "detect_frame"):
        fn = getattr(detector, method_name, None)
        if callable(fn):
            sig = inspect.signature(fn)
            params = list(sig.parameters.values())
            arg = frame_or_packet
            if params:
                first_name = params[0].name.lower()
                ann = str(params[0].annotation)
                if "packet" in first_name or "FramePacket" in ann:
                    if not isinstance(arg, FramePacket):
                        arg = FramePacket(image_bgr=arg, timestamp_s=0.0, frame_idx=0)
                elif "image" in first_name or "bgr" in first_name or "ndarray" in ann:
                    if isinstance(arg, FramePacket):
                        arg = arg.image_bgr
            return fn(arg)
    if callable(detector):
        return detector(frame_or_packet)
    pytest.fail("GaugeDetector exposes no public detect/process/analyze method")


def make_trigger_state_machine(config: Any | None = None, mouse: Any | None = None) -> Any:
    """Instantiate TriggerStateMachine via public signature inspection."""
    TriggerStateMachine = import_optional_attr(
        ("vrc_smith_auto.control.state_machine", "vrc_smith_auto.control"),
        ("TriggerStateMachine",),
    )
    sig = inspect.signature(TriggerStateMachine)
    params = sig.parameters
    kwargs: dict[str, Any] = {}
    if config is not None:
        if "config" in params:
            ann = str(params["config"].annotation)
            if "TriggerConfig" in ann and hasattr(config, "trigger"):
                kwargs["config"] = config.trigger
            else:
                kwargs["config"] = config
        elif "trigger_config" in params:
            kwargs["trigger_config"] = getattr(config, "trigger", config)
        elif "trigger" in params:
            kwargs["trigger"] = getattr(config, "trigger", config)
    if mouse is not None:
        if "mouse" in params:
            kwargs["mouse"] = mouse
        elif "mouse_backend" in params:
            kwargs["mouse_backend"] = mouse
    return TriggerStateMachine(**kwargs)


def step_state_machine(
    sm: Any,
    detection: Any,
    timestamp_s: float,
    mouse: Any | None = None,
    frame_idx: int | None = None,
) -> Any:
    """Advance TriggerStateMachine by one detection step."""
    for method_name in ("update", "step", "process", "on_detection"):
        fn = getattr(sm, method_name, None)
        if callable(fn):
            sig = inspect.signature(fn)
            params = sig.parameters
            kwargs: dict[str, Any] = {}
            pos_args: list[Any] = [detection]
            if "timestamp_s" in params:
                kwargs["timestamp_s"] = timestamp_s
            elif "now_s" in params:
                kwargs["now_s"] = timestamp_s
            elif "ts" in params:
                kwargs["ts"] = timestamp_s
            else:
                pos_args.append(timestamp_s)

            handles_mouse = False
            if mouse is not None:
                if "mouse" in params:
                    kwargs["mouse"] = mouse
                    handles_mouse = True
                elif "mouse_backend" in params:
                    kwargs["mouse_backend"] = mouse
                    handles_mouse = True
            if frame_idx is not None and "frame_idx" in params:
                kwargs["frame_idx"] = frame_idx

            decision = fn(*pos_args, **kwargs) if kwargs else fn(*pos_args)

            if not handles_mouse:
                sm_init_params = inspect.signature(type(sm).__init__).parameters
                handles_mouse = ("mouse" in sm_init_params) or ("mouse_backend" in sm_init_params)

            if not handles_mouse and mouse is not None:
                if hasattr(mouse, "tick"):
                    mouse.tick(timestamp_s)
                if getattr(decision, "should_click", False) is True:
                    if frame_idx is not None and hasattr(mouse, "set_frame_idx"):
                        mouse.set_frame_idx(frame_idx)
                    mouse.press_left(float(getattr(decision, "hold_ms", 30.0)), timestamp_s=timestamp_s)

            return decision
    pytest.fail("TriggerStateMachine exposes no public update/step/process method")


def make_image_source(image_path: Path) -> Any:
    """Instantiate ImageFrameSource with deferred import."""
    cls = import_optional_attr(
        ("vrc_smith_auto.capture.offline_source", "vrc_smith_auto.capture"),
        ("ImageFrameSource",),
    )
    return cls(image_path)


def make_video_source(video_path: Path, **kwargs: Any) -> Any:
    """Instantiate VideoFileFrameSource with deferred import."""
    cls = import_optional_attr(
        ("vrc_smith_auto.capture.offline_source", "vrc_smith_auto.capture"),
        ("VideoFileFrameSource",),
    )
    sig = inspect.signature(cls)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return cls(video_path, **filtered)


def make_capture_backend(backend_name: str = "auto", config: Any | None = None) -> Any:
    """Resolve a live capture FrameSource via public capture factory or class."""
    ensure_src_on_path()
    for mod_name in ("vrc_smith_auto.capture", "vrc_smith_auto.capture.base"):
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        for fn_name in (
            "create_live_capture",
            "create_capture_source",
            "create_frame_source",
            "build_capture_source",
            "make_capture_source",
            "select_capture_backend",
            "get_capture_backend",
        ):
            fn = getattr(mod, fn_name, None)
            if callable(fn):
                sig = inspect.signature(fn)
                params = sig.parameters
                kwargs: dict[str, Any] = {}
                if "backend" in params:
                    kwargs["backend"] = backend_name
                elif "name" in params:
                    kwargs["name"] = backend_name

                if "config" in params:
                    if config is not None:
                        if (
                            backend_name != "auto"
                            and "backend" not in params
                            and "name" not in params
                            and hasattr(config, "with_overrides")
                        ):
                            kwargs["config"] = config.with_overrides(capture_backend=backend_name)
                        else:
                            kwargs["config"] = config
                    elif "backend" not in params and "name" not in params:
                        kwargs["config"] = make_app_config(capture_backend=backend_name)
                    elif params["config"].default is inspect.Parameter.empty:
                        kwargs["config"] = make_app_config(capture_backend=backend_name)

                if kwargs:
                    return fn(**kwargs)
                if params:
                    first_name = next(iter(params))
                    if first_name != "config":
                        return fn(backend_name)
    if backend_name == "gdi":
        cls = import_optional_attr(
            ("vrc_smith_auto.capture.gdi_source", "vrc_smith_auto.capture"),
            ("GdiBitBltCapture",),
        )
        return cls()
    pytest.fail(f"No public capture backend factory found for '{backend_name}'")


def run_app_runner(
    source: Any,
    mouse: Any,
    config: Any | None = None,
    detector: Any | None = None,
    state_machine: Any | None = None,
    hotkeys: Any | None = None,
) -> Any:
    """Invoke vrc_smith_auto.app.runner public class or function cleanly."""
    runner_mod = import_pkg("vrc_smith_auto.app.runner")
    cfg = config if config is not None else make_app_config(dry_run=True, require_foreground=False)
    det = detector if detector is not None else make_gauge_detector(cfg)
    sm = state_machine if state_machine is not None else make_trigger_state_machine(cfg, mouse=mouse)

    def _bind_kwargs(sig: inspect.Signature) -> dict[str, Any]:
        kw: dict[str, Any] = {}
        mapping = {
            "source": source,
            "frame_source": source,
            "capture": source,
            "mouse": mouse,
            "mouse_backend": mouse,
            "config": cfg,
            "app_config": cfg,
            "detector": det,
            "state_machine": sm,
            "sm": sm,
            "trigger": sm,
            "hotkeys": hotkeys,
            "hotkey_poller": hotkeys,
        }
        for pname in sig.parameters:
            if pname in mapping and mapping[pname] is not None:
                kw[pname] = mapping[pname]
        return kw

    for cls_name in ("Runner", "AppRunner", "PipelineRunner", "AutoClickerRunner"):
        cls = getattr(runner_mod, cls_name, None)
        if isinstance(cls, type):
            kw = _bind_kwargs(inspect.signature(cls))
            inst = cls(**kw)
            for mname in ("run", "run_until_eof", "execute", "start"):
                m = getattr(inst, mname, None)
                if callable(m):
                    mkw = _bind_kwargs(inspect.signature(m))
                    return m(**mkw)

    for fn_name in ("run_offline", "run_pipeline", "run_loop", "run_runner", "run"):
        fn = getattr(runner_mod, fn_name, None)
        if callable(fn):
            kw = _bind_kwargs(inspect.signature(fn))
            return fn(**kw)

    pytest.fail("vrc_smith_auto.app.runner exposes no recognized public Runner class or run function")


def render_synthetic_gauge_frame(**kwargs: Any) -> Any:
    """Convenience wrapper delegating to synthetic_fixtures.render_gauge_frame."""
    ensure_src_on_path()
    import synthetic_fixtures as sf

    spec_fields = {f for f in sf.GaugeSpec.__dataclass_fields__}
    # Support Explorer 1 alias kwargs cleanly
    mapped: dict[str, Any] = {}
    alias_map = {
        "gold_offset_px": "gold_offset",
        "gold_len_px": "gold_len",
        "bar_width_px": "bar_width",
        "red_offset_px": "red_pos",
        "red_width_px": "red_width",
        "red_thickness_px": "red_thickness",
    }
    for k, v in kwargs.items():
        if k == "center_xy":
            mapped["cx"], mapped["cy"] = float(v[0]), float(v[1])
        elif k == "red_over_gauge":
            if not v:
                mapped["red_pos"] = None
        elif k in alias_map:
            mapped[alias_map[k]] = v
        elif k in spec_fields:
            mapped[k] = v
    return sf.render_gauge_frame(sf.GaugeSpec(**mapped))


def write_synthetic_video(path: Path, frames: list[Any], fps: float = 60.0) -> Path:
    """Convenience wrapper delegating to synthetic_fixtures.write_synthetic_mp4."""
    ensure_src_on_path()
    import synthetic_fixtures as sf

    return sf.write_synthetic_mp4(path, frames, fps=fps)


@pytest.fixture(scope="session")
def e2e() -> Any:
    """Return the e2e conftest helper module itself."""
    return sys.modules[__name__]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def reference_screenshot_path() -> Path:
    assert REFERENCE_SCREENSHOT_PATH.exists(), (
        f"Missing reference screenshot: {REFERENCE_SCREENSHOT_PATH}"
    )
    return REFERENCE_SCREENSHOT_PATH


@pytest.fixture(scope="session")
def reference_video_path() -> Path:
    assert REFERENCE_VIDEO_PATH.exists(), (
        f"Missing reference gameplay video: {REFERENCE_VIDEO_PATH}"
    )
    return REFERENCE_VIDEO_PATH


@pytest.fixture
def tmp_config_yaml(tmp_path: Path) -> Any:
    def _factory(overrides: dict[str, Any] | None = None, filename: str = "config.yaml") -> Path:
        yaml = import_yaml()
        cfg_path = tmp_path / filename
        payload = {"capture_backend": "gdi", "dry_run": True} if overrides is None else overrides
        cfg_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        return cfg_path

    return _factory


@pytest.fixture
def tmp_config_json(tmp_path: Path) -> Any:
    def _factory(overrides: dict[str, Any] | None = None, filename: str = "config.json") -> Path:
        cfg_path = tmp_path / filename
        payload = {"capture_backend": "gdi", "dry_run": True} if overrides is None else overrides
        cfg_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return cfg_path

    return _factory


@pytest.fixture
def synthetic_gauge_frame_factory() -> Any:
    return render_synthetic_gauge_frame


@pytest.fixture
def synthetic_gauge_video_factory(tmp_path: Path) -> Any:
    def _factory(
        frames: list[Any],
        filename: str = "synthetic.mp4",
        fps: float = 60.0,
    ) -> Path:
        return write_synthetic_video(tmp_path / filename, frames, fps=fps)

    return _factory
