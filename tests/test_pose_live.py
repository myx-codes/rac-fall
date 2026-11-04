from types import SimpleNamespace
import json
import sys

import pytest

from racfall.pose_live import COCO_SKELETON, draw_pose_overlay, pose_live


class FakeFrame:
    shape = (480, 640, 3)

    def copy(self):
        return self


class FakeCapture:
    def __init__(self, reads=None, opened=True):
        self.reads = list(reads or [])
        self.opened = opened
        self.released = False

    def isOpened(self):
        return self.opened

    def read(self):
        return self.reads.pop(0) if self.reads else (False, None)

    def release(self):
        self.released = True


class FakeCV2:
    CAP_AVFOUNDATION = 1200
    FONT_HERSHEY_SIMPLEX = 0
    LINE_AA = 16

    def __init__(self, capture, keys):
        self.capture = capture
        self.keys = list(keys)
        self.destroyed = False
        self.capture_args = None
        self.lines = []
        self.circles = []
        self.text = []

    def VideoCapture(self, *args):
        self.capture_args = args
        return self.capture

    def waitKey(self, _delay):
        return self.keys.pop(0)

    def imshow(self, _name, _frame):
        pass

    def destroyAllWindows(self):
        self.destroyed = True

    def line(self, *args):
        self.lines.append(args)

    def circle(self, *args):
        self.circles.append(args)

    def putText(self, *args):
        self.text.append(args)

    def rectangle(self, *_args):
        pass


class FakeModel:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def track(self, frame, **kwargs):
        self.calls.append((frame, kwargs))
        return [self.result]


MPS_TORCH = SimpleNamespace(backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)))


def test_draws_all_visible_coco_keypoints_and_tracking_label():
    cv2 = FakeCV2(FakeCapture(), [])
    xy = [[[10 + i, 20 + i] for i in range(17)]]
    result = SimpleNamespace(
        keypoints=SimpleNamespace(xy=xy, conf=[[0.9] * 17]),
        boxes=SimpleNamespace(id=[7], conf=[0.8]),
    )

    count = draw_pose_overlay(FakeFrame(), result, cv2, fps=24.5, inference_ms=12.3)

    assert count == 1
    assert len(cv2.circles) == 17
    assert len(cv2.lines) == len(COCO_SKELETON)
    assert any("ID 7  pose 0.90" in call[1] for call in cv2.text)
    assert any("FPS  24.5" in call[1] and "inference  12.3 ms" in call[1] for call in cv2.text)


def test_live_uses_avfoundation_tracking_pause_and_cleanup():
    capture = FakeCapture([(True, FakeFrame()), (True, FakeFrame())])
    cv2 = FakeCV2(capture, [ord("p"), ord("p"), ord("q")])
    result = SimpleNamespace(keypoints=None, boxes=None, speed={"inference": 10.0})
    model = FakeModel(result)
    times = iter([0.0, 1.0, 2.0, 4.0])

    summary = pose_live(
        "0", model_factory=lambda _weights: model, cv2_module=cv2,
        torch_module=MPS_TORCH, clock=lambda: next(times),
    )

    assert cv2.capture_args == (0, cv2.CAP_AVFOUNDATION)
    assert capture.released and cv2.destroyed
    assert len(model.calls) == 2  # no capture or inference while paused
    assert model.calls[0][1] == {
        "persist": True, "tracker": "bytetrack.yaml", "device": "mps", "imgsz": 416,
        "verbose": False, "save": False, "show": False,
    }
    assert summary["frames_processed"] == 2
    assert summary["mean_inference_latency_ms"] == 10.0
    assert summary["stores_rgb_or_poses"] is False


def test_camera_read_error_still_cleans_up():
    capture = FakeCapture([])
    cv2 = FakeCV2(capture, [])
    model = FakeModel(SimpleNamespace())

    with pytest.raises(RuntimeError, match="stopped returning frames"):
        pose_live(0, device="cpu", model_factory=lambda _weights: model,
                  cv2_module=cv2, torch_module=MPS_TORCH)

    assert capture.released and cv2.destroyed


def test_rejects_invalid_source_and_unavailable_mps():
    no_mps = SimpleNamespace(backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)))
    with pytest.raises(ValueError, match="webcam index"):
        pose_live("movie.mp4", model_factory=lambda _: None, cv2_module=object(), torch_module=MPS_TORCH)
    with pytest.raises(RuntimeError, match="MPS requested"):
        pose_live(0, model_factory=lambda _: None, cv2_module=object(), torch_module=no_mps)


def test_cli_pose_live_defaults(monkeypatch, capsys):
    from racfall import cli
    import racfall.pose_live as live_module

    called = {}

    def fake_pose_live(source, imgsz, device):
        called.update(source=source, imgsz=imgsz, device=device)
        return {"frames_processed": 1}

    monkeypatch.setattr(live_module, "pose_live", fake_pose_live)
    monkeypatch.setattr(sys, "argv", ["racfall", "pose-live"])
    cli.main()

    assert called == {"source": "0", "imgsz": 416, "device": "mps"}
    assert json.loads(capsys.readouterr().out) == {"frames_processed": 1}
