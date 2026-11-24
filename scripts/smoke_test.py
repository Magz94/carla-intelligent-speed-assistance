"""Check source syntax, bundled assets and a CPU model forward pass without CARLA."""
from pathlib import Path
import ast
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for path in sorted((ROOT / "src").glob("*.py")):
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    print("Syntax OK:", path.name)
for name in ("sign_30.png", "sign_60.png", "sign_90.png", "beep.mp3"):
    path = ROOT / "assets" / name
    assert path.is_file() and path.stat().st_size > 0, str(path)
net = cv2.dnn.readNetFromDarknet(
    str(ROOT / "models/yolov3-tiny-obj.cfg"),
    str(ROOT / "models/yolov3-tiny-obj_5000.weights"),
)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
net.setInput(np.zeros((1, 3, 608, 608), dtype=np.float32))
outputs = net.forward(net.getUnconnectedOutLayersNames())
assert len(outputs) == 3 and all(x.shape[-1] == 8 for x in outputs), "Unexpected output shape"
assert all(np.isfinite(x).all() for x in outputs), "Non-finite model output"
print("CPU forward pass OK:", [x.shape for x in outputs])
print("Smoke test passed; CARLA integration was not tested.")
