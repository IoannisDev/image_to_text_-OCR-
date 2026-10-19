import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "model"))

from model import CNN_FE, DBNet, TransformerEncoderHead  # noqa: E402
from ocr_pipeline import group_into_lines, load_pipeline  # noqa: E402

DEFAULTS = {
    "CHARSET_PATH": "data/charset.json",
    "DBNET_CKPT": "model_pt/dbnet_infer.pt",
    "CRNN_CKPT": "model_pt/crnn_infer.pt",
}


class OCRSetupError(RuntimeError):
    pass


@dataclass
class Settings:
    bin_thresh: float = 0.4
    box_thresh: float = 0.6
    unclip_scale: float = 0.8
    long_side: int = 640


@dataclass
class OCRResult:
    words: list
    lines: list
    text: str
    prob: np.ndarray
    seconds: float
    device: str
    image_size: tuple = field(default=(0, 0))


def _path(name):
    load_dotenv(ROOT / ".env")
    p = Path(os.getenv(name) or DEFAULTS[name]).expanduser()
    p = p if p.is_absolute() else ROOT / p
    if not p.exists():
        raise OCRSetupError(f"{name} points to a missing file: {p}")
    return str(p)


def build_pipeline():
    charset = _path("CHARSET_PATH")
    num_classes = json.loads(Path(charset).read_text(encoding="utf-8"))["num_classes"]
    return load_pipeline(
        dbnet_ckpt=_path("DBNET_CKPT"),
        crnn_ckpt=_path("CRNN_CKPT"),
        charset_json=charset,
        DBNet=lambda: DBNet(pretrained=False, prob_only=True),
        CNN_layer=CNN_FE,
        make_rnn=lambda: TransformerEncoderHead(num_classes),
    )


def run_ocr(pipe, rgb, settings):
    pipe.long_side = settings.long_side
    pipe.bin_thresh = settings.bin_thresh
    pipe.box_thresh = settings.box_thresh
    pipe.unclip_scale = settings.unclip_scale

    t0 = time.perf_counter()
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    boxes, scores, prob = pipe.detect(rgb)
    texts, confs = pipe.recognize(gray, boxes)

    words = [
        {"box": b, "text": t, "det_score": s, "rec_conf": c}
        for b, t, s, c in zip(boxes, texts, scores, confs)
    ]
    lines = [
        " ".join(texts[i] for i in ln if texts[i]) for ln in group_into_lines(boxes)
    ]
    lines = [ln for ln in lines if ln]
    return OCRResult(
        words=words,
        lines=lines,
        text="\n".join(lines),
        prob=prob,
        seconds=time.perf_counter() - t0,
        device=str(pipe.device),
        image_size=(rgb.shape[1], rgb.shape[0]),
    )
