import os
import json, threading, time
from pathlib import Path
import cv2, numpy as np
from dotenv import load_dotenv
from model import DBNet, CNN_FE, TransformerEncoderHead
from ocr_pipeline import load_pipeline

ROOT = Path(__file__).resolve().parent.parent       # project root, where .env lives
if not load_dotenv(ROOT / ".env"):
    raise SystemExit(f"no .env at {ROOT} - copy .env.example to .env and set your paths")

def env_path(name):
    # relative paths in .env are taken from the project root, not the cwd
    if not os.getenv(name):
        raise SystemExit(f"{name} is not set in .env")
    p = Path(os.environ[name]).expanduser()
    p = p if p.is_absolute() else ROOT / p
    if not p.exists():
        raise SystemExit(f"{name} points to a missing file: {p}")
    return str(p)

CHARSET = env_path("CHARSET_PATH")
num_classes = json.load(open(CHARSET, encoding="utf-8"))["num_classes"]

ocr = load_pipeline(
    dbnet_ckpt=env_path("DBNET_CKPT"),
    crnn_ckpt=env_path("CRNN_CKPT"),
    charset_json=CHARSET,
    DBNet=lambda: DBNet(pretrained=False, prob_only=True),  # skip ImageNet weights and the unused threshold head
    CNN_layer=CNN_FE,
    make_rnn=lambda: TransformerEncoderHead(num_classes),
    long_side=640, unclip_scale=0.8, box_thresh=0.6, bin_thresh=0.4)
print("running on:", ocr.device)

latest = {"frame": None, "frame_id": 0, "words": [], "secs": 0.0, "stop": False}

def ocr_worker():
    done_id = 0
    while not latest["stop"]:
        frame, frame_id = latest["frame"], latest["frame_id"]
        if frame is None or frame_id == done_id:     # nothing new to read
            time.sleep(0.005); continue
        t0 = time.time()
        words = ocr(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))["words"]
        latest["words"], latest["secs"] = words, time.time() - t0
        done_id = frame_id

threading.Thread(target=ocr_worker, daemon=True).start()

cam = int(os.getenv("CAMERA_INDEX", "1"))
cap = cv2.VideoCapture(cam)
if not cap.isOpened():
    raise SystemExit(f"camera {cam} did not open - try another CAMERA_INDEX in .env, and check camera permission for your terminal")
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
print("camera open, starting window")
while True:
    ok, frame = cap.read()
    if not ok:
        print("camera read failed, stopping")
        break
    latest["frame"] = frame.copy()
    latest["frame_id"] += 1

    for wd in latest["words"]:
        if wd["rec_conf"] < 0.6:
            continue
        box = wd["box"].astype(np.int32)
        cv2.polylines(frame, [box], True, (0, 255, 0), 2)
        cv2.putText(frame, wd["text"], (int(box[0][0]), int(box[0][1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"OCR {latest['secs']:.2f}s", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("OCR - press q to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

latest["stop"] = True
cap.release(); cv2.destroyAllWindows()
