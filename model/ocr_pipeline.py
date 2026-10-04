
import inspect
import json
from pathlib import Path

import cv2
import numpy as np

try:                      # torch is only needed to run the models, not the geometry
    import torch
    import torch.nn.functional as F
except ImportError:       # pragma: no cover
    torch = None

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
OUT_H = 32                # CRNN input height
STRIDE = 4                # CRNN width downsampling


def default_device():
    """CUDA if present, else Apple-silicon GPU (MPS), else CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_image(image):
    if isinstance(image, (str, Path)):
        bgr = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(image)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return np.asarray(image)


def order_corners(pts):
    pts = np.asarray(pts, dtype=np.float32)
    by_x = pts[np.argsort(pts[:, 0])]
    left, right = by_x[:2], by_x[2:]
    tl, bl = left[np.argsort(left[:, 1])]
    tr, br = right[np.argsort(right[:, 1])]
    return np.array([tl, tr, br, bl], dtype=np.float32)


#DBNet post processing 
def unclip_distance(w, h, shrink_ratio=0.4):
    """
    How far to grow a predicted region to recover the full word box.

    Training shrank each word by D = A(1 - r^2) / L, computed on the ORIGINAL
    box. Here we only see the shrunk w x h, so we solve for the D that would
    have produced it (exact for rectangles):

        (8 - 4k) D^2 + (2 - 2k) S D - k P = 0,   k = 1 - r^2, S = w + h, P = w h
    """
    k = 1.0 - shrink_ratio ** 2
    S, P = w + h, w * h
    a, b, c = 8 - 4 * k, (2 - 2 * k) * S, -k * P
    return (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)


def region_score(prob, contour):
    """Mean probability inside one contour."""
    x, y, w, h = cv2.boundingRect(contour)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [contour.reshape(-1, 2) - [x, y]], 1)
    return float(cv2.mean(prob[y:y + h, x:x + w], mask)[0])


def boxes_from_prob_map(prob, bin_thresh=0.3, box_thresh=0.6, shrink_ratio=0.4,
                        unclip_scale=1.0, min_size=3, max_candidates=2000):
    """
    prob: (H, W) float array in [0, 1] -- DBNet's probability map.
    Returns (boxes, scores): each box is 4x2 float32, corners ordered
    TL, TR, BR, BL, in the coordinates of `prob`.

    bin_thresh    pixels above this count as text
    box_thresh    drop regions whose mean probability is below this
    unclip_scale  multiply the growth distance; raise it if boxes cut letters
                  off, lower it if they swallow neighbouring words
    """
    binary = (prob > bin_thresh).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    boxes, scores = [], []
    for contour in contours[:max_candidates]:
        (cx, cy), (w, h), angle = cv2.minAreaRect(contour)
        w, h = w + 1, h + 1            # contours run through pixel centres
        if min(w, h) < min_size:
            continue
        score = region_score(prob, contour)
        if score < box_thresh:
            continue
        d = unclip_distance(w, h, shrink_ratio) * unclip_scale
        grown = cv2.boxPoints(((cx, cy), (w + 2 * d, h + 2 * d), angle))
        boxes.append(order_corners(grown))
        scores.append(score)
    return boxes, scores


#Crops the DBNet batch image
def crop_word(gray, box, out_h=OUT_H, min_w=8, max_w=2000):
    """Same perspective warp the CRNN was trained on: height 32, aspect kept."""
    p = np.asarray(box, dtype=np.float32)
    h = (np.linalg.norm(p[3] - p[0]) + np.linalg.norm(p[2] - p[1])) / 2
    w = (np.linalg.norm(p[1] - p[0]) + np.linalg.norm(p[2] - p[3])) / 2
    out_w = int(np.clip(round(w * out_h / max(h, 1e-3)), min_w, max_w))
    dst = np.float32([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]])
    M = cv2.getPerspectiveTransform(p, dst)
    return cv2.warpPerspective(gray, M, (out_w, out_h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)


# Reading Order
def group_into_lines(boxes):
    """
    Returns a list of lines, each a list of box indices, top-to-bottom then
    left-to-right. A word joins a line when its vertical centre is within half
    a text height of that line's centre.
    """
    if not boxes:
        return []
    cy = [float(b[:, 1].mean()) for b in boxes]
    cx = [float(b[:, 0].mean()) for b in boxes]
    hh = [float((np.linalg.norm(b[3] - b[0]) + np.linalg.norm(b[2] - b[1])) / 2)
          for b in boxes]

    lines = []                                   # each: {"idx": [...], "cy": .., "h": ..}
    for i in sorted(range(len(boxes)), key=lambda k: cy[k]):
        best, best_gap = None, None
        for ln in lines:
            gap = abs(cy[i] - ln["cy"])
            if gap < 0.5 * max(ln["h"], hh[i]) and (best is None or gap < best_gap):
                best, best_gap = ln, gap
        if best is None:
            lines.append({"idx": [i], "cy": cy[i], "h": hh[i]})
        else:
            best["idx"].append(i)
            n = len(best["idx"])
            best["cy"] += (cy[i] - best["cy"]) / n
            best["h"] += (hh[i] - best["h"]) / n

    lines.sort(key=lambda ln: ln["cy"])
    return [sorted(ln["idx"], key=lambda k: cx[k]) for ln in lines]


#the pipeline
class OCRPipeline:
    def __init__(self, dbnet, cnn, rnn, idx_to_char, device=None,
                 long_side=1024, bin_thresh=0.3, box_thresh=0.6,
                 unclip_scale=1.0, shrink_ratio=0.4, min_size=3, rec_batch=128):
        self.device = torch.device(device or default_device())
        self.dbnet = dbnet.to(self.device).eval()
        self.cnn = cnn.to(self.device).eval()
        self.rnn = rnn.to(self.device).eval()
        self.idx_to_char = idx_to_char
        self.pass_lengths = len(inspect.signature(rnn.forward).parameters) >= 2
        self.long_side = long_side
        self.bin_thresh, self.box_thresh = bin_thresh, box_thresh
        self.unclip_scale, self.shrink_ratio = unclip_scale, shrink_ratio
        self.min_size, self.rec_batch = min_size, rec_batch

    # ---------------- detection ----------------
    def detect(self, image):
        """Returns (boxes, scores, prob_map). Boxes are in ORIGINAL image pixels."""
        rgb = load_image(image)
        h, w = rgb.shape[:2]
        scale = self.long_side / max(h, w)
        nh, nw = max(32, int(round(h * scale))), max(32, int(round(w * scale)))
        img = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)

        # pad to a multiple of 32, as DBNet's forward requires
        ph, pw = (32 - nh % 32) % 32, (32 - nw % 32) % 32
        if ph or pw:
            img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        x = (img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(x.transpose(2, 0, 1))[None].to(self.device)

        with torch.no_grad():
            out = self.dbnet(x)
        if isinstance(out, dict):
            p = out.get("prob", out.get("P", next(iter(out.values()))))
        elif isinstance(out, (tuple, list)):
            p = out[0]
        else:
            p = out[:, :1]
        p = torch.sigmoid(p.float())             # DBNet's prob head returns logits
        if p.shape[-2:] != x.shape[-2:]:
            p = F.interpolate(p, size=x.shape[-2:], mode="bilinear", align_corners=False)
        prob = p[0, 0, :nh, :nw].cpu().numpy()

        boxes, scores = boxes_from_prob_map(
            prob, self.bin_thresh, self.box_thresh, self.shrink_ratio,
            self.unclip_scale, self.min_size)
        boxes = [b / scale for b in boxes]       # back to original pixels
        return boxes, scores, prob

    # recognition 
    def recognize(self, gray, boxes):
        """Returns (texts, confidences), one per box, in the order given."""
        n = len(boxes)
        texts, confs = [""] * n, [0.0] * n
        if n == 0:
            return texts, confs
        crops = [crop_word(gray, b) for b in boxes]

        # batch crops of similar width together so little is wasted on padding
        order = sorted(range(n), key=lambda i: crops[i].shape[1])
        for s in range(0, n, self.rec_batch):
            idx = order[s:s + self.rec_batch]
            widths = [crops[i].shape[1] for i in idx]
            batch = torch.zeros(len(idx), 1, OUT_H, max(widths))
            for k, i in enumerate(idx):
                batch[k, 0, :, :widths[k]] = torch.from_numpy(crops[i]).float() / 255.0
            lengths = torch.tensor([wd // STRIDE for wd in widths], dtype=torch.long)

            with torch.no_grad():
                feats = self.cnn(batch.to(self.device))
                lp = self.rnn(feats, lengths) if self.pass_lengths else self.rnn(feats)
            best_p, best_i = lp.float().max(dim=2)          # (T, B)
            best_p, best_i = best_p.exp().cpu(), best_i.cpu()
            T = best_i.shape[0]

            for k, i in enumerate(idx):
                L = min(int(lengths[k]), T)
                prev, chars, ps = None, [], []
                for c, pr in zip(best_i[:L, k].tolist(), best_p[:L, k].tolist()):
                    if c != prev and c != 0:                # greedy CTC decode
                        chars.append(self.idx_to_char.get(c, ""))
                        ps.append(pr)
                    prev = c
                texts[i] = "".join(chars)
                confs[i] = float(np.mean(ps)) if ps else 0.0
        return texts, confs

    # both
    def __call__(self, image):
        rgb = load_image(image)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        boxes, scores, _ = self.detect(rgb)
        texts, confs = self.recognize(gray, boxes)

        words = [{"box": b, "text": t, "det_score": s, "rec_conf": c}
                 for b, t, s, c in zip(boxes, texts, scores, confs)]
        lines = [" ".join(texts[i] for i in ln if texts[i])
                 for ln in group_into_lines(boxes)]
        lines = [ln for ln in lines if ln]
        return {"words": words, "lines": lines, "text": "\n".join(lines)}


def load_pipeline(dbnet_ckpt, crnn_ckpt, charset_json, DBNet, CNN_layer, make_rnn,
                  device=None, **pipeline_kwargs):
    """Build the three models, load the checkpoints, return an OCRPipeline."""
    device = torch.device(device or default_device())

    chars = json.loads(Path(charset_json).read_text(encoding="utf-8"))["chars"]
    idx_to_char = {i: c for i, c in enumerate(chars, start=1)}     # 0 = CTC blank

    dbnet = DBNet()
    ck = torch.load(dbnet_ckpt, map_location=device, weights_only=False)
    dbnet.load_state_dict(ck["model_state_dict"])
    print(f"DBNet: epoch {ck.get('epoch')}  pixel-F1 {ck.get('f1')}")

    cnn, rnn = CNN_layer(), make_rnn()
    ck = torch.load(crnn_ckpt, map_location=device, weights_only=False)
    saved = ck.get("num_classes")
    if saved is not None and saved != len(chars) + 1:
        raise ValueError(
            f"CRNN checkpoint has {saved} classes but {charset_json} defines "
            f"{len(chars) + 1}. This is not the charset the model was trained with, "
            f"so its output would decode to the wrong characters.")
    cnn.load_state_dict(ck["cnn_state_dict"])
    rnn.load_state_dict(ck["rnn_state_dict"])
    print(f"CRNN:  epoch {ck.get('epoch')}  val_CER {ck.get('val_cer')}")

    return OCRPipeline(dbnet, cnn, rnn, idx_to_char, device=device, **pipeline_kwargs)


# display
def show(image, result, save_path=None, figsize=(14, 10), min_conf=0.0):
    """Draw each box with its predicted text. Low-confidence words in orange."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    rgb = load_image(image)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(rgb)
    for wd in result["words"]:
        if wd["rec_conf"] < min_conf:
            continue
        color = "lime" if wd["rec_conf"] >= 0.8 else "orange"
        ax.add_patch(Polygon(wd["box"], closed=True, fill=False, edgecolor=color, linewidth=1.5))
        x, y = wd["box"][0]
        ax.text(x, y - 3, wd["text"], fontsize=8, color="black",
                bbox=dict(facecolor=color, alpha=0.75, pad=1, edgecolor="none"))
    ax.axis("off")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


#  detector evaluation
def _clip_path(poly, scale=16):
    return np.round(np.asarray(poly, dtype=np.float64) * scale).astype(np.int64).tolist()


def _area(poly):
    return float(abs(cv2.contourArea(np.asarray(poly, dtype=np.float32))))


def intersection_area(a, b, scale=16):
    """Area of overlap between two polygons (any shape, not just rectangles)."""
    import pyclipper
    a, b = np.asarray(a), np.asarray(b)
    if (a[:, 0].max() < b[:, 0].min() or b[:, 0].max() < a[:, 0].min() or
            a[:, 1].max() < b[:, 1].min() or b[:, 1].max() < a[:, 1].min()):
        return 0.0
    try:
        pc = pyclipper.Pyclipper()
        pc.AddPath(_clip_path(a, scale), pyclipper.PT_SUBJECT, True)
        pc.AddPath(_clip_path(b, scale), pyclipper.PT_CLIP, True)
        parts = pc.Execute(pyclipper.CT_INTERSECTION,
                           pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO)
    except pyclipper.ClipperException:
        return 0.0
    return sum(abs(pyclipper.Area(p)) for p in parts) / (scale * scale)


def match_detections(pred_boxes, gt_polys, gt_ignore, iou_thresh=0.5):
    """
    Box-level matching for one image. Returns (true_positives, n_pred, n_gt).

    Follows the usual scene-text protocol: ground-truth regions marked ignore
    (illegible or tiny) are not counted as misses, and a prediction that lands
    mostly on one of them is not counted as a false alarm.
    """
    care = [g for g, ig in zip(gt_polys, gt_ignore) if not ig]
    dont = [g for g, ig in zip(gt_polys, gt_ignore) if ig]

    preds = []
    for p in pred_boxes:
        pa = max(_area(p), 1e-6)
        if any(intersection_area(p, g) / pa > 0.5 for g in dont):
            continue
        preds.append(p)

    care_area = [_area(g) for g in care]
    taken, tp = set(), 0
    for p in preds:
        pa = _area(p)
        best, best_j = 0.0, -1
        for j, g in enumerate(care):
            if j in taken:
                continue
            inter = intersection_area(p, g)
            if inter <= 0:
                continue
            iou = inter / max(pa + care_area[j] - inter, 1e-6)
            if iou > best:
                best, best_j = iou, j
        if best >= iou_thresh:
            taken.add(best_j)
            tp += 1
    return tp, len(preds), len(care)


def evaluate_detector(pipeline, json_path, img_dir, image_ids, max_images=300,
                      min_text_size=8, iou_thresh=0.5):
    """
    Box-level precision / recall / F1 at IoU 0.5 over a set of TextOCR images.
    This is the real detection metric; the pixel-F1 printed during training is
    only a proxy. Pass the detector's validation ids (textocr_split_ids).
    """
    d = json.loads(Path(json_path).read_text())
    img_dir = Path(img_dir)
    tp = n_pred = n_gt = n_img = 0

    for img_id in list(image_ids)[:max_images]:
        name = d["imgs"].get(img_id, {}).get("file_name")
        path = img_dir / (Path(name).name if name else f"{img_id}.jpg")
        if not path.exists():
            continue
        polys, ignore = [], []
        for aid in d["imgToAnns"].get(img_id, []):
            ann = d["anns"][aid]
            pts = np.array(ann["points"], dtype=np.float32).reshape(-1, 2)
            if len(pts) < 3:
                continue
            short = min(pts[:, 0].max() - pts[:, 0].min(), pts[:, 1].max() - pts[:, 1].min())
            polys.append(pts)
            ignore.append(ann["utf8_string"] == "." or short < min_text_size)

        boxes, _, _ = pipeline.detect(path)
        a, b, c = match_detections(boxes, polys, ignore, iou_thresh)
        tp, n_pred, n_gt, n_img = tp + a, n_pred + b, n_gt + c, n_img + 1

    precision = tp / max(n_pred, 1)
    recall = tp / max(n_gt, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    print(f"images {n_img}   predictions {n_pred}   ground-truth words {n_gt}")
    print(f"precision {precision:.4f}   recall {recall:.4f}   F1 {f1:.4f}   (IoU >= {iou_thresh})")
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "n_pred": n_pred, "n_gt": n_gt, "images": n_img}