import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CONFIDENT = "#34c759"
UNSURE = "#ff9f0a"
CONF_CUTOFF = 0.8


def box_color(conf):
    return CONFIDENT if conf >= CONF_CUTOFF else UNSURE


def draw_boxes(rgb, words, min_conf=0.0, labels=True):
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    scale = max(rgb.shape[:2]) / 1000
    width = max(2, round(2 * scale))
    font = ImageFont.load_default(size=max(12, round(15 * scale)))

    for wd in words:
        if wd["rec_conf"] < min_conf:
            continue
        color = box_color(wd["rec_conf"])
        pts = [tuple(map(float, p)) for p in wd["box"]]
        draw.line(pts + [pts[0]], fill=color, width=width, joint="curve")
        if labels and wd["text"]:
            x, y = pts[0]
            tw = draw.textlength(wd["text"], font=font)
            th = font.size + 4
            top = max(0, y - th)
            draw.rectangle([x, top, x + tw + 8, top + th], fill=color)
            draw.text((x + 4, top + 1), wd["text"], fill="#000000", font=font)
    return np.asarray(img)


def heatmap(prob, size):
    heat = (np.clip(prob, 0.0, 1.0) * 255).astype(np.uint8)
    heat = cv2.resize(heat, size, interpolation=cv2.INTER_LINEAR)
    return np.stack([heat] * 3, axis=-1)


def visible_words(words, min_conf):
    return [w for w in words if w["rec_conf"] >= min_conf and w["text"]]
