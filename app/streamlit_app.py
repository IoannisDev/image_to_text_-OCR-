import numpy as np
import pandas as pd
import streamlit as st
from ocr_service import OCRSetupError, Settings, build_pipeline, run_ocr
from PIL import Image, ImageOps
from render import draw_boxes, heatmap, visible_words

st.set_page_config(page_title="Image to Text", page_icon="◻︎", layout="wide")


@st.cache_resource(show_spinner="Loading models")
def get_pipeline():
    return build_pipeline()


@st.cache_data(show_spinner="Reading")
def cached_ocr(_pipe, rgb, bin_thresh, box_thresh, unclip_scale, long_side):
    return run_ocr(_pipe, rgb, Settings(bin_thresh, box_thresh, unclip_scale, long_side))


def read_image(file):
    img = ImageOps.exif_transpose(Image.open(file)).convert("RGB")
    return np.asarray(img)


st.title("Image to Text")
st.caption("Upload a photo or take one with your camera and read the text in it.")

try:
    pipe = get_pipeline()
except (OCRSetupError, FileNotFoundError) as err:
    st.error(f"Could not load the model. {err}")
    st.stop()

defaults = Settings()
with st.sidebar:
    st.header("Detection")
    cfg = Settings(
        bin_thresh=st.slider(
            "Pixel threshold", 0.1, 0.9, defaults.bin_thresh, 0.05,
            help="Pixels above this count as text.",
        ),
        box_thresh=st.slider(
            "Box threshold", 0.1, 0.9, defaults.box_thresh, 0.05,
            help="Drops regions whose average probability is below this.",
        ),
        unclip_scale=st.slider(
            "Box growth", 0.5, 2.0, defaults.unclip_scale, 0.1,
            help="Raise it if boxes cut letters off, lower it if they merge words.",
        ),
        long_side=st.select_slider(
            "Image size", options=[320, 480, 640, 800, 1024, 1280],
            value=defaults.long_side,
            help="The longest side is scaled to this many pixels before detection.",
        ),
    )

    st.header("Display")
    min_conf = st.slider(
        "Minimum confidence", 0.0, 0.95, 0.0, 0.05,
        help="Hide words the model is less sure about.",
    )
    show_labels = st.toggle("Show labels on image", value=True)

source = st.segmented_control(
    "Source", ["Upload", "Camera"], default="Upload", label_visibility="collapsed"
)
if source == "Camera":
    file = st.camera_input("Take a photo", label_visibility="collapsed")
else:
    file = st.file_uploader(
        "Drop an image", type=["png", "jpg", "jpeg", "webp"], label_visibility="collapsed"
    )

if file is None:
    st.stop()

rgb = read_image(file)
result = cached_ocr(
    pipe, rgb, cfg.bin_thresh, cfg.box_thresh, cfg.unclip_scale, cfg.long_side
)

words = visible_words(result.words, min_conf)

c1, c2, c3 = st.columns(3)
c1.metric("Words", len(words))
c2.metric("Time", f"{result.seconds:.2f} s")
c3.metric("Device", result.device)

overlay_tab, text_tab, words_tab, heat_tab = st.tabs(
    ["Overlay", "Text", "Words", "Heatmap"]
)

with overlay_tab:
    st.image(
        draw_boxes(rgb, result.words, min_conf, show_labels), use_container_width=True
    )
    st.caption("Green boxes are confident reads, orange ones are less certain.")

with text_tab:
    if result.text:
        st.code(result.text, language=None, wrap_lines=True)
        st.download_button("Download .txt", result.text, file_name="ocr.txt")
    else:
        st.info("No text found. Try lowering the box threshold.")

with words_tab:
    st.dataframe(
        pd.DataFrame(
            {
                "Word": [w["text"] for w in words],
                "Detection": [w["det_score"] for w in words],
                "Recognition": [w["rec_conf"] for w in words],
            }
        ),
        column_config={
            "Detection": st.column_config.ProgressColumn(min_value=0, max_value=1),
            "Recognition": st.column_config.ProgressColumn(min_value=0, max_value=1),
        },
        hide_index=True,
        use_container_width=True,
    )

with heat_tab:
    st.image(heatmap(result.prob, (rgb.shape[1], rgb.shape[0])), use_container_width=True)
    st.caption("White is where the detector is sure there is text.")
