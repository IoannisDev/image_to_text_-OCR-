import numpy as np
import streamlit as st
from ocr_service import OCRSetupError, Settings, build_pipeline, run_ocr
from PIL import Image, ImageOps

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

st.image(rgb, use_container_width=True)
st.code(result.text or "No text found", language=None)
