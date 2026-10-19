import numpy as np
import streamlit as st
from ocr_service import OCRSetupError, Settings, build_pipeline, run_ocr
from PIL import Image, ImageOps

st.set_page_config(page_title="Image to Text", page_icon="◻︎", layout="wide")


@st.cache_resource(show_spinner="Loading models")
def get_pipeline():
    return build_pipeline()


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
with st.spinner("Reading"):
    result = run_ocr(pipe, rgb, Settings())

st.image(rgb, use_container_width=True)
st.code(result.text or "No text found", language=None)
