import streamlit as st
from PIL import Image
from inference import Captioner

st.set_page_config(page_title="Fast Image Captioner", page_icon="🖼️")

@st.cache_resource
def get_model():
    return Captioner()

captioner = get_model()

st.title("🖼️ → ✍️ Fast Image Captioner (BLIP)")
st.write("Upload an image to get a fast caption. Model: Salesforce/blip-image-captioning-base")

uploaded = st.file_uploader("Choose an image", type=["png","jpg","jpeg","webp"])
length = st.slider("Max tokens (length)", min_value=5, max_value=40, value=30, step=1)

if uploaded is not None:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded image", use_column_width=True)
    with st.spinner("Generating caption..."):
        captioner.gen_kwargs["max_length"] = int(length)
        caption = captioner.caption(img)
    st.success("Done!")
    st.text_area("Caption", value=caption, height=100)
else:
    st.info("Upload an image to begin.")
