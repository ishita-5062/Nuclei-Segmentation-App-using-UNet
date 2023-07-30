import streamlit as st
from PIL import Image
import test_model

st.set_page_config(layout='wide')
st.title("Nuclei Segmentation App")

img=st.sidebar.selectbox("Select Image", ("test_img_1.png",
                                          "test_img_2.png",
                                          "test_img_3.png",
                                          "test_img_4.png",
                                          "test_img_5.png"))

input_image="test_images/"+img
image=Image.open(input_image)
st.image(image, width=400)

detect_mask=st.button("Detect Mask")

if detect_mask:
    mask = test_model.do_pred(input_image)
    st.write("Predicted Mask:")
    st.image(mask, width=400)