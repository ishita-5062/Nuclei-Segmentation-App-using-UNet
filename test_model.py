import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
model=load_model("unet_model.h5")
#img="NucleiDataset/stage1_test/4f949bd8d914bbfa06f40d6a0e2b5b75c38bf53dbcbafc48c97f105bee4f8fac/images/4f949bd8d914bbfa06f40d6a0e2b5b75c38bf53dbcbafc48c97f105bee4f8fac.png"
from skimage.io import imshow, imread
from skimage.transform import resize

def do_pred(img):
    IMG_WIDTH=128
    IMG_HEIGHT=128
    IMG_CHANNELS=3

    X_test=np.zeros((1,IMG_WIDTH,IMG_HEIGHT,IMG_CHANNELS), dtype=np.uint8)
    img=imread(img)[:,:,:IMG_CHANNELS]
    img=resize(img, (IMG_HEIGHT, IMG_WIDTH), mode="constant", preserve_range=True)

    X_test[0]=img

    mask=model.predict(X_test)
    mask=mask.reshape(128,128)
    return mask