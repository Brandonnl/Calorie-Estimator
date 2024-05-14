from PIL import Image
import numpy as np


IMAGE_SIZE = (224,224)


def prepare_image (image):
  img = Image.pen (image)
  img = img.resize (IMAGE_SIZE)
  img = np.expand.dims(img,axis = 0 )
  img = img/255.0
  return img