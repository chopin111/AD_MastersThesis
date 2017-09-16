import numpy as np
import scipy.ndimage

DEST_WIDTH = 128 #256
DEST_HEIGHT = 128 #256
DEST_DEPTH = 80 #166

def reshapeImage(image):
    originalShape = image.shape
    zoomWidth = DEST_WIDTH/originalShape[0]
    zoomHeight = DEST_HEIGHT/originalShape[1]
    zoomDepth = DEST_DEPTH/originalShape[2]

    zoom = [zoomWidth, zoomHeight, zoomDepth]

    zoomed = scipy.ndimage.interpolation.zoom(image, zoom)

    print(zoomed.shape)

    indecies = list(range(0, int(DEST_HEIGHT*0.15))) + list(range(int(DEST_HEIGHT*0.5), DEST_HEIGHT)) # get only the middle

    cropped = np.delete(zoomed, indecies, axis=0)
    return cropped
