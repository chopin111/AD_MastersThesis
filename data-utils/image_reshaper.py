import numpy as np
import scipy.ndimage
import math

def reshapeImage(image, reshape_type="3d", step=0):
    if reshape_type == "3d":
        return reshapeImage3d(image)
    elif reshape_type == "2d":
        return reshapeImage2d(image)
    else:
        return reshapeImage2d(image, step)

def reshapeImage2d(image, step=0):
    DEST_WIDTH = 3
    DEST_HEIGHT = 224
    DEST_DEPTH = 224

    originalShape = image.shape
    zoomDepth = DEST_HEIGHT/originalShape[2]
    zoomHeight = DEST_DEPTH/originalShape[1]
    zoomWidth = zoomHeight

    zoom = [zoomWidth, zoomHeight, zoomDepth]

    zoomed = scipy.ndimage.interpolation.zoom(image, zoom)

    print(zoomed.shape)

    computed_height = zoomed.shape[0]

    slice_ratio = 0.35
    i = step-5 # get range from -5 to + 5
    slice_start = math.floor(computed_height*slice_ratio - math.floor(DEST_WIDTH/2) + DEST_WIDTH*i)
    slice_end = slice_start + DEST_WIDTH

    print(slice_start)
    print(slice_end)
    indecies = list(range(0, slice_start)) + list(range(slice_end, computed_height)) # get only the middle

    cropped = np.delete(zoomed, indecies, axis=0)

    #alter data

    if step > 0:
        print ("applying distortion...")

    #flip horizontally
    if step%4 == 0:
        cropped = np.flipud(cropped)

    #flip vertically
    if step%2 == 0:
        cropped = np.fliplr(cropped)

    #add noise
    if step%3 == 0:
        cropped = np.flipud(cropped)
        cropped = np.fliplr(cropped)

    print(cropped.shape)

    return cropped

def reshapeImage3d(image):

    DEST_WIDTH = 128 #256
    DEST_HEIGHT = 128 #256
    DEST_DEPTH = 80 #166

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
