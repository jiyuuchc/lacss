import numpy as np
from skimage import filters
import matplotlib.pyplot as plt
import matplotlib.patches

'''
these are numpy functions
'''

def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', edge_color=None)
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=3)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image

def draw_label(coordinates, patches, image_shape):
    '''
    Args:
      coordinates: [N, 2], coordinats of patches
      patches: [N, d0, d1] float32
      image_shape: a tuple of (height, width)
    return:
      image: filled with labels
    '''

    d0,d1 = patches.shape[1:3]
    h,w = image_shape
    image = np.zeros([h,w], dtype=int)

    valid_locations = np.logical_and(
        np.logical_and(coordinates[:,0] >= 0, coordinates[:,0] < h),
        np.logical_and(coordinates[:,1] >= 0, coordinates[:,1] < w)
    )
    coordinates = coordinates[valid_locations]
    patches = patches[valid_locations]

    #pad image to allow patches at borders
    padding_size = max(d0, d1) // 2 + 1
    image = np.pad(image, padding_size)
    coordinates += padding_size

    max_prob = np.zeros(image.shape, dtype=patches.dtype)

    # find the max prob at each pixel
    # we have to go through patchs sequentially to have a predictable execution order
    for coord, pred in zip(coordinates, patches):
        c0,c1 = list(coord)
        c0 = int(c0 - d0 / 2)
        c1 = int(c1 - d1 / 2)
        max_prob[c0:c0+d0,c1:c1+d1] = np.maximum(max_prob[c0:c0+d0,c1:c1+d1], pred)

    label = 1
    # now remove any pred output that is not the max
    for coord, pred in zip(coordinates, patches):
        c0,c1 = list(coord)
        c0 = int(c0 - d0 / 2)
        c1 = int(c1 - d1 / 2)
        pred[pred < max_prob[c0:c0+d0,c1:c1+d1]] = 0
        image[c0:c0+d0, c1:c1+d1] = np.maximum(image[c0:c0+d0, c1:c1+d1], (pred > 0.5) * label)
        label += 1

    # remove padding to recover original size
    image = image[padding_size:padding_size+h, padding_size:padding_size+w]

    return image

def draw_border(data, model, image):
    '''
    Args:
      coordinates: [N, 2], coordinats of patches
      patches: [N, H, W]
      image: a tuple of (height, width) or a 2D int array to be drawn on
    Returns:
      image: where instance borders were drawn
    '''
    #coords = data.coordinates
    #preds = tf.sigmoid(model(data.patches)).numpy().squeeze()
    #d0,d1 = preds.shape[-2:]

    d0,d1 = patches.shape[1:3]
    if type(image) is tuple:
        h,w = image
        image = np.zeros([h,w], dtype=int)
    else:
        h,w = image.shape

    # add a border so that erosion operation always work
    patches = np.pad(patches, ((0,0),(1,1),(1,1)))

    #pad image to allow patches at borders
    padding_size = max(d0, d1) // 2 + 1
    image = np.pad(image, padding_size)
    coordinats += padding_size

    for coord, pred in zip(coordinates, patches):
        c0,c1 = list(coord)
        c0 = int(c0 - d0 / 2)
        c1 = int(c1 - d1 / 2)

        pred = (pred > 0.5).astype(np.uint8)
        edge = pred - binary_erosion(pred)
        edge = edge[1:-1,1:-1] # remove extra border added above
        image[c0:c0+d0,c1:c1+d1] += edge

    # remove padding to recover original size
    image = image[padding_size:padding_size+h, padding_size:padding_size+w]

    return image

def visualize_locations(locations, shape):
    h,w = shape
    locations = locations.round().astype(int)
    mask = np.all((locations >= 0) & (locations < [h,w]), axis=-1)
    locations = locations[mask].transpose()
    loc_img = np.zeros(shape=shape, dtype=float)
    loc_img[tuple(locations)] = 1.0
    loc_img = filters.gaussian(loc_img, 3)
    loc_img = loc_img / loc_img.max()
    return loc_img
