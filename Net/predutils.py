import config
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.colors as pltc


def spatial_correction(prediction, radius=3):
    """
    function returning a copy of the prediction map with spacial correction. Each pixel is resampled
    with the most frequent class in a kernel surrounding the pixel-
    :param prediction: a 2-dim array containing the predicted classes
    :param radius: a positive integer indicating the radius of the "kernel", including the central pixel
                   (Default =3 => 5x5 kernel)
    :return: a copy of prediction with corrected labels
    """
    corrected = np.zeros(prediction.shape)
    max_r, max_c = prediction.shape
    for row in range(max_r):
        for col in range(max_c):
            upper_x = max(0, row - (radius - 1))
            upper_y = max(0, col - (radius - 1))
            # note: the lower bound for the moving "kernel" must be one unit greater for each coordinate than the
            # actual lower bound, since it will be discarded as the last index for the slices
            lower_x = min(max_r, row + radius)
            lower_y = min(max_c, col + radius)
            counter = Counter(prediction[upper_x:lower_x, upper_y:lower_y].ravel())
            counts = counter.most_common()
            if len(counts) > 1 and counts[0][1] == counts[1][1]:
                corrected[row, col] = prediction[row, col]
            else:
                corrected[row, col] = counts[0][0]
    return corrected


def plot_maps(prediction, label_map):
    """
    function plotting the original label map put beside the predicted label map
    :param prediction: the 2-dim array of shape (height x width) of predicted classes
    :param label_map: the 2-dim array of shape (height x width) loaded with the dataset
    :return: the plot of the two label map and the whole prediction
    """

    new_map = np.copy(prediction)
    replace_indexes = np.where(label_map == config.UNKNOWN_LABEL)
    new_map[replace_indexes] = config.UNKNOWN_LABEL

    cmap = pltc.ListedColormap(config.COLOR_MAP)
    fig = plt.figure(figsize=(16, 9))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)
    ax1.imshow(prediction, cmap=cmap, vmin=0, vmax=2)
    ax1.title.set_text("Total prediction")

    ax2.imshow(new_map, cmap=cmap, vmin=0, vmax=2)
    ax2.title.set_text("Comparable Prediction")

    ax3.imshow(label_map, cmap=cmap, vmin=0, vmax=2)
    ax3.title.set_text("Ground truth")
    plt.show()
    return fig
