import config
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as pltc


def spatial_correction(prediction):
    """

    :param prediction:
    :return:
    """
    

def plot_maps(prediction, label_map):
    """
    function plotting the original label map put beside the predicted label map
    :param prediction: the 1-dim array of predicted classes
    :param label_map: the 2-dim array of shape (height x width) loaded with the dataset
    :return: the plot of the two label map and the whole prediction
    """
    #TODO: spostare il reshape fuori
    predicted_map = np.reshape(prediction, label_map.shape)

    new_map = np.copy(predicted_map)
    replace_indexes = np.where(label_map == config.UNKNOWN_LABEL)
    new_map[replace_indexes] = config.UNKNOWN_LABEL

    cmap = pltc.ListedColormap(config.COLOR_MAP)
    fig = plt.figure(figsize=(16, 9))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)
    ax1.imshow(predicted_map, cmap=cmap, vmin=0, vmax=2)
    ax1.title.set_text("Total prediction")

    ax2.imshow(new_map, cmap=cmap, vmin=0, vmax=2)
    ax2.title.set_text("Comparable Prediction")

    ax3.imshow(label_map, cmap=cmap, vmin=0, vmax=2)
    ax3.title.set_text("Ground truth")
    plt.show()
    return fig
