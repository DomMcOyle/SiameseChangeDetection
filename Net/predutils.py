import config
import numpy as np
from collections import Counter
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
import matplotlib.colors as pltc
import tensorflow as tf

# main imports
import configparser
import dataprocessing as dp
import siamese as s
import sklearn.metrics as skm
import pickle

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


def pseudo_labels(first_img, second_img, dist_function, return_distances=False):
    """
    Function generating the pseudo labels for a given image pair. The pseudo labels are generated by applying
    the score function directly on the pair and then using otsu thresholding.
    :param first_img: the first images of the pair. It is a 2-dim array of shape (height x width, values), generally
                      a slice of the output of dataprocessing.preprocessing with keep_dims=True
    :param second_img: the first images of the pair. It is a 2-dim array of shape (height x width, values), generally
                      a slice of the output of dataprocessing.preprocessing with keep_dims=True
    :param dist_function: The function to be used for distance computation. It can be SAM or euclidean_dist
    :param return_distances: A boolean flag indicating whether to return distances (True) or labels (False)
    :return: a map of pseudo labels as a 1-dim array with shape (height x width) and the threshold used
    """

    img_a = tf.constant(first_img)
    img_b = tf.constant(second_img)
    distances = dist_function((img_a, img_b)).numpy()
    threshold = threshold_otsu(distances)
    if return_distances is True:
        returned_map = distances
    else:
        returned_map = np.where(distances > threshold, config.CHANGED_LABEL, config.UNCHANGED_LABEL)
    return returned_map, threshold


"""
    script for pseudolabels generation without the minmaxscaling
"""
if __name__ == '__main__':
    dataset = "BAY AREA"
    dist_func = s.SAM

    if dist_func is not s.euclidean_dist and dist_func is not s.SAM:
        raise NotImplementedError("Error: DISTANCE FUNCTION NOT IMPLEMENTED")

    parser = configparser.ConfigParser()
    parser.read(config.DATA_CONFIG_PATH)

    img_a, img_b, labels, names = dp.load_dataset(dataset, parser)
    processed_ab, processed_lab = dp.preprocessing(img_a, img_b, labels, parser[dataset],
                                                   keep_unlabeled=True, apply_rescaling=False)
    i = 0
    for lab in labels:
        pro_a = processed_ab[i:i+lab.size, 0]
        pro_b = processed_ab[i:i+lab.size, 1]
        pro_lab = processed_lab[i:i+lab.size]
        dist, thresh = pseudo_labels(pro_a, pro_b, dist_func, return_distances=True)

        print("Info: SAVING DISTANCES OF " + names[i] + " " + str(i+1) + "/" + str(len(labels)))
        dist_file = open(parser[dataset].get("pseudoPath") + "/" + names[i] + ".pickle", "wb")
        pickle.dump({'threshold': thresh, 'distances': dist}, dist_file, pickle.HIGHEST_PROTOCOL)
        dist_file.close()

        pseudo = np.where(dist > thresh, config.CHANGED_LABEL, config.UNCHANGED_LABEL)

        cm = skm.confusion_matrix(pro_lab, pseudo, labels=[config.CHANGED_LABEL, config.UNCHANGED_LABEL])

        metrics = s.get_metrics(cm)

        file = open(config.STAT_PATH + dataset + "_" + names[i] + "_" + dist_func.__name__ + "_pseudo_noscaling.csv",
                    "w")
        # printing columns names
        file.write("total_examples")
        for k in metrics.keys():
            file.write(", " + k)
        file.write(", threshold")
        file.write("\n" + str(len(pro_lab)))

        # printing metrics
        for k in metrics.keys():
            file.write(", " + str(metrics[k]))
        file.write(", " + str(thresh))
        file.write("\n" + str(len(pro_lab)))

        # saving the map plot
        lmap = np.reshape(pseudo, lab.shape)
        ground_t = dp.refactor_labels(lab, parser[dataset])
        fig = plot_maps(lmap, ground_t)
        fig.savefig(config.STAT_PATH + dataset+ "_" + names[i] + "_" + dist_func.__name__ + "_pseudo_noscaling.png",
                    dpi=300, bbox_inches='tight')

        # spacial correction + metrics + map
        corrected_map = spatial_correction(lmap)
        sccm = skm.confusion_matrix(pro_lab, corrected_map.ravel(),
                                    labels=[config.CHANGED_LABEL, config.UNCHANGED_LABEL])
        scmetrics = s.get_metrics(sccm)

        for k in scmetrics.keys():
            file.write(", " + str(scmetrics[k]))
        file.write(", " + str(thresh))
        file.write("\n")
        file.close()
        scfig = plot_maps(corrected_map, ground_t)
        scfig.savefig(config.STAT_PATH + dataset + "_" + names[i] + "_" + dist_func.__name__ +
                      "_pseudo_noscaling_corrected.png", dpi=300, bbox_inches='tight')

        i = i+lab.size
