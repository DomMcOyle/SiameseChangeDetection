import sklearn.preprocessing as sp
import numpy as np
import configparser
import config
import scipy.io as sio
import PIL


def read_mat(path, label):
    """
    function returning an image
    read from a .mat file in "path"
    """
    mat = sio.loadmat(path)
    return mat[label]


def minmax_pair(img1, img2):
    """
    function rescaling the pair of images with min-max scaling
    it assumes img1 and img2 are same-sized 2-dim arrays with shape (height x width,spectral bands)
    """
    # concat
    toscale = np.concatenate((img1, img2), axis=0)
    scaler = sp.MinMaxScaler()
    # fitting and transforming
    scaler.fit(toscale)
    scaled = scaler.transform(toscale)
    # re-splitting images
    return np.split(scaled, 2)


def refactor_labels(labels, conf_section):
    """
    Refactors the label of a dataset setting "0" as changed, "1" as not-changed and "2" as unknown.
    this helps to have a common symobol code for the labels
    :param labels: a numpy integer array containing the labels for a image pair
    :param conf_section: a ConfigParser Section containing the label specific values
    :return: the input labels refactored as described
    """
    # flag indicating whether a label must be changed
    c_set = False
    uc_set = False
    un_set = False

    if int(conf_section.get("changedLabel")) != config.CHANGED_LABEL:
        c_indexes = np.where(labels == int(conf_section.get("changedLabel")))
        c_set = True

    if int(conf_section.get("unchangedLabel")) != config.UNCHANGED_LABEL:
        uc_indexes = np.where(labels == int(conf_section.get("unchangedLabel")))
        uc_set = True

    if int(conf_section.get("unknownLabel")) != config.UNKNOWN_LABEL:
        un_indexes = np.where(labels == int(conf_section.get("unknownLabel")))
        un_set = True

    if c_set:
        labels[c_indexes] = config.CHANGED_LABEL
    if uc_set:
        labels[uc_indexes] = config.UNCHANGED_LABEL
    if un_set:
        labels[un_indexes] = config.UNKNOWN_LABEL

    return labels


def generate_set(img1, img2, labels, unknown_label):
    """
    Takes the two images, the respective labels and the value of the label for the unlabeled pixels and returns
    a 2-dim array containing the pair of pixel before/after and a 1-dim array containing the label for the pair.
    Every unlabeled pixel pair is removed.
    :param img1: a 2-dim numpy array of shape (height x width, spectral bands)
    :param img2: a 2-dim numpy array of shape (height x width, spectral bands) - the same shape as img1
    :param labels: a 1-dim numpy array containing a label for each pair
    :param unknown_label: the value of the "unlabeled pixels" label
    :return: a 3-dim numpy array of pixel pairs of shape (height x width, 2, spectral bands) and the "labels" array
            without "unknown" labels
    """
    pair_list = []
    label_list = []
    # TODO: check if there's a more efficient method
    for i in range(0, img1.shape[0]):
        if labels[i] != unknown_label:
            pair_list.append([img1[i], img2[i]])
            label_list.append(labels[i])
    return np.asarray(pair_list), np.asarray(label_list)


def load_dataset(name, conf):
    """
    function loading a two satellite multi-spectral or hyper-spectral images as 3-dim numpy arrays of shape
    (height, width, spectral bands) and the respective pixel-wise labels as a 2-dim numpy array (height, width)
    it also checks whether the dataset is available or not
    :param name: the name of the dataset to be loaded. If it doesn't exist, an exception is raised
    :param conf: a config parser instance pre-loaded
    :return: a list containing
        - the first image
        - the second image
        - the labels
    """
    if name not in conf.sections():
        raise ValueError(name + " dataset not available")
    if ".mat" in conf[name].get("imgAPath"):
        print("Info: LOADING FIRST IMAGE...")
        imgA = read_mat(conf[name].get("imgAPath"), conf[name].get("matLabel"))

        print("Info: LOADING SECOND IMAGE...")
        imgB = read_mat(conf[name].get("imgBPath"), conf[name].get("matLabel"))
    else:
        raise NotImplementedError("Error: CANNOT LOAD NON-MAT FILES")

    print("Info: LOADING LABELS...")
    if ".mat" in conf[name].get("labelPath"):

        label = read_mat(conf[name].get("labelPath"), conf[name].get("matLabel"))
    elif ".png" or ".tif" in conf[name].get("labelPath"):

        label = PIL.Image.open(conf[name].get("labelPath"))
    else:
        raise NotImplementedError("Error: CANNOT LOAD LABEL FILE FORMAT")
    return imgA, imgB, label


def preprocessing(imgA, imgB, label, conf_section):
    """
    Function that takes in input a pair of images and a pixel-wise label map and returns
    an array of minmaxscaled pixel pairs and an array of refactored labels.
    All the pixel pairs labeled as "unknown" are discarded
    :param imgA: a 3-dim numpy array of shape (height, width, spectral bands) from where the first pixel of the pair
                will be extracted
    :param imgB: a 3-dim numpy array of shape (height, width, spectral bands) from where the second pixel of the pair
                will be extracted
    :param label: a 2-dim array of shape (height, width) containing the label for each pixel pair. The array must be
                refactored as described in (refactor_labels()) before use
    :param conf_section: a config parser section instance containing info obout the dataset
    :return:a list containing:
            - an array containing the labeled pairs of pixels from the images
            - an array containing the labels of the respective pairs
    """

    print("Info: STARTING PREPROCESSING...")
    # linearization
    imgA = np.reshape(imgA, (imgA.shape[0] * imgA.shape[1], imgA.shape[2]))
    imgB = np.reshape(imgB, (imgB.shape[0] * imgB.shape[1], imgB.shape[2]))

    # min maxing

    imgA, imgB = minmax_pair(imgA, imgB)

    # linearization and refactoring of the labels
    label = refactor_labels(label, conf_section)
    label = np.reshape(label, label.shape[0] * label.shape[1])

    # pair generation
    print("Info: STARTING PAIRING PROCEDURE...")
    pairs, label = generate_set(imgA, imgB, label, config.UNKNOWN_LABEL)
    return pairs, label
