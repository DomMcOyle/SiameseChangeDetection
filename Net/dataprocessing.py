import sklearn.preprocessing as sp
import numpy as np
import scipy.io as sio
import configparser
import config


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


def refactor_labels(labels):
    """
    TODO: passare in input file di configurazione per il mapping
    Refactors the label of the AVIRIS dataset setting "1" as changed, "0" as not-changed (substituting "2")
    and "-1" (substituting "0") as unknown
    """
    labels = np.reshape(labels, labels.shape[0]*labels.shape[1])
    return np.where(labels == 0, -1, 2-labels)


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


def load_dataset(name, config):
    """
    function loading a two satellite multi-spectral or hyper-spectral images as 3-dim numpy arrays of shape
    (height, width, spectral bands) and the respective pixel-wise labels as a 2-dim numpy array (height, width)
    :param name: the name of the dataset to be loaded. If it doesn't exist, an exception is raised
    :param config: a config parser instance pre-loaded
    :return: a list containing
        - the first image
        - the second image
        - the labels
    """
    if name not in config.sections():
        raise ValueError(name + " dataset not available")
    print("Info: LOADING FIRST IMAGE...")
    imgA = read_mat(config[name].get("imgAPath"), config[name].get["matLabel"])

    print("Info: LOADING SECOND IMAGE...")
    imgB = read_mat(config[name].get("imgBPath"), config[name].get["matLabel"])

    print("Info: LOADING LABELS...")
    label = read_mat(config[name].get("labelPath"), config[name].get["matLabel"])

    return imgA, imgB, label


def preprocessing(imgA, imgB, label, name, config):
    """

    :param imgA:
    :param imgB:
    :param label:
    :param name:
    :param config:
    :return:
    """
    """
    reshape delle immagini
    refactor delle label
    generazione delle coppie
    minmaxing
    """
