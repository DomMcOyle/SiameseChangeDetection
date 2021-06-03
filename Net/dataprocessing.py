import sklearn.preprocessing as sp
import numpy as np
import configparser
import config
import scipy.io as sio
import PIL
import os


def read_mat(path, label):
    """
    function returning an image
    read from a .mat file in "path"
    :param path: string containing the path of the image
    :param label: label corresponding to the image in che .mat dictionary
    :return: a 3d numpy array containing the image
    """
    mat = sio.loadmat(path)
    return mat[label]


def minmax_pair(img1, img2):
    """
    function rescaling the pair of images with min-max scaling
    it assumes img1 and img2 are same-sized 2-dim arrays with shape (height x width,spectral bands)
    :param img1: 2-dim numpy array containing pixels of the first images
    :param img2: 2-dim numpy array containing pixels of the second images
    :return: two 2-dim numpy array containing the rescaled pixel values
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


def generate_set(img1, img2, labels, keep_unlabeled):
    """
    Takes the two images, the respective labels and the value of the label for the unlabeled pixels and returns
    a 2-dim array containing the pair of pixel before/after and a 1-dim array containing the label for the pair.
    The unlabeled pixels (marked as UNKNOWN_LABEL) can be removed, if necessary
    :param img1: a 2-dim numpy array of shape (height x width, spectral bands)
    :param img2: a 2-dim numpy array of shape (height x width, spectral bands) - the same shape as img1
    :param labels: a 1-dim numpy array containing a label for each pair
    :param keep_unlabeled: a Boolean flag. If True, keeps all the unlabeled pair of pixels (for testing purposes)
    :return: a 3-dim numpy array of pixel pairs of shape (height x width, 2, spectral bands) and the "labels" array
            without "unknown" labels
    """
    pair_list = []
    label_list = []
    # TODO: check if there's a more efficient method
    for i in range(0, img1.shape[0]):
        if keep_unlabeled or labels[i] != config.UNKNOWN_LABEL:
            pair_list.append([img1[i], img2[i]])
            label_list.append(labels[i])
    return np.asarray(pair_list), np.asarray(label_list)


def load_dataset(name, conf):
    """
    function loading a two satellite multi-spectral or hyper-spectral images as 3-dim numpy arrays of shape
    (height, width, spectral bands) and the respective pixel-wise labels as a 2-dim numpy array (height, width).
    the dataset must be stored in three different directories (before images, after images and labels) and each
    triplet of files (or directories) must have the same name.
    it also checks whether the dataset is available or not.
    :param name: the name of the dataset to be loaded. If it doesn't exist, an exception is raised
    :param conf: a config parser instance pre-loaded
    :return: a list containing
        - the first image
        - the second image
        - the labels
    """
    if name not in conf.sections():
        raise ValueError(name + " dataset not available")

    print("Info: LOADING DATASET " + name + "...")
    imgAList = []
    beforepath = conf[name].get("imgAPath")

    imgBList = []
    afterpath = conf[name].get("imgBPath")

    labellist = []
    labelpath = conf[name].get("labelPath")
    i = 0
    for file in os.listdir(beforepath):
        imgAList.append(load_image(beforepath + os.sep + file, conf[name]))
        imgBList.append(load_image(afterpath + os.sep + file, conf[name]))
        labellist.append(load_label(labelpath + os.sep + file, conf[name]))
        print(str(1)+"/"+str(len(os.listdir(beforepath))) + " pair(s) loaded")
        i = i + 1

    return imgAList, imgBList, labellist


def load_image(path, conf_section):
    if ".mat" in path:
        return read_mat(path, conf_section.get("matLabel"))
    else:
        raise NotImplementedError("Error: CANNOT LOAD NON-MAT FILES")
        #TODO: implementare la compressione dei .tif


def load_label(path, conf_section):
    if ".mat" in path:

        return read_mat(path, conf_section.get("matLabel"))
    elif ".png" or ".tif" in path:
        #TODO: eseguire controllo e compressione dei png a monocanale
        return PIL.Image.open(path)

    else:
        raise NotImplementedError("Error: CANNOT LOAD LABEL FILE FORMAT")


def preprocessing(limgA, limgB, llabel, conf_section, keep_unlabeled):
    """
    Function that takes in input a pair of images and a pixel-wise label map and returns
    an array of minmaxscaled pixel pairs and an array of refactored labels.
    All the pixel pairs labeled as "unknown" can be discarded
    :param limgA: a list of 3-dim numpy array of shape (height, width, spectral bands) from where the first pixel of the pair
                will be extracted
    :param limgB: a list of 3-dim numpy array of shape (height, width, spectral bands) from where the second pixel of the pair
                will be extracted
    :param llabel: a list of 2-dim array of shape (height, width) containing the label for each pixel pair.
    :param conf_section: a config parser section instance containing info obout the dataset
    :param keep_unlabeled: a Boolean flag. If True, keeps all the unlabeled pair of pixels (for testing purposes)
    :return:a list containing:
            - an array containing the labeled pairs of pixels from the images
            - an array containing the labels of the respective pairs
    """

    print("Info: STARTING PREPROCESSING...")
    # loading and linearization of the "A-type" images of the dataset
    imgA = np.empty(shape=(0, limgA[0].shape[2]))
    for img in limgA:
        imgr = np.reshape(img, (img.shape[0] * img.shape[1], img.shape[2]))
        imgA = np.append(imgA, imgr, axis=0)

    # loading and linearization of the "B-type" images of the dataset
    imgB = np.empty(shape=(0, limgB[0].shape[2]))
    for img in limgB:
        imgr = np.reshape(img, (img.shape[0] * img.shape[1], img.shape[2]))
        imgB = np.append(imgB, imgr, axis=0)

    # min maxing
    # imgA, imgB = minmax_pair(imgA, imgB)


    # linearization and refactoring of the labels
    label = np.empty(shape=(0))
    for l in llabel:
        lr = np.reshape(l, l.shape[0] * l.shape[1])
        label = np.append(label, refactor_labels(lr, conf_section))

    # pair generation
    print("Info: STARTING PAIRING PROCEDURE...")
    pairs, label = generate_set(imgA, imgB, label, keep_unlabeled)

    return pairs, label