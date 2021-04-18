import sklearn.preprocessing as sp
import numpy as np
import scipy.io as sio
import config as cfg

def read_mat(path, label="HypeRvieW"):
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
    Refactors the label of the AVIRIS dataset setting "1" as changed, "0" as not-changed (substituting "2")
    and "-1" (substituting "0") as unknown
    """
    labels = np.reshape(labels, labels.shape[0]*labels.shape[1])
    return np.where(labels == 0, -1, 2-labels)


def generate_set(img1, img2, labels, unknown_label):
    pair_set = []
    label_list = []
    # TODO: check if there's a more efficient method
    for i in range(0, img1.shape[0]):
        if labels[i] != unknown_label:
            pair_set.append([img1[i], img2[i]])
            label_list.append(labels[i])
    return np.asarray(pair_set), np.asarray(label_list)


def load_aviris_dataset():
    """
    function loading the AVIRIS dataset stored in the path hardcoded in the "config.py" module
    :return: a list containing:
        - the pairs of scaled pixel from the Bay Area images
        - the labels for the pixel pairs from the Bay Area images
        - the pairs of scaled pixel from the Santa Barbara images
        - the labels for the pixel pairs from the Santa Barbara images
    """
    ba1 = read_mat(cfg.BAYAREA_A_PATH)
    ba2 = read_mat(cfg.BAYAREA_B_PATH)
    bal = read_mat(cfg.BAYAREA_LABEL_PATH)

    # linearization
    ba1 = np.reshape(ba1, (ba1.shape[0] * ba1.shape[1], ba1.shape[2]))
    ba2 = np.reshape(ba2, (ba2.shape[0] * ba2.shape[1], ba2.shape[2]))

    # minmax scaling
    ba1, ba2 = minmax_pair(ba1, ba2)

    # generating the set
    ba_pairs, ba_labels = generate_set(ba1, ba2, refactor_labels(bal), -1)

    sb1 = read_mat(cfg.SBARBARA_A_PATH)
    sb2 = read_mat(cfg.SBARBARA_B_PATH)
    sbl = read_mat(cfg.SBARBARA_LABEL_PATH)

    # linearization

    sb1 = np.reshape(sb1, (sb1.shape[0] * sb1.shape[1], sb1.shape[2]))
    sb2 = np.reshape(sb2, (sb2.shape[0] * sb2.shape[1], sb2.shape[2]))

    # minmax scaling
    sb1, sb2 = minmax_pair(sb1, sb2)

    #generating the set
    sb_pairs, sb_labels = generate_set(sb1, sb2, refactor_labels(sbl), -1)

    return ba_pairs, ba_labels, sb_pairs, sb_labels
