import sklearn.preprocessing as sp
import numpy as np
import scipy.io as sio


"""
function returning an image
read from a .mat file in "path"
"""
def readmat(path, label="HypeRvieW"):
    mat = sio.loadmat(path)
    return mat[label]


"""
function rescaling the pair of images with min-max scaling
it assumes img1 and img2 are same-sized 2-dim arrays with shape (height x width,spectral bands)
"""
def minmaxpair(img1, img2):
    # concat
    toscale = np.concatenate((img1, img2), axis=0)
    scaler = sp.MinMaxScaler()
    # fitting and transforming
    scaler.fit(toscale)
    scaled = scaler.transform(toscale)
    # re-splitting images
    return np.split(scaled, 2)

"""
Refactors the label of the AVIRIS dataset setting "1" as changed, "0" as not-changed (substituting "2") 
and "-1" (substituting "0") as unknown
"""
def refactorLabels(labels):
    labels = np.reshape(labels, labels.shape[0]*labels.shape[1])
    return np.where(labels == 0, -1, 2-labels)


def generateSet(img1, img2, labels, unknown_label):
    pairset = []
    # TODO: check if there's a more efficient method
    for i in range(0, img1.shape[0]):
        if labels[i] != unknown_label :
            pairset.append(img1[i])
            pairset.append(img2[i])
            pairset.append(labels[i])
    return pairset
